# model.py

import torch
import torch.nn as nn
import copy

class EvolvingNN(nn.Module):
    """
    A neural network that can evolve its architecture by adding or removing layers
    while attempting to preserve weights from previous training.
    """
    def __init__(self, input_size=784, hidden_layers=None, output_size=10):
        super(EvolvingNN, self).__init__()
        if hidden_layers is None:
            hidden_layers = [128]
            
        self.input_size = input_size
        self.hidden_layers = list(hidden_layers)
        self.output_size = output_size
        
        self.layers = nn.ModuleList()
        self._build_initial_model()

    def _build_initial_model(self):
        """Constructs the initial layer stack."""
        in_features = self.input_size
        
        for hidden in self.hidden_layers:
            self.layers.append(nn.Linear(in_features, hidden))
            in_features = hidden
            
        self.output_layer = nn.Linear(in_features, self.output_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for i, layer in enumerate(self.layers):
            x = self.activation(layer(x))
        return self.output_layer(x)

    def add_layer(self, new_size=None):
        """
        Adds a new hidden layer at the end of the current hidden layers.
        Preserves existing weights.
        """
        if new_size is None:
            # Default to the size of the last hidden layer or 64
            new_size = self.hidden_layers[-1] if self.hidden_layers else 64
            
        # 1. Store old output layer to potentially reuse/bridge
        # In this simple version, we'll just replace it, but we preserve the hidden layers.
        
        last_hidden_size = self.hidden_layers[-1]
        
        # Add new linear layer
        new_layer = nn.Linear(last_hidden_size, new_size)
        
        # Initialize new layer (standard initialization)
        nn.init.kaiming_normal_(new_layer.weight, nonlinearity='relu')
        nn.init.constant_(new_layer.bias, 0)
        
        self.layers.append(new_layer)
        self.hidden_layers.append(new_size)
        
        # Update output layer to match new last hidden layer size
        old_output_weight = self.output_layer.weight.data
        old_output_bias = self.output_layer.bias.data
        
        self.output_layer = nn.Linear(new_size, self.output_size)
        
        # If new_size == last_hidden_size, we could theoretically copy some output weights,
        # but since the input features to the output layer changed completely (new layer),
        # it's better to let it relearn.
        
    def grow_layer(self, layer_idx, increase_by=32):
        """Increases the number of neurons in a specific hidden layer."""
        if layer_idx >= len(self.layers):
            return
            
        old_layer = self.layers[layer_idx]
        old_in = old_layer.in_features
        old_out = old_layer.out_features
        new_out = old_out + increase_by
        
        new_layer = nn.Linear(old_in, new_out)
        with torch.no_grad():
            new_layer.weight[:old_out, :] = old_layer.weight
            new_layer.bias[:old_out] = old_layer.bias
            
        self.layers[layer_idx] = new_layer
        self.hidden_layers[layer_idx] = new_out
        
        if layer_idx + 1 < len(self.layers):
            next_layer = self.layers[layer_idx + 1]
            new_next = nn.Linear(new_out, next_layer.out_features)
            with torch.no_grad():
                new_next.weight[:, :old_out] = next_layer.weight
                new_next.bias = next_layer.bias
            self.layers[layer_idx + 1] = new_next
        else:
            new_output = nn.Linear(new_out, self.output_size)
            with torch.no_grad():
                new_output.weight[:, :old_out] = self.output_layer.weight
                new_output.bias = self.output_layer.bias
            self.output_layer = new_output

    def shrink_layer(self, layer_idx, decrease_by=32):
        """Decreases neurons in a layer (Pruning)."""
        if layer_idx >= len(self.layers):
            return
            
        old_layer = self.layers[layer_idx]
        old_in = old_layer.in_features
        old_out = old_layer.out_features
        new_out = max(8, old_out - decrease_by) # Keep at least 8 neurons
        
        if new_out == old_out:
            return

        new_layer = nn.Linear(old_in, new_out)
        with torch.no_grad():
            new_layer.weight.copy_(old_layer.weight[:new_out, :])
            new_layer.bias.copy_(old_layer.bias[:new_out])
            
        self.layers[layer_idx] = new_layer
        self.hidden_layers[layer_idx] = new_out
        
        if layer_idx + 1 < len(self.layers):
            next_layer = self.layers[layer_idx + 1]
            new_next = nn.Linear(new_out, next_layer.out_features)
            with torch.no_grad():
                new_next.weight.copy_(next_layer.weight[:, :new_out])
                new_next.bias.copy_(next_layer.bias)
            self.layers[layer_idx + 1] = new_next
        else:
            new_output = nn.Linear(new_out, self.output_size)
            with torch.no_grad():
                new_output.weight.copy_(self.output_layer.weight[:, :new_out])
                new_output.bias.copy_(self.output_layer.bias)
            self.output_layer = new_output

    def remove_layer(self):
        """Removes the last hidden layer if the model is too deep."""
        if len(self.layers) <= 1:
            return # Keep at least one hidden layer
            
        removed_layer = self.layers.pop()
        self.hidden_layers.pop()
        
        # New last layer now connects directly to output
        new_last_hidden_size = self.hidden_layers[-1]
        self.output_layer = nn.Linear(new_last_hidden_size, self.output_size)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

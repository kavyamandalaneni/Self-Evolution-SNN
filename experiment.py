# experiment.py

import torch
import torch.nn as nn
import torch.optim as optim
from trainer import train, evaluate
from model import EvolvingNN
from mutation import should_mutate
import random

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def run_experiment(train_loader, test_loader, input_size, output_size, 
                   model_type="evolving", epochs=15, 
                   log_callback=None, progress_callback=None):
    """
    Orchestrates the training and evolution process.
    """
    device = get_device()
    msg = f"Using device: {device}"
    print(msg)
    if log_callback:
        log_callback(msg)

    # Initial Model Configuration with dynamic input/output sizes
    if model_type == "small":
        model = EvolvingNN(input_size=input_size, output_size=output_size, hidden_layers=[32])
    elif model_type == "large":
        model = EvolvingNN(input_size=input_size, output_size=output_size, hidden_layers=[128, 64])
    else:
        model = EvolvingNN(input_size=input_size, output_size=output_size, hidden_layers=[64])

    model.to(device)
    # ... rest of the function ...

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Histories for analysis/visualization
    accuracy_history = []
    loss_history = []
    architecture_history = []
    param_history = []
    mutations = []

    for epoch in range(epochs):
        # 1. Train and Evaluate
        avg_train_loss = train(model, train_loader, optimizer, criterion, device, 
                               progress_callback=progress_callback)
        acc, avg_test_loss = evaluate(model, test_loader, device)
        
        # 2. Update histories
        accuracy_history.append(acc)
        loss_history.append(avg_test_loss)
        architecture_history.append(list(model.hidden_layers))
        param_history.append(model.parameter_count())

        epoch_msg = f"Epoch {epoch+1:02d}/{epochs} | Loss: {avg_train_loss:.4f} | Acc: {acc:.2f}% | Params: {param_history[-1]}"
        print(epoch_msg)
        if log_callback:
            log_callback(epoch_msg)

        # 3. Handle Mutation (only for evolving models)
        if model_type == "evolving":
            is_mutate, strategy = should_mutate(epoch, accuracy_history)
            
            if is_mutate:
                mutate_msg = f"🔄 Mutating Architecture! Strategy: {strategy}"
                print(mutate_msg)
                if log_callback:
                    log_callback(mutate_msg)
                
                if strategy == "add_layer":
                    new_size = max(16, model.hidden_layers[-1] // 2)
                    model.add_layer(new_size)
                elif strategy == "grow_layer":
                    idx = random.randint(0, len(model.layers) - 1)
                    model.grow_layer(idx, increase_by=32)
                elif strategy == "shrink_layer":
                    idx = random.randint(0, len(model.layers) - 1)
                    model.shrink_layer(idx, decrease_by=32)
                elif strategy == "remove_layer":
                    model.remove_layer()
                
                model.to(device)
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                mutations.append(epoch + 1)

    return {
        "accuracy_history": accuracy_history,
        "loss_history": loss_history,
        "architecture_history": architecture_history,
        "parameter_history": param_history,
        "final_architecture": model.hidden_layers,
        "mutations": mutations,
        "parameters": model.parameter_count(),
        "device": str(device)
    }

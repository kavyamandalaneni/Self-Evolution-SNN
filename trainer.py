# trainer.py

import torch
from tqdm import tqdm

def train(model, train_loader, optimizer, criterion, device, progress_callback=None):
    """
    Trains the model for one epoch.
    progress_callback: A function that takes (current_batch, total_batches, current_loss)
    """
    model.train()
    total_loss = 0
    total_batches = len(train_loader)
    
    # Terminal progress bar
    progress_bar = tqdm(train_loader, desc="Training", leave=False)

    for i, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        total_loss += current_loss
        progress_bar.set_postfix(loss=current_loss)
        
        # Update UI if callback is provided
        if progress_callback:
            progress_callback(i + 1, total_batches, current_loss)

    return total_loss / total_batches


def evaluate(model, test_loader, device):
    """
    Evaluates the model on test data.
    Returns: Accuracy percentage and average loss.
    """
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = test_loss / len(test_loader)
    
    return accuracy, avg_loss

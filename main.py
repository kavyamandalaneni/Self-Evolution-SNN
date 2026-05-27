# main.py

import argparse
from data_loader import get_data_loaders
from experiment import run_experiment

def main():
    parser = argparse.ArgumentParser(description="Self-Evolving Neural Network Training")
    parser.add_argument("--dataset", type=str, default="MNIST", 
                        choices=["MNIST", "Fashion-MNIST", "CIFAR-10"],
                        help="Dataset to use for training")
    parser.add_argument("--model", type=str, default="evolving", 
                        choices=["small", "large", "evolving"],
                        help="Type of model to train")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    
    args = parser.parse_args()

    print(f"🚀 Starting Experiment: Dataset={args.dataset}, Model={args.model}, Epochs={args.epochs}")
    
    # Load data and get dynamic sizes
    train_loader, test_loader, input_size, num_classes = get_data_loaders(
        dataset_name=args.dataset, 
        batch_size=args.batch_size
    )
    
    results = run_experiment(
        train_loader=train_loader, 
        test_loader=test_loader, 
        input_size=input_size,
        output_size=num_classes,
        model_type=args.model, 
        epochs=args.epochs
    )

    print("\n✅ Training Complete.")
    print(f"Final Architecture: {results['final_architecture']}")
    print(f"Final Accuracy: {results['accuracy_history'][-1]:.2f}%")
    print(f"Total Parameters: {results['parameters']}")
    if results['mutations']:
        print(f"Mutations occurred at epochs: {results['mutations']}")

if __name__ == "__main__":
    main()

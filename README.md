# Self-Evolving Neural Network Using Performance-Driven Architecture Mutation

## Project Overview
This project implements a neural network that dynamically modifies its own architecture during training. Unlike traditional neural networks with a fixed structure, this model monitors its performance (accuracy) and detects learning plateaus. When a plateau is reached, the network triggers a "mutation" to increase its capacity, either by adding a new layer or expanding an existing one.

### Key Features
- **Dynamic Mutation**: Automatically adds or grows layers based on plateau detection.
- **Weight Preservation**: When the architecture changes, existing weights are preserved, allowing the model to "evolve" rather than restart from scratch.
- **Interactive Dashboard**: A Streamlit-based web interface for real-time monitoring and analysis of the evolution process.
- **Hardware Acceleration**: Automatically detects and uses CUDA or MPS (Apple Silicon) if available.

## Problem Statement
Fixed neural network architectures often face a trade-off: small models may lack the capacity to learn complex patterns, while large models are computationally expensive and prone to overfitting. A self-evolving architecture starts lean and only grows as needed, leading to more efficient learning and optimized final structures.


2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Command Line Interface
You can run a training experiment directly from the CLI:
```bash
python main.py --model evolving --epochs 20
```

### Web Interface (Streamlit)
For a more interactive experience and detailed visualizations:
```bash
streamlit run app.py
```

## Project Structure
- `model.py`: Core `EvolvingNN` class with weight preservation logic.
- `mutation.py`: Logic for detecting plateaus and choosing mutation strategies.
- `trainer.py`: Training and evaluation loops with progress bars.
- `experiment.py`: Orchestration logic for running full training/evolution cycles.
- `app.py`: Streamlit dashboard implementation.
- `data_loader.py`: MNIST data handling.

## Results & Analysis
The system tracks several metrics:
- **Test Accuracy**: Performance over epochs.
- **Model Complexity**: Parameter count growth during mutations.
- **Architecture History**: A log of every architectural change made by the system.

## Future Scope
- **Architecture Pruning**: Implementing logic to remove redundant neurons or layers.
- **Cross-Dataset Validation**: Testing on CIFAR-10 or more complex datasets.
- **Neuroevolution**: Integrating genetic algorithms for more diverse mutation strategies.

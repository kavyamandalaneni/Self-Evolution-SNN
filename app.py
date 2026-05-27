# app.py

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from data_loader import get_data_loaders
from experiment import run_experiment

st.set_page_config(page_title="Self-Evolving NN", page_icon="🧠", layout="wide")

st.title("🧠 Self-Evolving Neural Network Analysis")
st.markdown("""
This project demonstrates a neural network that **dynamically evolves its architecture** during training. 
By monitoring performance plateaus, the system automatically adds or expands layers to overcome learning bottlenecks.
""")

with st.sidebar:
    st.header("Configuration")
    dataset_choice = st.selectbox(
        "Select Dataset",
        ["MNIST", "Fashion-MNIST", "CIFAR-10"],
        help="MNIST: Hand-written digits. Fashion: Clothing. CIFAR-10: Color objects (Slower on CPU)."
    )
    model_choice = st.selectbox(
        "Select Model Type",
        ["evolving", "small", "large"],
        help="'evolving' will mutate during training. 'small' and 'large' are fixed baselines."
    )
    epochs = st.slider("Max Epochs", 5, 30, 15)
    batch_size = st.number_input("Batch Size", 16, 256, 64)

if st.button("🚀 Run Experiment", use_container_width=True):
    # Load dataset-specific data
    train_loader, test_loader, input_size, num_classes = get_data_loaders(
        dataset_name=dataset_choice, 
        batch_size=batch_size
    )

    # UI Placeholders for live updates
    status_container = st.status(f"Initializing Training on {dataset_choice}...", expanded=True)
    with status_container:
        log_placeholder = st.empty()
        progress_bar = st.progress(0, text="Batch progress")
        all_logs = []

        def update_logs(new_msg):
            all_logs.append(new_msg)
            log_placeholder.code("\n".join(all_logs))

        def update_progress(current, total, loss):
            percent = current / total
            if current % 5 == 0 or current == total:
                progress_bar.progress(percent, text=f"Batch {current}/{total} | Loss: {loss:.4f}")

        results = run_experiment(
            train_loader, 
            test_loader, 
            input_size,
            num_classes,
            model_choice, 
            epochs,
            log_callback=update_logs,
            progress_callback=update_progress
        )
    
    status_container.update(label="Training Complete!", state="complete", expanded=False)
    st.success("Analysis Ready!")

    # Top Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Final Accuracy", f"{results['accuracy_history'][-1]:.2f}%")
    col2.metric("Total Parameters", f"{results['parameters']:,}")
    col3.metric("Device Used", results['device'])
    col4.metric("Mutations", len(results['mutations']))

    # Visualizations
    tab1, tab2, tab3 = st.tabs(["📈 Performance", "🏗️ Architecture Evolution", "📊 Raw Data"])

    with tab1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy Plot
        ax1.plot(results["accuracy_history"], marker='o', color='#1f77b4', label='Test Accuracy')
        for m_epoch in results['mutations']:
            ax1.axvline(x=m_epoch-1, color='r', linestyle='--', alpha=0.5, label='Mutation' if m_epoch == results['mutations'][0] else "")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy (%)")
        ax1.set_title("Accuracy Over Epochs")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Parameter Count Plot
        ax2.plot(results["parameter_history"], marker='s', color='#ff7f0e', label='Parameters')
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Parameter Count")
        ax2.set_title("Model Complexity Over Time")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        st.pyplot(fig)

    with tab2:
        st.subheader("Layer Structure Evolution")
        arch_data = []
        for i, arch in enumerate(results["architecture_history"]):
            arch_data.append({"Epoch": i+1, "Layers": str(arch), "Count": len(arch)})
        
        df_arch = pd.DataFrame(arch_data)
        st.table(df_arch)

        # Plotting layer count
        fig_lc, ax_lc = plt.subplots(figsize=(10, 3))
        ax_lc.step(range(1, len(results["architecture_history"]) + 1), 
                   [len(a) for a in results["architecture_history"]], where='post')
        ax_lc.set_xlabel("Epoch")
        ax_lc.set_ylabel("Number of Hidden Layers")
        ax_lc.set_title("Network Depth Growth")
        st.pyplot(fig_lc)

    with tab3:
        st.subheader("Training Logs")
        df_results = pd.DataFrame({
            "Epoch": range(1, len(results["accuracy_history"]) + 1),
            "Accuracy (%)": results["accuracy_history"],
            "Loss": results["loss_history"],
            "Parameters": results["parameter_history"]
        })
        st.dataframe(df_results, use_container_width=True)

    st.subheader("Final Model Architecture Summary")
    st.code(f"Input (784) -> {' -> '.join(map(str, results['final_architecture']))} -> Output (10)")

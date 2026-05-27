# mutation.py

import numpy as np

def should_mutate(epoch, accuracy_history, patience=3, min_delta=0.05, threshold=98.0):
    """
    Determines if the model should undergo mutation.
    """
    if epoch < 3 or len(accuracy_history) < patience:
        return False, None

    # 1. UPWARD EVOLUTION (Adding capacity)
    # Check for plateau below threshold
    if (accuracy_history[-1] - accuracy_history[-patience]) < min_delta:
        if accuracy_history[-1] < threshold:
            if len(accuracy_history) % 2 == 0:
                return True, "add_layer"
            else:
                return True, "grow_layer"
                
    # 2. DOWNWARD EVOLUTION (Optimizing/Pruning)
    # If we are performing exceptionally well (e.g. > 99%), let's try to shrink 
    # to find the most efficient architecture.
    if accuracy_history[-1] > 99.2:
        # Occasionally try to prune to maintain efficiency
        if epoch % 10 == 0: 
            return True, "shrink_layer"

    return False, None

import os
import matplotlib.pyplot as plt

def plot_training_history(history, model_name, output_dir='experiments/logs'):
    """
    Generates and saves a Loss/Accuracy plot for the training run.
    """
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(14, 5))

    # --- Plot 1: Loss ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    plt.title(f'{model_name.upper()} - Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # --- Plot 2: Accuracy ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'g-', label='Train Acc')
    plt.plot(epochs, history['val_acc'], 'y-', label='Val Acc')
    plt.title(f'{model_name.upper()} - Training & Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f'history_{model_name}.png')
    plt.savefig(save_path, dpi=300)
    print(f"[Visualizer] Training history plot saved to {save_path}")
    plt.close()
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, class_names, model_name='model', output_dir='experiments/logs/'):
    """
    Computes and plots a beautiful, publication-ready confusion matrix.
    Dynamically handles model naming for automated benchmarking.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate the raw matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plotting aesthetics
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, 
                yticklabels=class_names,
                cbar=False, linewidths=1, linecolor='black')
    
    # Dynamic Title
    plt.title(f'HAR {model_name.upper()} Confusion Matrix', fontsize=16, pad=20)
    plt.ylabel('True Physical Activity', fontsize=12, labelpad=10)
    plt.xlabel('Predicted Physical Activity', fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Dynamic Filename to prevent overwriting during baseline runs
    save_path = os.path.join(output_dir, f'confusion_matrix_{model_name.lower()}.png')
    plt.savefig(save_path, dpi=300)
    print(f"[Evaluation] Confusion matrix saved to {save_path}")
    plt.close()
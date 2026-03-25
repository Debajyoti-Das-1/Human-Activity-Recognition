import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from captum.attr import IntegratedGradients
from src.models.advanced_hybrid import HAR_AdvancedHybrid
from src.data.dataset import get_dataloaders

def get_sample_data(class_idx, X_test, y_test):
    """Finds the first sample in the test set matching the target class."""
    idx = np.where(y_test == class_idx)[0][0]
    return torch.tensor(X_test[idx], dtype=torch.float32).unsqueeze(0), y_test[idx]

def run_interpretation():
    # 1. Setup Environment
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[Interpret] Running on {device}")

    CLASS_NAMES = ['Walking', 'Walking Upstairs', 'Walking Downstairs', 
                   'Sitting', 'Standing', 'Laying']
    
    # 2. Load the Advanced Hybrid Model
    model = HAR_AdvancedHybrid(num_classes=6, in_channels=9).to(device)
    checkpoint_path = 'experiments/checkpoints/advanced_hybrid/best_model.pth'
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # 3. Load Data
    X_test = np.load('data/processed/X_test.npy')
    y_test = np.load('data/processed/y_test.npy')

    # Let's investigate a "Standing" sample (Index 4)
    # We want to know exactly what makes the model think the person is standing.
    target_class_idx = 4 
    input_tensor, true_label = get_sample_data(target_class_idx, X_test, y_test)
    input_tensor = input_tensor.to(device)

    # 4. Initialize Captum Integrated Gradients
    ig = IntegratedGradients(model)
    
    # Baseline is a zero tensor (no movement) of the same shape
    baseline = torch.zeros_like(input_tensor).to(device)

    print(f"[Interpret] Calculating Axiomatic Attributions for: {CLASS_NAMES[target_class_idx]}...")
    
    # Calculate attributions with 50 approximation steps
    attributions, delta = ig.attribute(
        inputs=input_tensor,
        baselines=baseline,
        target=target_class_idx,
        n_steps=50,
        return_convergence_delta=True
    )

    print(f"[Interpret] Convergence Delta (Completeness Check): {delta.item():.6f}")

    # 5. Process for Visualization
    # Shape goes from [1, 128, 9] -> [9, 128] for heatmap plotting
    attr_np = attributions.squeeze(0).cpu().detach().numpy().T
    input_np = input_tensor.squeeze(0).cpu().detach().numpy().T

    # 6. Plotting the Saliency Heatmap
    plt.figure(figsize=(16, 8))
    
    # We use a diverging colormap: Red = Positive attribution, Blue = Negative attribution
    # Center the colormap at 0
    max_abs = np.max(np.abs(attr_np))
    
    sensor_labels = [
        'Total Acc X', 'Total Acc Y', 'Total Acc Z',
        'Body Acc X', 'Body Acc Y', 'Body Acc Z',
        'Body Gyro X', 'Body Gyro Y', 'Body Gyro Z'
    ]

    sns.heatmap(attr_np, cmap='coolwarm', center=0, vmin=-max_abs, vmax=max_abs,
                yticklabels=sensor_labels, cbar_kws={'label': 'IG Attribution Score'})
    
    plt.title(f"Pareto Attribution Map: What the Model 'Sees' for {CLASS_NAMES[target_class_idx]}", fontsize=16)
    plt.xlabel("Time Ticks (50Hz -> 2.56 seconds total)", fontsize=12)
    plt.ylabel("Sensor Channels", fontsize=12)
    
    plt.tight_layout()
    os.makedirs('experiments/logs/', exist_ok=True)
    plt.savefig(f'experiments/logs/ig_attribution_{CLASS_NAMES[target_class_idx]}.png', dpi=300)
    print(f"[Interpret] Heatmap saved to experiments/logs/ig_attribution_{CLASS_NAMES[target_class_idx]}.png")
    plt.show()

if __name__ == "__main__":
    run_interpretation()
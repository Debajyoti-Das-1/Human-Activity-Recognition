import os
import argparse
import yaml
import torch
import numpy as np
from src.data.dataset import get_dataloaders
from src.models.cnn import HAR_1DCNN
from src.models.lstm import HAR_LSTM
from src.models.cnn_lstm import HAR_CNNLSTM
from src.models.transformer import HAR_Transformer
from src.models.advanced_hybrid import HAR_AdvancedHybrid # <-- NEW IMPORT
from src.evaluation.visualizer import plot_confusion_matrix
from sklearn.metrics import classification_report

def parse_args():
    parser = argparse.ArgumentParser(description="HAR Model Evaluation Engine")
    parser.add_argument(
        '--model', 
        type=str, 
        default='advanced_hybrid', 
        choices=['cnn', 'lstm', 'cnn_lstm', 'transformer', 'advanced_hybrid'], # <-- UPDATED
        help="Select the architecture to evaluate"
    )
    return parser.parse_args()

def load_config(model_name):
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
    path = f"configs/{model_name}.yaml"
    if os.path.exists(path):
        with open(path, "r") as f:
            model_config = yaml.safe_load(f)
            config.update(model_config)
    return config

def evaluate_model():
    args = parse_args()
    config = load_config(args.model)

    device = torch.device("mps" if torch.backends.mps.is_available() else 
                          "cuda" if torch.cuda.is_available() else "cpu")

    CLASS_NAMES = ['Walking', 'Walking Upstairs', 'Walking Downstairs', 
                   'Sitting', 'Standing', 'Laying']

    _, test_loader = get_dataloaders(
        config['data']['train_x'], config['data']['train_y'],
        config['data']['test_x'], config['data']['test_y'],
        batch_size=64
    )

    print(f"[System] Initializing {args.model.upper()} for Evaluation...")
    
    if args.model == 'cnn':
        model = HAR_1DCNN(num_classes=6, in_channels=9)
    elif args.model == 'lstm':
        model = HAR_LSTM(input_dim=9, hidden_dim=config['model'].get('hidden_dim', 64), 
                         num_layers=config['model'].get('num_layers', 2), num_classes=6)
    elif args.model == 'cnn_lstm':
        model = HAR_CNNLSTM(num_classes=6, in_channels=9)
    elif args.model == 'transformer':
        model = HAR_Transformer(
            num_classes=6, in_channels=9, d_model=config['model'].get('d_model', 128),
            nhead=config['model'].get('nhead', 8), num_layers=config['model'].get('num_layers', 4)
        )
    elif args.model == 'advanced_hybrid': # <-- NEW BLOCK
        model = HAR_AdvancedHybrid(
            num_classes=6, in_channels=9,
            cnn_dim=config['model'].get('cnn_dim', 64),
            lstm_hidden=config['model'].get('lstm_hidden', 64),
            nhead=config['model'].get('nhead', 4),
            trans_layers=config['model'].get('trans_layers', 2)
        )

    checkpoint_path = f'experiments/checkpoints/{args.model}/best_model.pth'
    if not os.path.exists(checkpoint_path):
        print(f"[Error] Weights not found at {checkpoint_path}. Run training first.")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    print(f"[System] Testing on hardware: {device}")
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.numpy())

    print("\n" + "="*60)
    print(f" RESEARCH GRADE REPORT: {args.model.upper()} ARCHITECTURE ")
    print("="*60)
    
    report = classification_report(all_targets, all_preds, target_names=CLASS_NAMES, digits=4)
    print(report)
    
    report_save_path = f'experiments/logs/report_{args.model}.txt'
    with open(report_save_path, 'w') as f:
        f.write(report)

    plot_confusion_matrix(all_targets, all_preds, CLASS_NAMES, model_name=args.model)
    print(f"[System] Evaluation complete. Results saved in experiments/logs/")

if __name__ == "__main__":
    evaluate_model()
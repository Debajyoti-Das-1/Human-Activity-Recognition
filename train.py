import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim

from src.utils.seed import set_seed
from src.data.dataset import get_dataloaders
from src.models.cnn import HAR_1DCNN
from src.models.lstm import HAR_LSTM
from src.models.cnn_lstm import HAR_CNNLSTM
from src.models.transformer import HAR_Transformer
from src.models.advanced_hybrid import HAR_AdvancedHybrid # <-- NEW IMPORT
from src.training.trainer import Trainer
from src.evaluation.history_plotter import plot_training_history

def parse_args():
    parser = argparse.ArgumentParser(description="HAR Deep Learning Training Engine")
    parser.add_argument(
        '--model', 
        type=str, 
        default='cnn', 
        choices=['cnn', 'lstm', 'cnn_lstm', 'transformer', 'advanced_hybrid'], # <-- UPDATED
        help="Select the architecture to train"
    )
    parser.add_argument('--lr', type=float, help="Override learning rate")
    parser.add_argument('--batch_size', type=int, help="Override batch size")
    parser.add_argument('--epochs', type=int, help="Override epoch count")
    return parser.parse_args()

def load_config(model_name):
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
    model_config_path = f"configs/{model_name}.yaml"
    if os.path.exists(model_config_path):
        with open(model_config_path, "r") as f:
            model_config = yaml.safe_load(f)
            for key, value in model_config.items():
                if isinstance(value, dict) and key in config:
                    config[key].update(value)
                else:
                    config[key] = value
    return config

def main():
    args = parse_args()
    config = load_config(args.model)
    
    # Apply CLI Overrides
    if args.lr: config['training']['learning_rate'] = args.lr
    if args.batch_size: config['training']['batch_size'] = args.batch_size
    if args.epochs: config['training']['epochs'] = args.epochs
    
    set_seed(config['experiment']['seed'])
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[System] Engine running on: {device}")

    BATCH_SIZE = config['training']['batch_size']
    EPOCHS = config['training']['epochs']
    LEARNING_RATE = config['training']['learning_rate']
    PATIENCE = config['training'].get('patience', 10) 
    NUM_CLASSES = config['data']['num_classes']
    NUM_FEATURES = config['data']['num_features']
    CHECKPOINT_DIR = f"{config['experiment']['checkpoint_dir']}/{args.model}"

    print("[System] Loading DataLoaders...")
    train_loader, test_loader = get_dataloaders(
        config['data']['train_x'], config['data']['train_y'], 
        config['data']['test_x'], config['data']['test_y'], 
        batch_size=BATCH_SIZE
    )

    print(f"[System] Initializing {args.model.upper()}...")
    if args.model == 'cnn':
        model = HAR_1DCNN(num_classes=NUM_CLASSES, in_channels=NUM_FEATURES)
    elif args.model == 'lstm':
        model = HAR_LSTM(input_dim=NUM_FEATURES, hidden_dim=config['model'].get('hidden_dim', 64), 
                         num_layers=config['model'].get('num_layers', 2), num_classes=NUM_CLASSES)
    elif args.model == 'cnn_lstm':
        model = HAR_CNNLSTM(num_classes=NUM_CLASSES, in_channels=NUM_FEATURES)
    elif args.model == 'transformer':
        model = HAR_Transformer(
            num_classes=NUM_CLASSES, in_channels=NUM_FEATURES,
            d_model=config['model'].get('d_model', 128),
            nhead=config['model'].get('nhead', 8),
            num_layers=config['model'].get('num_layers', 4)
        )
    elif args.model == 'advanced_hybrid': # <-- NEW BLOCK
        model = HAR_AdvancedHybrid(
            num_classes=NUM_CLASSES, in_channels=NUM_FEATURES,
            cnn_dim=config['model'].get('cnn_dim', 64),
            lstm_hidden=config['model'].get('lstm_hidden', 64),
            nhead=config['model'].get('nhead', 4),
            trans_layers=config['model'].get('trans_layers', 2)
        )
        
    print(model.format_details())
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=config['training'].get('weight_decay', 0.0001))

    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=test_loader, 
        criterion=criterion, optimizer=optimizer, device=device,
        checkpoint_dir=CHECKPOINT_DIR, patience=PATIENCE
    )

    history = trainer.fit(epochs=EPOCHS)
    plot_training_history(history, args.model)
    print(f"[System] {args.model.upper()} Deep Convergence Complete!")

if __name__ == "__main__":
    main()
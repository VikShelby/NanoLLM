import argparse
import yaml
import torch
import os
import logging
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nano_llm.models import SimpleTransformerDecoder
from nano_llm.data_processing import create_dataloaders
from nano_llm.training import train_pipeline # Import the training pipeline
from nano_llm.utils import setup_logging

def main():
    parser = argparse.ArgumentParser(description="Train the Simple LLM")
    parser.add_argument('--model_config', type=str, default='config/model_config.yaml', help='Path to model configuration file')
    parser.add_argument('--training_config', type=str, default='config/training_config.yaml', help='Path to training configuration file')
    parser.add_argument('--data_config', type=str, default='config/data_config.yaml', help='Path to data configuration file')
    # parser.add_argument('--resume_checkpoint', type=str, help='Optional path to resume training from checkpoint') # Add this later
    args = parser.parse_args()

    # Load configurations
    with open(args.model_config, 'r') as f:
        model_config = yaml.safe_load(f)
    with open(args.training_config, 'r') as f:
        training_config = yaml.safe_load(f)
    with open(args.data_config, 'r') as f:
        data_config = yaml.safe_load(f)

    # Determine device and setup logging using directory from training config
    device = torch.device(training_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    # Ensure the device is valid — fallback if CUDA is not available
    if training_config.get('device') == 'cuda' and not torch.cuda.is_available():
     logging.warning("CUDA requested but not available. Falling back to CPU.")
     device = torch.device('cpu')
     training_config['device'] = 'cpu'
    else:
     device = torch.device(training_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
     training_config['device'] = str(device)

    training_config['device'] = str(device) # Store effective device back in config for logging/checkpoint
    os.makedirs(training_config['log_dir'], exist_ok=True)
    setup_logging(training_config['log_dir'])
    logging.info(f"Using device: {device}")

    # Load processed data and create dataloaders
    train_loader, val_loader, tokenizer = create_dataloaders(
        processed_dir=data_config['processed_data_dir'],
        seq_length=data_config['seq_length'],
        batch_size=training_config['batch_size']
    )

    # Update vocab size in model config from loaded tokenizer BEFORE building model
    model_config['vocab_size'] = tokenizer.vocab_size
    model_config['seq_length'] = data_config['seq_length']  # Add this line
    logging.info(f"Vocab size from data: {model_config['vocab_size']}")
    logging.info(f"Sequence length: {model_config['seq_length']}")


    # Build the model
    model = SimpleTransformerDecoder(**model_config).to(device)
    logging.info("Model built.")
    logging.info(f"Model config: {model_config}")


    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config['learning_rate'])

    # Define loss function (CrossEntropyLoss for predicting next token)
    criterion = torch.nn.CrossEntropyLoss()

    # Start training pipeline
    train_pipeline(model, train_loader, val_loader, optimizer, criterion, training_config, device)

    logging.info("Training finished.")


if __name__ == '__main__':
    main()
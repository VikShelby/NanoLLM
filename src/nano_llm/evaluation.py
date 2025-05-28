import torch
from torch.utils.data import DataLoader
from nano_llm.models import SimpleTransformerDecoder
from nano_llm.data_processing import LLMDataset, create_dataloaders, CharTokenizer
import logging
import yaml
import os
from tqdm import tqdm

def evaluate(model, dataloader, criterion, device):
    """Evaluates the model on a dataset."""
    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    num_batches = 0

    logging.info("Starting evaluation...")
    with torch.no_grad(): # Disable gradient calculations
        for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader, desc="Evaluating")):
            inputs, targets = inputs.to(device), targets.to(device)

            logits = model(inputs) # (batch_size, seq_length, vocab_size)

            # Reshape for criterion: (N, C) and (N)
            # N = batch_size * seq_length, C = vocab_size
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    logging.info(f"Evaluation Loss: {avg_loss:.4f}")

    return avg_loss

if __name__ == '__main__':
    import argparse
    from nano_llm.utils import setup_logging, load_checkpoint

    parser = argparse.ArgumentParser(description="Evaluate a Simple LLM")
    parser.add_argument('--model_config', type=str, default='config/model_config.yaml', help='Path to model configuration file')
    parser.add_argument('--data_config', type=str, default='config/data_config.yaml', help='Path to data configuration file')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint file')
    args = parser.parse_args()

    setup_logging() # Setup logging

    # Load configurations
    with open(args.model_config, 'r') as f:
        model_config = yaml.safe_load(f)
    with open(args.data_config, 'r') as f:
        data_config = yaml.safe_load(f)

    # Determine device
    device = torch.device(data_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    logging.info(f"Using device: {device}")

    # Load processed data and create dataloaders (only val_loader needed for evaluation)
    # Need tokenizer to get vocab size
    _, val_loader, tokenizer = create_dataloaders(
        processed_dir=data_config['processed_data_dir'],
        seq_length=data_config['seq_length'],
        batch_size=data_config.get('eval_batch_size', 64) # Use training batch size if eval_batch_size not specified
    )

    # Update vocab size in model config from loaded tokenizer
    model_config['vocab_size'] = tokenizer.vocab_size
    logging.info(f"Vocab size from data: {model_config['vocab_size']}")

    # Build the model
    model = SimpleTransformerDecoder(**model_config).to(device)
    logging.info("Model built.")

    # Load model weights from checkpoint
    load_checkpoint(args.checkpoint_path, model, device=device)

    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Run evaluation
    evaluate(model, val_loader, criterion, device)
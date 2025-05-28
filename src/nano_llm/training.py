import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# Ensure these imports match your project structure
from nano_llm.models import SimpleTransformerDecoder
from nano_llm.data_processing import LLMDataset, create_dataloaders, CharTokenizer
from nano_llm.evaluation import evaluate # Assume evaluate function is correctly imported
from nano_llm.utils import save_checkpoint, load_checkpoint, setup_logging
import yaml
import os
import time
import logging
from tqdm import tqdm
import sys # Needed for sys.exit

def train_epoch(model, dataloader, optimizer, criterion, device, epoch, log_freq=10):
    """Trains the model for one epoch."""
    model.train() # Set model to training mode
    total_loss = 0.0
    start_time = time.time()

    logging.info(f"Starting epoch {epoch}...")
    for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch} Training")):
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        logits = model(inputs) # (batch_size, seq_length, vocab_size)

        # Reshape for criterion: (N, C) and (N)
        # N = batch_size * seq_length, C = vocab_size
        # targets need to be long() as CrossEntropyLoss expects Long type for targets
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1).long())

        # Backward pass and optimization
        optimizer.zero_grad() # Clear previous gradients
        loss.backward()       # Compute gradients
        optimizer.step()      # Update weights

        total_loss += loss.item()

        # --- FIX: Removed duplicate logging line ---
        if (batch_idx + 1) % log_freq == 0:
            elapsed = time.time() - start_time
            avg_time_per_batch = elapsed / (batch_idx + 1)
            logging.info(f"Epoch {epoch}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}, Time per batch: {avg_time_per_batch:.4f}s")

    avg_loss = total_loss / len(dataloader)
    logging.info(f"Epoch {epoch} finished. Avg Training Loss: {avg_loss:.4f}")
    return avg_loss


def train_pipeline(model, train_loader, val_loader, optimizer, criterion, training_config, device):
    """Orchestrates the entire training process with early stopping."""
    epochs = training_config['epochs']
    checkpoint_dir = training_config['checkpoint_dir']
    checkpoint_freq = training_config.get('checkpoint_freq', epochs + 1) # Default to never if not specified
    log_freq = training_config.get('log_freq', 10) # Default to 10
    early_stopping_patience = training_config.get('early_stopping_patience', None) # Get patience
    log_dir = training_config.get('log_dir', 'logs') # Default log dir

    # Ensure logging is set up with dir (already done in main, but good safeguard)
    setup_logging(log_dir)

    best_val_loss = float('inf')
    epochs_since_last_improvement = 0 # Counter for early stopping

    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)


    for epoch in range(1, epochs + 1):
        # Train for one epoch
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            log_freq=log_freq
        )

        # Evaluate on validation set
        # Ensure model is set to eval mode inside evaluate function
        val_loss = evaluate(model, val_loader, criterion, device)

        # Check for validation loss improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_since_last_improvement = 0 # Reset patience counter

            # --- FIX: Save the BEST model checkpoint ---
            # You can use a fixed name for the best model
            best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, best_val_loss, checkpoint_dir, filename='best_model.pth')
            logging.info(f"Validation loss improved. Saving best model to {best_checkpoint_path}. Best loss: {best_val_loss:.4f}")
        else:
            epochs_since_last_improvement += 1
            logging.info(f"Validation loss did not improve. Best loss: {best_val_loss:.4f} (Epoch {epoch-epochs_since_last_improvement}). Patience: {epochs_since_last_improvement}/{early_stopping_patience if early_stopping_patience is not None else 'N/A'}")


        # Save periodic checkpoint (optional, for resuming)
        if epoch % checkpoint_freq == 0:
            # Use a specific name like epoch_{}.pth
            periodic_checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch:04d}.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_dir, filename=f'epoch_{epoch:04d}.pth')
            logging.info(f"Saved periodic checkpoint to {periodic_checkpoint_path}")


        # Check for Early Stopping
        if early_stopping_patience is not None and epochs_since_last_improvement >= early_stopping_patience:
            logging.info(f"Early stopping triggered after {epoch} epochs. Validation loss did not improve for {early_stopping_patience} epochs.")
            break # Exit the training loop

    logging.info("Training finished.")


# This __main__ block allows running training directly from this file,
# but the preferred way is via the script in src/scripts/train.py
if __name__ == '__main__':
    import argparse
    # from nano_llm.utils import setup_logging # Already imported above

    parser = argparse.ArgumentParser(description="Train a Simple LLM")
    parser.add_argument('--model_config', type=str, default='config/model_config.yaml', help='Path to model configuration file')
    parser.add_argument('--training_config', type=str, default='config/training_config.yaml', help='Path to training configuration file')
    parser.add_argument('--data_config', type=str, default='config/data_config.yaml', help='Path to data configuration file')
    # parser.add_argument('--resume_checkpoint', type=str, help='Path to a checkpoint to resume training from') # Add resume capability later
    args = parser.parse_args()

    # Load configurations
    try:
        with open(args.model_config, 'r') as f:
            model_config = yaml.safe_load(f)
        with open(args.training_config, 'r') as f:
            training_config = yaml.safe_load(f)
        with open(args.data_config, 'r') as f:
            data_config = yaml.safe_load(f)
    except FileNotFoundError as e:
        logging.error(f"Configuration file not found: {e}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file: {e}")
        sys.exit(1)


    # Determine device
    # Check for 'device' key in training_config, default to cuda if available
    device_str = training_config.get('device', 'auto').lower()

    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device_str == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            logging.warning("CUDA requested but not available. Falling back to CPU.")
            device = torch.device('cpu')
    elif device_str == 'cpu':
        device = torch.device('cpu')
    else:
        logging.warning(f"Unknown device '{device_str}' specified in training_config. Falling back to CPU.")
        device = torch.device('cpu')

    logging.info(f"Using device: {device}")
    training_config['device'] = str(device) # Store effective device back in config


    # Create directories for output if they don't exist
    # Use .get for checkpoint_dir and log_dir to provide defaults if missing in config
    checkpoint_dir = training_config.get('checkpoint_dir', 'checkpoints')
    log_dir = training_config.get('log_dir', 'logs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    training_config['checkpoint_dir'] = checkpoint_dir # Ensure path is stored back
    training_config['log_dir'] = log_dir # Ensure path is stored back


    # Setup logging now that log_dir is guaranteed to exist
    setup_logging(training_config['log_dir'])
    # Re-log device info after full logging setup
    logging.info(f"Using device: {device} (Effective)")


    # Load processed data and create dataloaders
    processed_dir = data_config.get('processed_data_dir', 'data/processed')
    seq_length = data_config.get('seq_length', 128)
    batch_size = training_config.get('batch_size', 64)

    try:
        train_loader, val_loader, tokenizer = create_dataloaders(
            processed_dir=processed_dir,
            seq_length=seq_length,
            batch_size=batch_size
        )
    except FileNotFoundError as e:
        logging.error(f"Data files not found: {e}. Ensure preprocess_data.py has been run.")
        sys.exit(1)
    except Exception as e:
         logging.error(f"Error creating dataloaders: {e}")
         sys.exit(1)

    logging.info(f"Train DataLoader batches: {len(train_loader)}, Validation DataLoader batches: {len(val_loader)}")


    # Update vocab size and seq_length in model config BEFORE building model
    # Use .get for safety, though these are likely required
    model_config['vocab_size'] = tokenizer.vocab_size
    model_config['seq_length'] = seq_length # --- FIX: Add seq_length to model_config ---

    logging.info(f"Model config vocab size: {model_config['vocab_size']}, seq_length: {model_config['seq_length']}")


    # Build the model
    try:
        model = SimpleTransformerDecoder(**model_config).to(device)
    except Exception as e:
        logging.error(f"Error building the model with config {model_config}: {e}")
        sys.exit(1)

    logging.info("Model built.")
    logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} Million")


    # Define optimizer
    learning_rate = training_config.get('learning_rate', 1e-3) # Default LR
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    logging.info(f"Optimizer created with learning rate: {learning_rate}")

    # Define loss function (CrossEntropyLoss for predicting next token)
    criterion = torch.nn.CrossEntropyLoss()
    logging.info("Criterion (Loss Function): CrossEntropyLoss")


    # --- Add resume capability later ---
    # if args.resume_checkpoint:
    #     try:
    #         start_epoch, best_val_loss = load_checkpoint(args.resume_checkpoint, model, optimizer, device)
    #         logging.info(f"Resuming training from checkpoint {args.resume_checkpoint} at epoch {start_epoch}")
    #         # Adjust the training loop range and best_val_loss/patience counter accordingly
    #     except FileNotFoundError:
    #         logging.error(f"Resume checkpoint not found at {args.resume_checkpoint}. Starting from scratch.")
    #     except Exception as e:
    #         logging.error(f"Error loading resume checkpoint {args.resume_checkpoint}: {e}. Starting from scratch.")
    #         start_epoch = 0 # Ensure start_epoch is defined even if resume fails
    # else:
    #     start_epoch = 0 # Start from epoch 0 if not resuming

    # Start training pipeline
    # Potentially pass start_epoch and initial best_val_loss if resuming
    train_pipeline(model, train_loader, val_loader, optimizer, criterion, training_config, device)

    logging.info("Script finished.")
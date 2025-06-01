import logging
import sys
import torch
import os

def setup_logging(log_dir=None):
    """Sets up basic logging."""

    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            stream=sys.stdout
        )
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'))
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logging.getLogger().addHandler(file_handler)
        logging.info("Logging configured.")
    else:

         pass




def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, filename):
    """Saves model and optimizer state to a specific filename."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    try:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, filepath)
        logging.info(f"Checkpoint saved to {filepath}")
    except Exception as e:
        logging.error(f"Error saving checkpoint to {filepath}: {e}")




def load_checkpoint(filepath, model, optimizer=None, device='cpu'):
    """Loads model and optionally optimizer state from a checkpoint."""
    if not os.path.exists(filepath):

        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    try:
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info(f"Checkpoint loaded successfully from {filepath} (Epoch {checkpoint.get('epoch', '?')}, Loss {checkpoint.get('loss', '?'):.4f})")
        return checkpoint.get('epoch', 0), checkpoint.get('loss', float('inf'))
    except Exception as e:
        logging.error(f"Error loading checkpoint from {filepath}: {e}")



        raise RuntimeError(f"Failed to load checkpoint data from {filepath}") from e






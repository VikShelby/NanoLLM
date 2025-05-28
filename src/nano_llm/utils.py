import logging
import sys
import torch
import os

def setup_logging(log_dir=None):
    """Sets up basic logging."""
    # Prevent adding multiple handlers if called multiple times
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            stream=sys.stdout # Log to console by default
        )
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'))
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logging.getLogger().addHandler(file_handler)
        logging.info("Logging configured.")
    else:
         # Logging already configured
         pass


# --- Corrected save_checkpoint ---
# Now accepts a 'filename' parameter and only saves ONE file per call
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


# --- Corrected load_checkpoint ---
# Handles file not found by raising FileNotFoundError (which the caller should catch)
def load_checkpoint(filepath, model, optimizer=None, device='cpu'):
    """Loads model and optionally optimizer state from a checkpoint."""
    if not os.path.exists(filepath):
        # Re-raise the specific error so the caller can handle it appropriately
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
        # Could re-raise a different error or just let it return None/fail later
        # For now, log error and return None might be okay depending on caller
        # Or better, re-raise after logging
        raise RuntimeError(f"Failed to load checkpoint data from {filepath}") from e


# Keep other utility functions if you have them (e.g., tokenization utilities if not in data_processing)
# Example placeholder if you had other functions:
# def another_utility_function():
#     pass
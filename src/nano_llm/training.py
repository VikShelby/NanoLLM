
import torch
import torch.nn as nn
from torch.utils.data import DataLoader






from .models import SimpleTransformerDecoder




from .evaluation import evaluate
import yaml
import os
import time
import logging
from tqdm import tqdm
import sys


def setup_logging(log_dir):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, 'training.log'))
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(fh)

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, filename='checkpoint.pth', scaler=None):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    if scaler:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    torch.save(checkpoint, path)
    logging.info(f"Saved checkpoint to {path}")





class DummyTokenizer:
    def __init__(self):
        self.vocab_size = 50
        self.pad_token_id = 0
    def encode(self, text): return [ord(c) for c in text[:10]]
    def decode(self, ids): return "".join([chr(i) for i in ids])

def create_dataloaders(processed_dir, seq_length, batch_size):
    logging.warning("Placeholder `create_dataloaders` is being used. "
                    "You MUST implement subword tokenization and data loading.")



    num_dummy_samples = 1000
    dummy_train_data = [torch.randint(0, 50, (seq_length,)) for _ in range(int(num_dummy_samples*0.9))]
    dummy_val_data = [torch.randint(0, 50, (seq_length,)) for _ in range(int(num_dummy_samples*0.1))]


    dummy_train_inputs = torch.stack([d[:-1] for d in dummy_train_data])
    dummy_train_targets = torch.stack([d[1:] for d in dummy_train_data])
    dummy_val_inputs = torch.stack([d[:-1] for d in dummy_val_data])
    dummy_val_targets = torch.stack([d[1:] for d in dummy_val_data])

    from torch.utils.data import TensorDataset
    train_dataset = TensorDataset(dummy_train_inputs, dummy_train_targets)
    val_dataset = TensorDataset(dummy_val_inputs, dummy_val_targets)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    tokenizer = DummyTokenizer()
    tokenizer.vocab_size = 50257

    logging.info(f"Using DUMMY tokenizer with vocab_size: {tokenizer.vocab_size}")
    return train_loader, val_loader, tokenizer



def train_epoch(model, dataloader, optimizer, criterion, device, epoch,
                grad_accumulation_steps, use_amp, scaler, grad_clip, log_freq):
    model.train()
    total_loss = 0.0
    accumulated_loss = 0.0
    optimizer.zero_grad()


    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch} Training")

    for batch_idx, (inputs, targets) in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)


        device_type = device.type

        with torch.amp.autocast(device_type=device_type, dtype=torch.float16 if device_type == 'cuda' else torch.bfloat16, enabled=use_amp):
             logits = model(inputs)
             loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1).long())
             loss = loss / grad_accumulation_steps

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accumulated_loss += loss.item() * grad_accumulation_steps


        if (batch_idx + 1) % grad_accumulation_steps == 0:
            if use_amp:
                if grad_clip > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            optimizer.zero_grad()


            if ( (batch_idx + 1) // grad_accumulation_steps ) % log_freq == 0 or \
               (batch_idx + 1) == len(dataloader):
                avg_accumulated_loss = accumulated_loss / (grad_accumulation_steps * min(log_freq, (batch_idx + 1) // grad_accumulation_steps if (batch_idx + 1) // grad_accumulation_steps > 0 else 1 ))
                progress_bar.set_postfix({'loss': f"{avg_accumulated_loss:.4f}"})
                logging.info(f"Epoch {epoch}, Step {(batch_idx + 1)//grad_accumulation_steps}/{len(dataloader)//grad_accumulation_steps}, "
                             f"MicroBatch {batch_idx+1}/{len(dataloader)}, Avg Acc Loss: {avg_accumulated_loss:.4f}")
                accumulated_loss = 0.0

        total_loss += loss.item() * grad_accumulation_steps

    avg_epoch_loss = total_loss / (len(dataloader))
    logging.info(f"Epoch {epoch} finished. Avg Training Micro-Batch Loss: {avg_epoch_loss:.4f}")
    return avg_epoch_loss


def train_pipeline(model, train_loader, val_loader, optimizer, criterion, training_config, device, model_config):
    epochs = training_config['epochs']
    checkpoint_dir = training_config['checkpoint_dir']

    checkpoint_freq = training_config.get('checkpoint_freq', 1)
    log_freq_steps = training_config.get('log_freq', 10)
    early_stopping_patience = training_config.get('early_stopping_patience', None)
    log_dir = training_config.get('log_dir', 'output/logs')

    grad_accumulation_steps = training_config.get('gradient_accumulation_steps', 1)
    use_amp = training_config.get('use_amp', False) and device.type == 'cuda'
    grad_clip = training_config.get('grad_clip', 0.0)


    scaler = None
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
        logging.info("Using Automatic Mixed Precision (AMP).")

    setup_logging(log_dir)

    best_val_loss = float('inf')
    epochs_since_last_improvement = 0

    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        logging.info(f"--- Starting Epoch {epoch}/{epochs} ---")
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
            grad_accumulation_steps, use_amp, scaler, grad_clip, log_freq_steps
        )

        val_loss = evaluate(model, val_loader, criterion, device)

        logging.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_since_last_improvement = 0
            save_checkpoint(model, optimizer, epoch, best_val_loss, checkpoint_dir,
                            filename='best_model.pth', scaler=scaler if use_amp else None)
            logging.info(f"Validation loss improved. Saving best model. Best Val Loss: {best_val_loss:.4f}")
        else:
            epochs_since_last_improvement += 1
            logging.info(f"Validation loss did not improve. Patience: {epochs_since_last_improvement}/{early_stopping_patience if early_stopping_patience else 'N/A'}")

        if epoch % checkpoint_freq == 0:
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_dir,
                            filename=f'epoch_{epoch:03d}.pth', scaler=scaler if use_amp else None)

        if early_stopping_patience is not None and epochs_since_last_improvement >= early_stopping_patience:
            logging.info(f"Early stopping triggered after {epoch} epochs.")
            break

    logging.info("Training finished.")
    logging.info(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train NanoLLM")
    parser.add_argument('--model_config', type=str, default='config/model_config.yaml', help='Path to model configuration file')
    parser.add_argument('--training_config', type=str, default='config/training_config.yaml', help='Path to training configuration file')
    parser.add_argument('--data_config', type=str, default='config/data_config.yaml', help='Path to data configuration file')
    args = parser.parse_args()

    try:
        with open(args.model_config, 'r') as f: model_config_values = yaml.safe_load(f)
        with open(args.training_config, 'r') as f: training_config = yaml.safe_load(f)
        with open(args.data_config, 'r') as f: data_config = yaml.safe_load(f)
    except FileNotFoundError as e:
        print(f"Configuration file not found: {e}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        sys.exit(1)


    log_dir_main = training_config.get('log_dir', 'output/logs')
    setup_logging(log_dir_main)


    device_str = training_config.get('device', 'auto').lower()
    if device_str == 'auto': device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device_str == 'cuda' and torch.cuda.is_available(): device = torch.device('cuda')
    else: device = torch.device('cpu')
    logging.info(f"Using device: {device}")
    if device.type == 'cuda':
        logging.info(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        logging.info(f"CUDA Version: {torch.version.cuda}")
    training_config['device'] = str(device)





    processed_dir = data_config.get('processed_data_dir', 'data/processed')
    seq_length_data = data_config.get('seq_length', 128)
    batch_size_micro = training_config.get('batch_size', 64)


    if model_config_values.get('seq_length') is None:
        model_config_values['seq_length'] = seq_length_data
        logging.info(f"Setting model_config seq_length from data_config: {seq_length_data}")
    elif model_config_values['seq_length'] != seq_length_data:
        logging.warning(f"Mismatch: model_config seq_length ({model_config_values['seq_length']}) "
                        f"!= data_config seq_length ({seq_length_data}). Using model_config value.")


    model_seq_length = model_config_values['seq_length']

    try:

        train_loader, val_loader, tokenizer = create_dataloaders(
            processed_dir=processed_dir,
            seq_length=model_seq_length,
            batch_size=batch_size_micro
        )
    except Exception as e:
        logging.error(f"Error creating dataloaders: {e}. Ensure preprocess_data.py has been run with subword tokenization.")
        sys.exit(1)



    if tokenizer is None or not hasattr(tokenizer, 'vocab_size'):
        logging.error("Tokenizer could not be loaded or has no vocab_size. "
                      "This is critical for model initialization.")
        sys.exit(1)
    model_config_values['vocab_size'] = tokenizer.vocab_size
    logging.info(f"Using vocab_size from tokenizer: {tokenizer.vocab_size}")



    try:

        model = SimpleTransformerDecoder(**model_config_values).to(device)
    except Exception as e:
        logging.error(f"Error building the model with config {model_config_values}: {e}")

        import traceback
        traceback.print_exc()
        sys.exit(1)

    logging.info("Model built successfully.")



    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.get('learning_rate', 1e-3))
    criterion = nn.CrossEntropyLoss()




    logging.info("Starting training pipeline...")
    train_pipeline(model, train_loader, val_loader, optimizer, criterion, training_config, device, model_config_values)

    logging.info("Script finished.")

import argparse
import yaml
import torch
import os
import logging
import sys





sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nano_llm.models import SimpleTransformerDecoder
from nano_llm.data_processing import create_dataloaders
from nano_llm.training import train_pipeline
from nano_llm.utils import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Train the NanoLLM")
    parser.add_argument('--model_config', type=str, default='config/model_config.yaml', help='Path to model configuration YAML')
    parser.add_argument('--training_config', type=str, default='config/training_config.yaml', help='Path to training configuration YAML')
    parser.add_argument('--data_config', type=str, default='config/data_config.yaml', help='Path to data configuration YAML')
    args = parser.parse_args()


    try:
        with open(args.model_config, 'r') as f: model_cfg_values = yaml.safe_load(f)
        with open(args.training_config, 'r') as f: training_cfg = yaml.safe_load(f)
        with open(args.data_config, 'r') as f: data_cfg = yaml.safe_load(f)
    except FileNotFoundError as e:
        print(f"FATAL: Configuration file not found: {e}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"FATAL: Error parsing YAML configuration: {e}")
        sys.exit(1)


    log_dir = training_cfg.get('log_dir', 'output/logs')
    os.makedirs(log_dir, exist_ok=True)
    setup_logging(log_dir)

    logging.info("--- NanoLLM Training Script ---")
    logging.info(f"Model Config: {args.model_config}")
    logging.info(f"Training Config: {args.training_config}")
    logging.info(f"Data Config: {args.data_config}")


    device_str = training_cfg.get('device', 'auto').lower()
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device_str == 'cuda':
        if not torch.cuda.is_available():
            logging.warning("CUDA specified but not available. Falling back to CPU.")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    training_cfg['device'] = str(device)
    logging.info(f"Effective device: {device}")
    if device.type == 'cuda':
        logging.info(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")





    micro_batch_size = training_cfg['batch_size']
    try:
        logging.info(f"Creating dataloaders with micro_batch_size: {micro_batch_size} and seq_length from data_config: {data_cfg['seq_length']}")
        train_loader, val_loader, tokenizer = create_dataloaders(
            data_config=data_cfg,
            training_batch_size=micro_batch_size
        )
    except FileNotFoundError as e:
        logging.error(f"Data/tokenizer files not found: {e}. Ensure 'src/scripts/preprocess_data.py --config {args.data_config}' has been run successfully.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Failed to create dataloaders: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


    actual_vocab_size = tokenizer.get_vocab_size()
    model_cfg_values['vocab_size'] = actual_vocab_size
    


    if 'seq_length' not in model_cfg_values:
        model_cfg_values['seq_length'] = data_cfg['seq_length']
        logging.info(f"Model 'seq_length' not in model_config, taking from data_config: {data_cfg['seq_length']}")
    elif model_cfg_values['seq_length'] != data_cfg['seq_length']:
        logging.warning(f"Mismatch: model_config.seq_length ({model_cfg_values['seq_length']}) != data_config.seq_length ({data_cfg['seq_length']}). "
                        f"Ensure data was preprocessed with model's expected seq_length. Using model_config value: {model_cfg_values['seq_length']}")




    logging.info(f"Final model configuration values to be used: {model_cfg_values}")


    try:
        model = SimpleTransformerDecoder(**model_cfg_values).to(device)
    except Exception as e:
        logging.error(f"Error building the model with resolved config {model_cfg_values}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)



    optimizer = torch.optim.AdamW(model.parameters(), lr=training_cfg['learning_rate'])
    logging.info(f"Optimizer: AdamW with LR {training_cfg['learning_rate']}")




    pad_token_id = tokenizer.token_to_id("<pad>")
    if pad_token_id is None:
        logging.warning("'<pad>' token not found in tokenizer. Loss will not ignore padding.")
        criterion = torch.nn.CrossEntropyLoss()
    else:
        logging.info(f"Using CrossEntropyLoss, ignoring index: {pad_token_id}")
        criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)



    logging.info("Starting training pipeline...")
    train_pipeline(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        training_config=training_cfg,
        device=device,
        model_config=model_cfg_values
    )

    logging.info("Training script finished.")

if __name__ == '__main__':
    main()

import argparse
import yaml
import torch
import os
import logging
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nano_llm.models import SimpleTransformerDecoder
from tokenizers import Tokenizer



def setup_logging(log_dir=None):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])
    if log_dir:
        pass

def load_checkpoint(checkpoint_path, model, optimizer=None, scaler=None, device='cpu'):
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scaler and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))
    logging.info(f"Checkpoint loaded. Epoch: {epoch}, Loss: {loss:.4f}")
    return epoch, loss


def main():
    parser = argparse.ArgumentParser(description="Run inference on a trained NanoLLM")
    parser.add_argument('--model_config', type=str, default='config/model_config.yaml', help='Path to model configuration file')
    parser.add_argument('--data_config', type=str, default='config/data_config.yaml', help='Path to data configuration file')
    parser.add_argument('--inference_config', type=str, default='config/inference_config.yaml', help='Path to inference configuration file')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint file (e.g., best_model.pth)')
    parser.add_argument('--prompt', type=str, required=True, help='Initial text prompt for generation')
    args = parser.parse_args()

    setup_logging()


    try:
        with open(args.model_config, 'r') as f: model_config_values = yaml.safe_load(f)
        with open(args.data_config, 'r') as f: data_config = yaml.safe_load(f)
        with open(args.inference_config, 'r') as f: inference_config = yaml.safe_load(f)
    except FileNotFoundError as e:
        logging.error(f"Configuration file not found: {e}"); sys.exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file: {e}"); sys.exit(1)


    device_str = inference_config.get('device', 'auto').lower()
    if device_str == 'auto': device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device_str == 'cuda' and torch.cuda.is_available(): device = torch.device('cuda')
    else: device = torch.device('cpu')
    logging.info(f"Using device: {device}")


    processed_dir = data_config.get('processed_data_dir', 'data/processed')
    tokenizer_path = os.path.join(processed_dir, 'tokenizer.json')
    if not os.path.exists(tokenizer_path):
         logging.error(f"BPE Tokenizer file not found at {tokenizer_path}. Run preprocess_data.py first.")
         sys.exit(1)
    try:
        tokenizer = Tokenizer.from_file(tokenizer_path)
    except Exception as e:
         logging.error(f"Error loading BPE tokenizer from {tokenizer_path}: {e}"); sys.exit(1)

    actual_vocab_size = tokenizer.get_vocab_size()
    logging.info(f"BPE Tokenizer loaded. Actual vocab size: {actual_vocab_size}")


    model_config_values['vocab_size'] = actual_vocab_size


    if 'seq_length' not in model_config_values:
        model_config_values['seq_length'] = data_config.get('seq_length', 512)
        logging.info(f"Setting model_config seq_length from data_config/default: {model_config_values['seq_length']}")

    logging.info(f"Model config to be used: vocab_size={model_config_values['vocab_size']}, seq_length={model_config_values['seq_length']}")


    try:
        model = SimpleTransformerDecoder(**model_config_values).to(device)
    except Exception as e:
        logging.error(f"Error building the model with config {model_config_values}: {e}"); sys.exit(1)
    logging.info("Model built.")



    try:
        load_checkpoint(args.checkpoint_path, model, device=device)
        logging.info(f"Model weights loaded from {args.checkpoint_path}")
    except FileNotFoundError:
        logging.error(f"Checkpoint file not found at {args.checkpoint_path}."); sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading checkpoint {args.checkpoint_path}: {e}"); sys.exit(1)

    model.eval()
    logging.info("Model set to evaluation mode.")


    prompt_encoding = tokenizer.encode(args.prompt)
    prompt_token_ids = prompt_encoding.ids

    if not prompt_token_ids:
        logging.warning("Prompt resulted in empty token list. Cannot generate."); return

    prompt_tensor = torch.tensor([prompt_token_ids], dtype=torch.long, device=device)

    logging.info(f"Generating from prompt: '{args.prompt}' (Token IDs: {prompt_token_ids})")


    max_new_tokens = inference_config.get('max_new_tokens', 100)
    temperature = inference_config.get('temperature', 1.0)
    top_k = inference_config.get('top_k', None)

    logging.info(f"Generation parameters: max_new_tokens={max_new_tokens}, temperature={temperature}, top_k={top_k}")


    with torch.no_grad():
        generated_tokens_tensor = model.generate(
            prompt_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )

    generated_ids_list = generated_tokens_tensor[0].tolist()
    generated_text = tokenizer.decode(generated_ids_list)

    print("\n" + "=" * 40)
    print("Input Prompt:")
    print(args.prompt)
    print("-" * 40)
    print("Generated Text:")

    print(generated_text)
    print("=" * 40 + "\n")

if __name__ == '__main__':
    main()
    logging.info("Inference script finished.")
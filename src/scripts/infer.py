import argparse
import yaml
import torch
import os
import logging
import sys
# Ensure the path to the nano_llm package is in sys.path
# This assumes inference.py is in src/scripts relative to the project root
# If your script location differs, adjust this path accordingly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from nano_llm.models import SimpleTransformerDecoder
# Assuming CharTokenizer is in data_processing or utils
# Adjust import based on its actual location if necessary
from nano_llm.data_processing import CharTokenizer
from nano_llm.utils import load_checkpoint, setup_logging

def main():
    parser = argparse.ArgumentParser(description="Run inference on a trained Simple LLM")
    parser.add_argument('--model_config', type=str, default='config/model_config.yaml', help='Path to model configuration file')
    parser.add_argument('--data_config', type=str, default='config/data_config.yaml', help='Path to data configuration file')
    parser.add_argument('--inference_config', type=str, default='config/inference_config.yaml', help='Path to inference configuration file')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint file')
    parser.add_argument('--prompt', type=str, required=True, help='Initial text prompt for generation')
    args = parser.parse_args()

    setup_logging() # Setup basic logging

    # Load configurations
    try:
        with open(args.model_config, 'r') as f:
            model_config = yaml.safe_load(f)
        with open(args.data_config, 'r') as f:
            data_config = yaml.safe_load(f)
        with open(args.inference_config, 'r') as f:
            inference_config = yaml.safe_load(f)
    except FileNotFoundError as e:
        logging.error(f"Configuration file not found: {e}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file: {e}")
        sys.exit(1)


    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Check if CUDA is requested but not available
    if 'device' in inference_config and inference_config['device'] == 'cuda' and not torch.cuda.is_available():
        logging.warning("CUDA device requested in inference_config but not available. Falling back to CPU.")
        device = torch.device('cpu')
    elif 'device' in inference_config and inference_config['device'] != 'auto': # Allow 'auto' or specific 'cpu'/'cuda'
         device = torch.device(inference_config['device'])


    logging.info(f"Using device: {device}")


    # Load tokenizer
    processed_dir = data_config.get('processed_data_dir', 'data/processed') # Use .get for default
    vocab_path = os.path.join(processed_dir, 'vocab.json')
    if not os.path.exists(vocab_path):
         logging.error(f"Vocabulary file not found at {vocab_path}. Run preprocess_data.py first.")
         sys.exit(1)
    try:
        tokenizer = CharTokenizer.load_vocab(vocab_path)
    except Exception as e:
         logging.error(f"Error loading tokenizer vocabulary from {vocab_path}: {e}")
         sys.exit(1)

    logging.info(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

    # Update model config with values determined from data/tokenizer
    model_config['vocab_size'] = tokenizer.vocab_size
    # --- FIX --- Add seq_length to model_config, crucial for model initialization
    model_config['seq_length'] = data_config.get('seq_length', 128) # Use .get for default, matching config default
    logging.info(f"Model config vocab size: {model_config['vocab_size']}, seq_length: {model_config['seq_length']}")


    # Build the model
    try:
        model = SimpleTransformerDecoder(**model_config).to(device)
    except Exception as e:
        logging.error(f"Error building the model: {e}")
        sys.exit(1)
    logging.info("Model built.")

    # Load model weights from checkpoint
    try:
        # load_checkpoint function needs model, path, and device
        load_checkpoint(args.checkpoint_path, model, device=device)
        logging.info(f"Model weights loaded from {args.checkpoint_path}")
    except FileNotFoundError:
        logging.error(f"Checkpoint file not found at {args.checkpoint_path}.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading checkpoint {args.checkpoint_path}: {e}")
        sys.exit(1)


    # --- FIX --- Set the model to evaluation mode
    model.eval()
    logging.info("Model set to evaluation mode.")

    # Encode the prompt
    prompt_token_ids = tokenizer.encode(args.prompt)

    if not prompt_token_ids:
        logging.warning("Prompt resulted in empty token list (might contain only unknown characters). Cannot generate.")
        # Optionally, you could add a default start token here if applicable
        return

    # Convert to tensor and move to device
    # Unsqueeze adds a batch dimension of 1
    prompt_tensor = torch.tensor([prompt_token_ids], dtype=torch.long, device=device) # Shape (1, prompt_length)

    logging.info(f"Generating from prompt: '{args.prompt}'")
    logging.info(f"Prompt token IDs: {prompt_token_ids}")
    logging.info(f"Prompt tensor shape: {prompt_tensor.shape}")


    # Get generation parameters from inference_config
    max_new_tokens = inference_config.get('max_new_tokens', 100) # Default to 100
    temperature = inference_config.get('temperature', 1.0)       # Default to 1.0
    top_k = inference_config.get('top_k')                        # Default to None

    logging.info(f"Generation parameters: max_new_tokens={max_new_tokens}, temperature={temperature}, top_k={top_k}")


    # Generate text using the model's generate method
    try:
        # The generate method takes the prompt tensor and generation parameters
        generated_tokens = model.generate(
            prompt_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )
    except Exception as e:
        logging.error(f"Error during text generation: {e}")
        # You might want to print partial generation or the error
        return


    # Decode the generated tokens
    # generated_tokens shape: (batch_size, prompt_length + max_new_tokens)
    # Since we use batch_size=1 for inference, take the first row [0]
    generated_ids_list = generated_tokens[0].tolist() # Get the single sequence as a list

    # Decode the list of token IDs back into a string
    generated_text = tokenizer.decode(generated_ids_list)

    print("\n" + "=" * 40)
    print("Generated Text:")
    print(generated_text)
    print("=" * 40 + "\n")


if __name__ == '__main__':
    main()
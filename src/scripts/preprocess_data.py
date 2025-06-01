
import argparse
import yaml
import logging
import os
import numpy as np
import sys
import random


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, ByteLevel


def setup_logging_custom():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def train_tokenizer_and_tokenize_data(data_config):
    raw_data_path = data_config['raw_data_path']
    processed_dir = data_config['processed_data_dir']
    vocab_size_target = data_config.get('tokenizer_vocab_size', 32000)
    split_ratio = data_config.get('train_val_split_ratio', 0.9)
    tokenizer_type = data_config.get('tokenizer_engine', 'bpe_bytelevel')
    lines_per_chunk_tokenizer_train = data_config.get('lines_per_chunk_tokenizer_train', 100000)
    lines_per_chunk_tokenize_data = data_config.get('lines_per_chunk_tokenize_data', 50000)

    logging.info(f"Starting data preprocessing with config: {data_config}")
    os.makedirs(processed_dir, exist_ok=True)


    tokenizer_save_path = os.path.join(processed_dir, 'tokenizer.json')

    if os.path.exists(tokenizer_save_path) and not data_config.get('force_retrain_tokenizer', False):
        logging.info(f"Tokenizer already exists at {tokenizer_save_path}. Loading it.")
        tokenizer = Tokenizer.from_file(tokenizer_save_path)
    else:
        logging.info(f"Training a new tokenizer. Target vocab size: {vocab_size_target}")
        if not os.path.exists(raw_data_path):
            logging.error(f"Raw data file not found: {raw_data_path}")
            sys.exit(1)

        if tokenizer_type == 'bpe_whitespace':
            tokenizer = Tokenizer(BPE(unk_token="<unk>"))
            tokenizer.pre_tokenizer = Whitespace()
        elif tokenizer_type == 'bpe_bytelevel':
            tokenizer = Tokenizer(BPE(unk_token="<unk>"))
            tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=data_config.get('bytelevel_add_prefix_space', True), use_regex=True)
        else:
            logging.error(f"Unsupported tokenizer_engine: {tokenizer_type}")
            sys.exit(1)

        special_tokens = ["<unk>", "<pad>", "<bos>", "<eos>"]
        trainer = BpeTrainer(vocab_size=vocab_size_target, special_tokens=special_tokens,
                             min_frequency=data_config.get('bpe_min_frequency', 2))


        def get_training_corpus_iterator(filepath, lines_per_chunk):
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                chunk = []
                for i, line in enumerate(f):
                    chunk.append(line.strip())
                    if (i + 1) % lines_per_chunk == 0:
                        yield chunk
                        chunk = []
                if chunk:
                    yield chunk
        
        logging.info(f"Training tokenizer on: {raw_data_path} using an iterator.")

        tokenizer.train_from_iterator(get_training_corpus_iterator(raw_data_path, lines_per_chunk_tokenizer_train), trainer=trainer)
        
        tokenizer.save(tokenizer_save_path)
        logging.info(f"Tokenizer trained with actual vocab size {tokenizer.get_vocab_size()} and saved to {tokenizer_save_path}")

    actual_vocab_size = tokenizer.get_vocab_size()
    pad_token_id = tokenizer.token_to_id("<pad>")
    if pad_token_id is None:
        logging.warning("<pad> token not found in tokenizer. Padding might not work as expected.")
        pad_token_id = 0

    logging.info(f"Loaded/trained tokenizer. Actual vocab size: {actual_vocab_size}. Pad token ID: {pad_token_id}")


    train_bin_path = os.path.join(processed_dir, 'train.bin')
    val_bin_path = os.path.join(processed_dir, 'val.bin')

    if (os.path.exists(train_bin_path) and os.path.exists(val_bin_path) and
            not data_config.get('force_retokenize_data', False)):
        logging.info(f"Tokenized .bin files already exist in {processed_dir}. Skipping tokenization.")
    else:
        logging.info(f"Tokenizing {raw_data_path} and saving to .bin files (streaming)...")
        

        dtype = np.uint16 if actual_vocab_size < 65535 else np.uint32
        
        num_train_tokens = 0
        num_val_tokens = 0


        with open(train_bin_path, 'wb') as f_train, open(val_bin_path, 'wb') as f_val, \
             open(raw_data_path, 'r', encoding='utf-8', errors='ignore') as f_raw:
            
            lines_buffer = []
            for i, line in enumerate(f_raw):
                lines_buffer.append(line.strip())
                if (i + 1) % lines_per_chunk_tokenize_data == 0:
                    logging.info(f"Processing lines up to {i+1}...")
                    encodings = tokenizer.encode_batch(lines_buffer)
                    lines_buffer = []

                    for encoded_line in encodings:
                        token_ids = np.array(encoded_line.ids, dtype=dtype)
                        if random.random() < split_ratio:
                            f_train.write(token_ids.tobytes())
                            num_train_tokens += len(token_ids)
                        else:
                            f_val.write(token_ids.tobytes())
                            num_val_tokens += len(token_ids)
                    logging.info(f"  Current tokens: Train={num_train_tokens/1e6:.2f}M, Val={num_val_tokens/1e6:.2f}M")


            if lines_buffer:
                logging.info(f"Processing remaining {len(lines_buffer)} lines...")
                encodings = tokenizer.encode_batch(lines_buffer)
                for encoded_line in encodings:
                    token_ids = np.array(encoded_line.ids, dtype=dtype)
                    if random.random() < split_ratio:
                        f_train.write(token_ids.tobytes())
                        num_train_tokens += len(token_ids)
                    else:
                        f_val.write(token_ids.tobytes())
                        num_val_tokens += len(token_ids)

        logging.info(f"Tokenization complete.")
        logging.info(f"Total Train tokens: {num_train_tokens} ({num_train_tokens/1e6:.2f}M)")
        logging.info(f"Total Validation tokens: {num_val_tokens} ({num_val_tokens/1e6:.2f}M)")
        logging.info(f"Train tokens saved to {train_bin_path}")
        logging.info(f"Validation tokens saved to {val_bin_path}")

    logging.info("Data preprocessing finished.")
    logging.info(f"Tokenizer: {tokenizer_save_path}")
    logging.info(f"Actual vocab size (from tokenizer): {actual_vocab_size}")


def main():
    setup_logging_custom()
    parser = argparse.ArgumentParser(description="Preprocess raw data using subword tokenization")
    parser.add_argument('--config', type=str, default='config/data_config.yaml',
                        help='Path to data configuration YAML file (e.g., data_config.yaml)')
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config_params = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file {args.config}: {e}")
        sys.exit(1)

    if 'raw_data_path' not in config_params or 'processed_data_dir' not in config_params:
        logging.error("Config file must contain 'raw_data_path' and 'processed_data_dir'.")
        sys.exit(1)

    train_tokenizer_and_tokenize_data(config_params)

if __name__ == "__main__":
    main()
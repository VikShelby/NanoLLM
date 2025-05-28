import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
import numpy as np
import logging

class CharTokenizer:
    """Basic character-level tokenizer."""
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)} # string to integer
        self.itos = {i: ch for i, ch in enumerate(chars)} # integer to string
        self.vocab_size = len(chars)

    def encode(self, s):
        """Encodes a string into a list of integers."""
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, l):
        """Decodes a list of integers into a string."""
        return ''.join([self.itos[i] for i in l if i in self.itos])

    def save_vocab(self, filepath):
        """Saves the vocabulary to a JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({'stoi': self.stoi, 'itos': self.itos}, f, ensure_ascii=False, indent=4)

    @classmethod
    def load_vocab(cls, filepath):
        """Loads the vocabulary from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        tokenizer = cls.__new__(cls) # Create an instance without calling __init__
        tokenizer.stoi = vocab_data['stoi']
        tokenizer.itos = {int(k): v for k, v in vocab_data['itos'].items()}
        tokenizer.vocab_size = len(tokenizer.stoi)
        return tokenizer


class LLMDataset(Dataset):
    """Dataset for next-token prediction."""
    def __init__(self, token_ids, seq_length):
        self.token_ids = token_ids
        self.seq_length = seq_length

        # Ensure we have enough data for at least one sequence
        if len(token_ids) < seq_length + 1:
             raise ValueError(f"Dataset size ({len(token_ids)}) is too small for seq_length ({seq_length}). Need at least {seq_length + 1} tokens.")

    def __len__(self):
        # We can create (len(token_ids) - seq_length) samples
        return len(self.token_ids) - self.seq_length

    def __getitem__(self, idx):
        # Get a chunk of length seq_length + 1
        chunk = self.token_ids[idx : idx + self.seq_length + 1]
        # Input is the first seq_length tokens
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        # Target is the next token
        y = torch.tensor(chunk[1:], dtype=torch.long) # Predict the *next* token for each token in x
        return x, y

def preprocess_and_save(config):
    """Reads raw data, creates tokenizer, tokenizes, and saves processed data."""
    raw_path = config['raw_data_path']
    processed_dir = config['processed_data_dir']
    tokenizer_type = config['tokenizer_type'] # 'char' in this example
    seq_length = config['seq_length']

    logging.info(f"Loading data from {raw_path}")
    with open(raw_path, 'r', encoding='utf-8') as f:
        text = f.read()

    os.makedirs(processed_dir, exist_ok=True)

    if tokenizer_type == 'char':
        tokenizer = CharTokenizer(text)
        logging.info(f"Character tokenizer created. Vocab size: {tokenizer.vocab_size}")
        vocab_path = os.path.join(processed_dir, 'vocab.json')
        tokenizer.save_vocab(vocab_path)
        logging.info(f"Vocabulary saved to {vocab_path}")
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

    logging.info("Tokenizing data...")
    token_ids = tokenizer.encode(text)
    token_ids_path = os.path.join(processed_dir, 'tokenized_data.pt')
    torch.save(torch.tensor(token_ids, dtype=torch.long), token_ids_path)
    logging.info(f"Tokenized data saved to {token_ids_path}")
    logging.info(f"Total tokens: {len(token_ids)}")

    # Basic check for dataset size feasibility
    if len(token_ids) < seq_length + 1:
         logging.warning(f"Total tokens ({len(token_ids)}) is less than sequence length + 1 ({seq_length + 1}). Training may not be possible.")


def create_dataloaders(processed_dir, seq_length, batch_size):
    """Loads processed data and creates DataLoaders."""
    token_ids_path = os.path.join(processed_dir, 'tokenized_data.pt')
    vocab_path = os.path.join(processed_dir, 'vocab.json')

    if not os.path.exists(token_ids_path) or not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Processed data not found in {processed_dir}. Run preprocess_data.py first.")

    logging.info(f"Loading tokenized data from {token_ids_path}")
    token_ids = torch.load(token_ids_path).tolist() # Load as list for dataset indexing

    tokenizer = CharTokenizer.load_vocab(vocab_path) # Load tokenizer separately if needed later

    # Splitting data: Simplistic split (e.g., 90% train, 10% val)
    # For large datasets, need a more robust split strategy (e.g., by document, not just a linear split)
    split_idx = int(len(token_ids) * 0.9)
    train_token_ids = token_ids[:split_idx]
    val_token_ids = token_ids[split_idx:]

    logging.info(f"Train tokens: {len(train_token_ids)}, Validation tokens: {len(val_token_ids)}")

    train_dataset = LLMDataset(train_token_ids, seq_length)
    val_dataset = LLMDataset(val_token_ids, seq_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    logging.info(f"Created train DataLoader with {len(train_loader)} batches of size {batch_size}")
    logging.info(f"Created validation DataLoader with {len(val_loader)} batches of size {batch_size}")

    return train_loader, val_loader, tokenizer  
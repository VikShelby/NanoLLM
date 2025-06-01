
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import logging
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from tokenizers import Tokenizer

class PretokenizedDataset(Dataset):
    """
    Dataset for loading pre-tokenized data from a .bin file (memory-mapped).
    Each item returns a sequence of `seq_length` for input and `seq_length` for target.
    """
    def __init__(self, data_file, seq_length, block_size_for_len_calc=None):
        super().__init__()
        self.seq_length = seq_length

        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")








        
        logging.info(f"Loading pretokenized data from {data_file} with mmap.")












        











        




        







        self.token_ids = None
        self.num_effective_samples = 0


    def set_data(self, token_array_mmap):
        self.token_ids = token_array_mmap
        if len(self.token_ids) <= self.seq_length:
            logging.warning(f"Dataset (len {len(self.token_ids)}) is too short for seq_length ({self.seq_length}). Effective samples: 0")
            self.num_effective_samples = 0
        else:
            self.num_effective_samples = len(self.token_ids) - self.seq_length
        logging.info(f"Dataset set with {len(self.token_ids)} tokens. Effective samples: {self.num_effective_samples} for seq_length {self.seq_length}")


    def __len__(self):

        return self.num_effective_samples

    def __getitem__(self, idx):
        if self.token_ids is None:
            raise RuntimeError("Dataset not initialized with data. Call set_data() first.")
        if idx >= self.num_effective_samples:
            raise IndexError("Index out of bounds")



        chunk = self.token_ids[idx : idx + self.seq_length + 1]
        

        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def create_dataloaders(data_config, training_batch_size):
    """
    Loads the trained subword tokenizer and pre-tokenized .bin data,
    then creates train and validation DataLoaders.

    Args:
        data_config (dict): Configuration dictionary (from data_config.yaml).
        training_batch_size (int): Batch size for the DataLoaders.
    """
    processed_dir = data_config['processed_data_dir']
    seq_length = data_config['seq_length']

    tokenizer_path = os.path.join(processed_dir, 'tokenizer.json')
    train_bin_path = os.path.join(processed_dir, 'train.bin')
    val_bin_path = os.path.join(processed_dir, 'val.bin')

    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}. Run preprocess_data.py first.")
    if not os.path.exists(train_bin_path):
        raise FileNotFoundError(f"Train data file not found: {train_bin_path}. Run preprocess_data.py first.")
    if not os.path.exists(val_bin_path):
        raise FileNotFoundError(f"Validation data file not found: {val_bin_path}. Run preprocess_data.py first.")

    logging.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    actual_vocab_size = tokenizer.get_vocab_size()
    logging.info(f"Tokenizer loaded. Vocab size: {actual_vocab_size}")



    dtype = np.uint16 if actual_vocab_size < 65535 else np.uint32

    logging.info(f"Loading pre-tokenized data using mmap (dtype: {dtype})...")
    try:
        train_ids_mmap = np.memmap(train_bin_path, dtype=dtype, mode='r')
        val_ids_mmap = np.memmap(val_bin_path, dtype=dtype, mode='r')
    except Exception as e:
        logging.error(f"Error memory-mapping .bin files. Ensure they exist and dtype {dtype} is correct: {e}")
        raise

    logging.info(f"Train tokens via mmap: {len(train_ids_mmap)}, Validation tokens via mmap: {len(val_ids_mmap)}")


    train_dataset = PretokenizedDataset(data_file=train_bin_path, seq_length=seq_length)
    train_dataset.set_data(train_ids_mmap)

    val_dataset = PretokenizedDataset(data_file=val_bin_path, seq_length=seq_length)
    val_dataset.set_data(val_ids_mmap)
    
    if len(train_dataset) == 0:
        logging.warning("Train dataset is empty after processing. Check data and seq_length.")
    if len(val_dataset) == 0:
        logging.warning("Validation dataset is empty after processing. Check data and seq_length.")





    num_workers = data_config.get('dataloader_num_workers', 0)
    pin_memory = torch.cuda.is_available() and data_config.get('dataloader_pin_memory', True)


    train_loader = DataLoader(
        train_dataset,
        batch_size=training_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    logging.info(f"Created train DataLoader with ~{len(train_loader)} batches of size {training_batch_size}")
    logging.info(f"Created validation DataLoader with ~{len(val_loader)} batches of size {training_batch_size}")

    return train_loader, val_loader, tokenizer
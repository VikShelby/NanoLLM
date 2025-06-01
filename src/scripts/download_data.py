from datasets import load_dataset, Value
import os
import logging
import psutil # For checking disk space

# --- Configuration ---
TARGET_TOTAL_GB = 100 # Your desired total GB of raw text

# Approximate average bytes per document (VERY ROUGH ESTIMATE - adjust based on observation)
# This is used if we want to limit by number of documents when streaming.
# C4 documents can vary wildly. Let's say 3000 bytes (3KB) on average.
# OpenWebText might be similar or slightly smaller.
AVG_BYTES_PER_DOC_C4 = 3000
AVG_BYTES_PER_DOC_OWT = 2500

# Calculate target number of documents based on GB and avg size (can be an alternative to max_gb_save)
# For C4: 80GB target / (3000 bytes/doc) = (80 * 1024**3) / 3000 docs
# This gives a very large number of docs. The max_gb_save is probably more direct.

DATA_SAVE_DIR = "text_datasets_streamed_100gb"
COMBINED_RAW_INPUT_TXT_TARGET = "../data/raw/input_streamed_100gb.txt" # Adjust path as needed

# Datasets configuration
# We'll try to get ~80GB from C4 and ~20GB from OpenWebText
# The 'max_gb_save' will be the primary control for output file size.
# 'num_examples_to_take' is a secondary control if using streaming heavily,
# but can be set to None to rely solely on max_gb_save.

DATASETS_CONFIG = [
    {
        "name": "c4_english_slice",
        "hf_id": "allenai/c4", # Using allenai/c4 as it's commonly referenced
        "hf_name": "en", # Or None if not applicable for the specific hf_id
        "split": "train", # We stream from the beginning of train
        "trust_remote_code": True,
        "text_column": "text",
        "max_gb_save": 80.0, # Target GB for the output .txt file
        "num_examples_to_take": None # Let max_gb_save control it primarily
                                     # Or estimate: int((80 * 1024**3) / AVG_BYTES_PER_DOC_C4)
    },
    {
        "name": "openwebtext_slice",
        "hf_id": "openwebtext",
        "hf_name": None,
        "split": "train",
        "trust_remote_code": True,
        "text_column": "text",
        "max_gb_save": 20.0,
        "num_examples_to_take": None # Let max_gb_save control it
                                     # Or estimate: int((20 * 1024**3) / AVG_BYTES_PER_DOC_OWT)
    }
]

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.makedirs(DATA_SAVE_DIR, exist_ok=True)

def get_available_disk_space_gb(path="."):
    try:
        hdd = psutil.disk_usage(path)
        return hdd.free / (1024**3)
    except Exception as e:
        logging.warning(f"Could not check disk space: {e}")
        return float('inf') # Assume enough space if check fails

def save_to_txt_streaming(dataset_iterable, filename_base, text_column, max_gb=None, num_examples_limit=None):
    filepath = os.path.join(DATA_SAVE_DIR, f"{filename_base}.txt")

    if os.path.exists(filepath):
        logging.info(f"Output file {filepath} already exists. Skipping to save time. Delete it to re-generate.")
        # Optionally, check its size and compare with max_gb if needed
        return filepath, os.path.getsize(filepath) / (1024**3)

    total_bytes_written = 0
    gb_limit_bytes = (max_gb * 1024**3) if max_gb is not None else float('inf')
    lines_written = 0

    # Check initial disk space
    available_space_start_gb = get_available_disk_space_gb(os.path.dirname(filepath) or ".")
    logging.info(f"Available disk space before writing {filename_base}: {available_space_start_gb:.2f} GB")
    if max_gb and max_gb > available_space_start_gb * 0.95: # Leave 5% margin
        logging.error(f"Not enough disk space for {filename_base}. Need ~{max_gb:.2f} GB, have ~{available_space_start_gb:.2f} GB.")
        return None, 0

    logging.info(f"Starting to save {filename_base} to {filepath} (max_gb: {max_gb}, num_examples_limit: {num_examples_limit})")

    with open(filepath, "w", encoding="utf-8") as f:
        for i, row in enumerate(dataset_iterable):
            if num_examples_limit is not None and i >= num_examples_limit:
                logging.info(f"{filename_base}: Reached document limit of {num_examples_limit}. Stopping.")
                break

            text = row.get(text_column) if isinstance(row, dict) and text_column in row else ""
            # Handle if row itself is a string (some simpler datasets might yield strings)
            if isinstance(row, str) and text_column == "text": # A common default
                text = row
            
            if not text or not isinstance(text, str): # Skip empty or non-string entries
                if i % 50000 == 0 : logging.debug(f"Skipped empty/invalid row {i} for {filename_base}")
                continue
            
            line_to_write = text.strip() + "\n"
            try:
                line_bytes = len(line_to_write.encode('utf-8'))
            except UnicodeEncodeError:
                logging.warning(f"UnicodeEncodeError for a line in {filename_base} at doc {i}. Skipping line.")
                continue


            if total_bytes_written + line_bytes > gb_limit_bytes:
                logging.info(f"{filename_base}: Reached approximately {max_gb:.2f}GB limit ({total_bytes_written / (1024**3):.2f}GB written). Stopping.")
                break
            
            try:
                f.write(line_to_write)
                total_bytes_written += line_bytes
                lines_written +=1
            except Exception as e_write:
                logging.error(f"Error writing to file {filepath} at line {lines_written}: {e_write}")
                logging.info("Attempting to close file and stop for this dataset.")
                break


            if lines_written % 50000 == 0 and lines_written > 0: # Log progress less frequently for speed
                current_gb = total_bytes_written / (1024**3)
                logging.info(f"{filename_base}: Saved {lines_written} lines, ~{current_gb:.2f} GB written. (Processed {i+1} input documents)")
        
        final_gb = total_bytes_written / (1024**3)
        logging.info(f"{filename_base}: Finished saving. Total lines written: {lines_written}. Total output size: ~{final_gb:.2f} GB.")
    return filepath, final_gb

# --- Main Processing Logic ---
saved_files_list = []
total_gb_written_all_files = 0.0

# Optional: Clean Hugging Face Hub cache for specific datasets if you want a fresh start
# from huggingface_hub import HfApi, snapshot_download
# hf_api = HfApi()
# def clear_hf_dataset_cache(repo_id):
#     try:
#         # This is a bit tricky, as 'delete_repo' is for user repos.
#         # Easiest is manual deletion of ~/.cache/huggingface/hub/models--repo--id or datasets--repo--id
#         cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
#         dataset_cache_path = os.path.join(cache_dir, f"datasets--{repo_id.replace('/', '--')}")
#         if os.path.exists(dataset_cache_path):
#             logging.info(f"Consider manually deleting cache for {repo_id} at {dataset_cache_path} for a completely fresh download.")
#             # import shutil
#             # shutil.rmtree(dataset_cache_path)
#             # logging.info(f"Deleted cache {dataset_cache_path}")
#     except Exception as e:
#         logging.warning(f"Could not clear cache for {repo_id}: {e}")

# for ds_config in DATASETS_CONFIG:
#    clear_hf_dataset_cache(ds_config["hf_id"]) # UNCOMMENT TO ATTEMPT CACHE CLEARING (MANUAL IS SAFER)


for ds_config in DATASETS_CONFIG:
    logging.info(f"\n--- Processing dataset: {ds_config['name']} ---")
    
    try:
        logging.info(f"Loading dataset: id='{ds_config['hf_id']}', name='{ds_config.get('hf_name', 'N/A')}', split='{ds_config['split']}', streaming=True")
        # Use streaming=True
        # For C4, some configurations might not stream perfectly or might require specific sub-configs.
        # 'allenai/c4' with 'en' name should generally support streaming.
        dataset = load_dataset(
            ds_config["hf_id"],
            name=ds_config.get("hf_name"),
            split=ds_config["split"],
            streaming=True, # KEY CHANGE
            trust_remote_code=ds_config["trust_remote_code"]
        )
        logging.info(f"Successfully created streaming dataset object for {ds_config['name']}.")

        # If we have a num_examples_to_take limit for streaming
        # This is an alternative/complement to max_gb_save for streamed datasets
        # dataset_iterable = dataset.take(ds_config["num_examples_to_take"]) if ds_config.get("num_examples_to_take") else dataset
        dataset_iterable = dataset # Iterate full stream, save_to_txt will limit by GB

        filepath, gb_written = save_to_txt_streaming(
            dataset_iterable,
            ds_config["name"],
            ds_config["text_column"],
            max_gb=ds_config.get("max_gb_save"),
            num_examples_limit=ds_config.get("num_examples_to_take") # Pass this along
        )
        if filepath:
            saved_files_list.append(filepath)
            total_gb_written_all_files += gb_written
        else:
            logging.warning(f"Failed to save or skipped {ds_config['name']}.")


    except Exception as e:
        logging.error(f"Could not load or process {ds_config['name']}. Error: {e}", exc_info=True)
        logging.info(f"Skipping dataset {ds_config['name']}.")

logging.info(f"\n--- ✅ Script Finished ---")
logging.info(f"Output text files are in: '{DATA_SAVE_DIR}'")
for f_path in saved_files_list:
    actual_size_gb = os.path.getsize(f_path) / (1024**3)
    logging.info(f" - {f_path} ({actual_size_gb:.2f} GB)")
logging.info(f"Total data written to .txt files: ~{total_gb_written_all_files:.2f} GB")
logging.info(f"\nNext steps:")
logging.info(f"1. Inspect the .txt files in '{DATA_SAVE_DIR}'.")
logging.info(f"2. If satisfied, concatenate them into a single input file for your model (e.g., to '{COMBINED_RAW_INPUT_TXT_TARGET}').")
logging.info(f"   Example (Linux/macOS): cat {os.path.join(DATA_SAVE_DIR, '*.txt')} > {COMBINED_RAW_INPUT_TXT_TARGET}")

# --- Start of fix ---
# Pre-calculate the Windows-style path for the CMD example log message
win_cmd_target_path = COMBINED_RAW_INPUT_TXT_TARGET.replace('/', '\\')
logging.info(f"   Example (Windows CMD): copy /b {os.path.join(DATA_SAVE_DIR, '*.txt')} {win_cmd_target_path}") # Adjust for copy
# --- End of fix ---

logging.info(f"   For Windows, `copy /b` might be slow. PowerShell: Get-Content {os.path.join(DATA_SAVE_DIR, '*.txt')} | Set-Content {COMBINED_RAW_INPUT_TXT_TARGET}")
logging.info(f"3. Then run your 'src/scripts/preprocess_data.py' script on this combined file.")
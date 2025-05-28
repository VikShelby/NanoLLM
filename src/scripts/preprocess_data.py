import argparse
import yaml
import logging
import os
from nano_llm.data_processing import preprocess_and_save

def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Preprocess raw data")
    parser.add_argument('--data_config', type=str, default='config/data_config.yaml', help='Path to data config YAML')
    args = parser.parse_args()

    with open(args.data_config, 'r') as f:
        data_config = yaml.safe_load(f)

    os.makedirs(data_config['processed_data_dir'], exist_ok=True)

    preprocess_and_save(data_config)

if __name__ == "__main__":
    main()

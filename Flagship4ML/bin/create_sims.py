#!/usr/bin/env python3

import argparse
import yaml
import subprocess
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Run sims_generator.py with config file')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error(f"Error reading config file: {e}")
            sys.exit(1)

def build_command(config):
    # Start with the base command
    cmd = ['python', 'sims_generator.py']
    
    # Add each config parameter as a command-line argument
    for key, value in config.items():
        if value is None:
            continue
        # Convert boolean values to lowercase strings
        if isinstance(value, bool):
            value = str(value).lower()
        # Add the argument
        cmd.extend([f'--{key}', str(value)])
    
    return cmd

def main():
    args = parse_args()
    config = load_config(args.config)
    
    try:
        from Flagship4ML.f4ml.sims_generator import CreateSimulatedImages
        logger.info("Creating simulated images...")
        CreateSimulatedImages(**config).generate_simulated_catalogue()
    except Exception as e:
        logger.error(f"Error creating simulated images: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()

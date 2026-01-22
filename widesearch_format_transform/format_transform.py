#!/usr/bin/env python3
"""
Convert MiroThinker rollout logs to WideSearch evaluation format.

This script:
1. Reads task_*.json files from run_1, run_2, run_3, run_4 directories
2. Extracts the final assistant response from message history
3. Converts to WideSearch response format (jsonl files)
4. Outputs to widesearch_format/ directory in the log folder
"""

import json
import os
import re
import glob
import argparse
from pathlib import Path


def extract_instance_id(task_id: str) -> str:
    """Extract instance_id from task_id (e.g., 'ws_en_001_attempt-1_format-retry-0' -> 'ws_en_001')"""
    match = re.match(r'(ws_(?:en|zh)_\d+)', task_id)
    if match:
        return match.group(1)
    return task_id.split('_attempt')[0]


def convert_single_file(task_json_path: str, trial_idx: int) -> dict:
    """Convert a single MiroThinker task JSON to WideSearch response format."""
    with open(task_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    task_id = data.get('task_id', '')
    instance_id = extract_instance_id(task_id)
    response = data.get('final_boxed_answer', '')

    return {
        'instance_id': instance_id,
        'response': response,
        'trial_idx': trial_idx
    }


def convert_run_folder(run_folder: str, trial_idx: int, output_dir: str, model_config_name: str):
    """Convert all task JSON files in a run folder to WideSearch format."""
    task_files = glob.glob(os.path.join(run_folder, 'task_*.json'))

    converted_count = 0
    for task_file in task_files:
        try:
            response_data = convert_single_file(task_file, trial_idx)
            instance_id = response_data['instance_id']

            # Output file naming: {model_config_name}_{instance_id}_{trial_idx}_response.jsonl
            output_filename = f"{model_config_name}_{instance_id}_{trial_idx}_response.jsonl"
            output_path = os.path.join(output_dir, output_filename)

            # Write as jsonl (single line per file for WideSearch compatibility)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(response_data, f, ensure_ascii=False)
                f.write('\n')

            converted_count += 1
        except Exception as e:
            print(f"Error processing {task_file}: {e}")

    return converted_count


def main():
    parser = argparse.ArgumentParser(description='Convert MiroThinker logs to WideSearch format')
    parser.add_argument(
        '--log_dir',
        type=str,
        default='/mnt/project_rlinf/zhangruize/project/MAS/MiroThinker/logs/widesearch/2026-01-21-16-37-13_qwen_qwen-3_mirothinker_v1.0_keep5_widesearch_nulltasks_4runs',
        help='Path to the MiroThinker log directory containing run_1, run_2, etc.'
    )
    parser.add_argument(
        '--model_config_name',
        type=str,
        default='mirothinker',
        help='Model config name to use in output filenames'
    )
    parser.add_argument(
        '--output_subdir',
        type=str,
        default='widesearch_format',
        help='Subdirectory name for output files'
    )

    args = parser.parse_args()

    log_dir = args.log_dir
    model_config_name = args.model_config_name
    output_dir = os.path.join(log_dir, args.output_subdir)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Map run folders to trial indices
    run_folders = [
        ('run_1', 0),
        ('run_2', 1),
        ('run_3', 2),
        ('run_4', 3),
    ]

    total_converted = 0
    for run_name, trial_idx in run_folders:
        run_folder = os.path.join(log_dir, run_name)
        if os.path.exists(run_folder):
            count = convert_run_folder(run_folder, trial_idx, output_dir, model_config_name)
            print(f"Converted {count} files from {run_name} (trial_idx={trial_idx})")
            total_converted += count
        else:
            print(f"Warning: {run_folder} does not exist, skipping")

    print(f"\nTotal converted: {total_converted} files")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()

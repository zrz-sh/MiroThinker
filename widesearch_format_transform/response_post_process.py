#!/usr/bin/env python3
"""
Post-process WideSearch responses to ensure proper markdown format.

This script:
1. Reads response files from widesearch_format/
2. Uses GPT-4.1-mini to check and fix markdown format
3. Outputs to widesearch_format_post_processed/
"""

import json
import os
import glob
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI


def fix_response_with_llm(response: str, client: OpenAI, model: str = "gpt-4.1-mini") -> str:
    """Use LLM to fix response format."""
    # Skip empty or error responses
    if not response or "No \\boxed{} content found" in response or "No boxed content found" in response:
        return response

    # Already has proper wrapper
    if response.strip().startswith("```markdown") and response.strip().endswith("```"):
        return response

    prompt = f"""You are a format fixer. The response below should be a markdown table wrapped with ```markdown at the start and ``` at the end.

Your task:
1. If the response already has correct format (starts with ```markdown and ends with ```), return it exactly as-is.
2. If the response contains table data but is missing the wrapper, wrap it with ```markdown at the beginning and ``` at the end.
3. Do NOT modify the table content itself, only add the wrapper if missing.

IMPORTANT: Only output the fixed response, nothing else. No explanations.

Response to fix:
{response}"""

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=8192
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM call failed: {e}")
        return response


def process_single_file(input_path: str, output_path: str, client: OpenAI, model: str) -> dict:
    """Process a single response file."""
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.loads(f.readline())

    response = data.get('response', '')
    original_response = response

    # Use LLM to fix format
    response = fix_response_with_llm(response, client, model)
    data['response'] = response

    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
        f.write('\n')

    return {
        'file': os.path.basename(input_path),
        'modified': original_response != response
    }


def main():
    parser = argparse.ArgumentParser(description='Post-process WideSearch responses')
    parser.add_argument(
        '--input_dir',
        type=str,
        default='/mnt/project_rlinf/zhangruize/project/MAS/MiroThinker/logs/widesearch/2026-01-21-16-37-13_qwen_qwen-3_mirothinker_v1.0_keep5_widesearch_nulltasks_4runs/widesearch_format',
        help='Input directory containing response files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/mnt/project_rlinf/zhangruize/project/MAS/MiroThinker/logs/widesearch/2026-01-21-16-37-13_qwen_qwen-3_mirothinker_v1.0_keep5_widesearch_nulltasks_4runs/widesearch_format_post_processed',
        help='Output directory for post-processed files'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4.1-mini',
        help='OpenAI model to use'
    )
    parser.add_argument(
        '--openai_base_url',
        type=str,
        default='https://api.openai.com/v1',
        help='OpenAI API base URL'
    )
    parser.add_argument(
        '--thread_num',
        type=int,
        default=8,
        help='Number of threads for parallel processing'
    )

    args = parser.parse_args()

    # Get API key from environment
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key, base_url=args.openai_base_url)

    # Get all input files
    input_files = glob.glob(os.path.join(args.input_dir, '*.jsonl'))
    print(f"Found {len(input_files)} files to process")

    modified_count = 0

    # Use thread pool for parallel LLM calls
    with ThreadPoolExecutor(max_workers=args.thread_num) as executor:
        futures = {}
        for input_path in input_files:
            filename = os.path.basename(input_path)
            output_path = os.path.join(args.output_dir, filename)
            future = executor.submit(process_single_file, input_path, output_path, client, args.model)
            futures[future] = filename

        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result['modified']:
                modified_count += 1
                print(f"[{i+1}/{len(input_files)}] Modified: {result['file']}")
            else:
                if (i + 1) % 100 == 0:
                    print(f"[{i+1}/{len(input_files)}] Progress...")

    print(f"\nProcessing complete!")
    print(f"Total files: {len(input_files)}")
    print(f"Modified files: {modified_count}")
    print(f"Output directory: {args.output_dir}")

    # Generate eval shell script
    generate_eval_script(args.output_dir)


def generate_eval_script(output_dir: str, model_config_name: str = "mirothinker"):
    """Generate the eval.sh script for WideSearch evaluation."""
    log_dir = os.path.dirname(output_dir)
    result_save_dir = os.path.join(log_dir, 'widesearch_eval_results')

    script_content = f'''#!/bin/bash
# WideSearch Evaluation Script
# Generated by response_post_process.py

# Configuration
WIDESEARCH_ROOT="/mnt/project_rlinf/zhangruize/project/MAS/WideSearch"
RESPONSE_ROOT="{output_dir}"
RESULT_SAVE_ROOT="{result_save_dir}"
MODEL_CONFIG_NAME="{model_config_name}"
TRIAL_NUM=4

# Create result directory
mkdir -p "$RESULT_SAVE_ROOT"

# Change to WideSearch directory
cd "$WIDESEARCH_ROOT"

# Run evaluation (eval only, since responses already exist)
python scripts/run_infer_and_eval_batching.py \\
    --stage eval \\
    --model_config_name "$MODEL_CONFIG_NAME" \\
    --response_root "$RESPONSE_ROOT" \\
    --result_save_root "$RESULT_SAVE_ROOT" \\
    --trial_num $TRIAL_NUM \\
    --use_cache \\
    --thread_num 8

echo "Evaluation completed!"
echo "Results saved to: $RESULT_SAVE_ROOT"
'''

    script_path = os.path.join(log_dir, 'run_widesearch_eval.sh')
    with open(script_path, 'w') as f:
        f.write(script_content)

    # Make script executable
    os.chmod(script_path, 0o755)

    print(f"Generated eval script: {script_path}")


if __name__ == '__main__':
    main()

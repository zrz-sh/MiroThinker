#!/usr/bin/env python3
"""
Append final_boxed_answer for multi-agent tasks that don't have one.

This script:
1. Finds multi-agent task JSON files with "No \\boxed{} content found in the final answer."
2. Uses LLM to generate a final answer based on main_agent_message_history.message_history and task_description
3. Saves to run_X_append_final_box_answer/ directories (does not overwrite original files)
"""

import json
import os
import re
import glob
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI


def extract_boxed_content(text: str) -> str:
    """Extract markdown table content from response."""
    # First try to find ```markdown ... ``` block BEFORE removing anything
    markdown_pattern = r'```markdown\s*(.*?)\s*```'
    match = re.search(markdown_pattern, text, re.DOTALL)
    if match:
        return f"```markdown\n{match.group(1).strip()}\n```"

    # Try to find \boxed{...} pattern
    pattern = r'\\boxed\{(.*)\}'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        content = match.group(1).strip()
        if content.startswith("```markdown"):
            return content
        return f"```markdown\n{content}\n```"

    # Try to find any table (lines starting with |) in the original text
    lines = text.split('\n')
    table_lines = []
    in_table = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('|'):
            in_table = True
            table_lines.append(stripped)
        elif in_table and not stripped:
            continue  # Allow empty lines within table
        elif in_table and not stripped.startswith('|') and stripped:
            break  # End of table on non-empty non-table line

    if table_lines:
        return f"```markdown\n" + "\n".join(table_lines) + "\n```"

    # Return empty if no valid content found
    return ""


def generate_final_answer(task_description: str, message_history: list, client: OpenAI, model: str) -> str:
    """Use LLM to generate a final answer based on conversation history."""

    # Extract only assistant messages (they contain all the gathered information)
    assistant_contents = []
    for msg in message_history:
        role = msg.get('role', '')
        content = msg.get('content', '')
        if role == 'assistant' and isinstance(content, str) and content:
            assistant_contents.append(content)

    # Join all assistant messages and truncate if too long
    all_assistant_text = "\n\n---\n\n".join(assistant_contents)

    # Truncate to avoid exceeding context limit (32k context - 8k output = 24k input tokens ≈ 48k chars, keep ~30k to be safe)
    max_chars = 30000
    if len(all_assistant_text) > max_chars:
        all_assistant_text = "... [earlier content truncated] ...\n\n" + all_assistant_text[-max_chars:]

    prompt = f"""You are a data summarization assistant. The data collection phase is COMPLETE. Your ONLY job now is to SUMMARIZE the already-collected data into a markdown table.

ABSOLUTE RESTRICTIONS - VIOLATION WILL CAUSE FAILURE:
1. NEVER use <use_mcp_tool> tags - ALL TOOLS ARE DISABLED
2. NEVER use <think> tags
3. NEVER output any XML tags
4. NEVER try to search or browse - all data is already provided below
5. ONLY output the final markdown table starting with ```markdown

The original task was:
{task_description}

Here is ALL the data that was already collected (summarize THIS data only, do NOT try to collect more):
{all_assistant_text}

YOUR OUTPUT FORMAT - start directly with:
```markdown
| Column1 | Column2 | Column3 |
|---------|---------|---------|
| data1   | data2   | data3   |
```

Rules:
- Include ALL rows with ANY data
- Use "N/A" for missing cells
- Do NOT skip any rows

Output the markdown table now (start with ```markdown):"""

    try:
        system_msg = "You are a data summarization assistant. Output ONLY a markdown table. ALL TOOLS ARE DISABLED - never use <use_mcp_tool> or any XML tags. Start directly with ```markdown."
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=8192,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )
        response = completion.choices[0].message.content.strip()
        extracted = extract_boxed_content(response)
        if not extracted:
            # Debug: print first 500 chars of response when extraction fails
            print(f"  [DEBUG] Extraction failed. Response preview: {response[:500]}...")
        return extracted
    except Exception as e:
        print(f"LLM call failed: {e}")
        return ""


def process_single_file(input_path: str, output_path: str, client: OpenAI, model: str) -> dict:
    """Process a single task JSON file."""
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    final_answer = data.get('final_boxed_answer', '')

    # Check if needs processing
    if "No \\boxed{} content found" not in final_answer and "No boxed content found" not in final_answer:
        # Already has answer, just copy
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return {
            'file': os.path.basename(input_path),
            'status': 'skipped',
            'had_answer': True
        }

    # Extract task description and message history for multi-agent
    task_description = data.get('input', {}).get('task_description', '')

    # For multi-agent: use main_agent_message_history.message_history
    msg_history = data.get('main_agent_message_history', {})
    messages = msg_history.get('message_history', [])

    if not task_description:
        # Try alternative location
        task_description = data.get('task_description', '')

    if not messages:
        # If no messages found, skip this file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return {
            'file': os.path.basename(input_path),
            'status': 'skipped',
            'had_answer': False,
            'reason': 'no_messages'
        }

    # Generate new answer
    new_answer = generate_final_answer(task_description, messages, client, model)

    if new_answer:
        data['final_boxed_answer'] = new_answer
        data['final_boxed_answer_source'] = 'llm_appended_multi_agent'

    # Save to output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return {
        'file': os.path.basename(input_path),
        'status': 'processed',
        'had_answer': False,
        'new_answer_length': len(new_answer) if new_answer else 0
    }


def process_run_folder(run_folder: str, output_folder: str, client: OpenAI, model: str, thread_num: int):
    """Process all task files in a run folder."""
    os.makedirs(output_folder, exist_ok=True)

    task_files = glob.glob(os.path.join(run_folder, 'task_*.json'))
    print(f"Found {len(task_files)} task files in {os.path.basename(run_folder)}")

    processed_count = 0
    skipped_count = 0

    with ThreadPoolExecutor(max_workers=thread_num) as executor:
        futures = {}
        for input_path in task_files:
            filename = os.path.basename(input_path)
            output_path = os.path.join(output_folder, filename)
            future = executor.submit(process_single_file, input_path, output_path, client, model)
            futures[future] = filename

        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result['status'] == 'processed':
                processed_count += 1
                print(f"[{i+1}/{len(task_files)}] Processed: {result['file']} (new answer length: {result.get('new_answer_length', 0)})")
            else:
                skipped_count += 1
                reason = result.get('reason', 'has_answer')
                if (i + 1) % 50 == 0:
                    print(f"[{i+1}/{len(task_files)}] Progress... (skipped: {skipped_count})")

    return processed_count, skipped_count


def main():
    parser = argparse.ArgumentParser(description='Append final_boxed_answer for multi-agent tasks without one')
    parser.add_argument(
        '--log_dir',
        type=str,
        default='/mnt/project_rlinf/zhangruize/project/MAS/MiroThinker/logs/widesearch/2026-01-22-17-13-30_qwen_qwen-3_multi_agent_mirothinker_v1.0_8b_nulltasks_4runs',
        help='Path to the MiroThinker multi-agent log directory'
    )
    parser.add_argument(
        '--llm_base_url',
        type=str,
        default='http://0.0.0.0:61002/v1',
        help='LLM API base URL'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='qwen-3',
        help='Model name to use'
    )
    parser.add_argument(
        '--thread_num',
        type=int,
        default=4,
        help='Number of threads for parallel processing'
    )
    parser.add_argument(
        '--runs',
        type=str,
        default='1,2,3,4',
        help='Comma-separated list of run numbers to process'
    )

    args = parser.parse_args()

    # Initialize OpenAI client with local endpoint
    client = OpenAI(api_key='EMPTY', base_url=args.llm_base_url)

    run_numbers = [int(x.strip()) for x in args.runs.split(',')]

    total_processed = 0
    total_skipped = 0

    for run_num in run_numbers:
        run_folder = os.path.join(args.log_dir, f'run_{run_num}_append_final_box_answer_ninth')
        output_folder = os.path.join(args.log_dir, f'run_{run_num}_append_final_box_answer_tenth')

        if not os.path.exists(run_folder):
            print(f"Warning: {run_folder} does not exist, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {run_folder}")
        print(f"Output to {output_folder}")
        print(f"{'='*60}")

        processed, skipped = process_run_folder(run_folder, output_folder, client, args.model, args.thread_num)
        total_processed += processed
        total_skipped += skipped

        print(f"Run {run_num}: Processed {processed}, Skipped {skipped}")

    print(f"\n{'='*60}")
    print(f"Total: Processed {total_processed}, Skipped {total_skipped}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

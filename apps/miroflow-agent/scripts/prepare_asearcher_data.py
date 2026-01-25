#!/usr/bin/env python3
"""
Prepare Asearcher test data for MiroThinker benchmarks.
Converts various dataset formats to standardized_data.jsonl format.
"""

import json
import os
from pathlib import Path

# Source data directory
SOURCE_DIR = Path("/mnt/project_rlinf/zhangruize/data/Asearcher-test-data")

# Dataset configurations: (folder_name, task_id_field, question_field, answer_field)
DATASETS = {
    "Bamboogle": ("question", "question", "answer"),  # No explicit ID, use index
    "HotpotQA_rand1000": ("_id", "question", "answer"),
    "2WikiMultihopQA_rand1000": ("_id", "question", "answer"),
    "Musique_rand1000": ("id", "question", "answer"),
    "NQ_rand1000": ("question", "question", "answer"),  # No explicit ID, use index
    "PopQA_rand1000": ("question", "question", "answer"),  # No explicit ID, use index
    "TriviaQA_rand1000": ("question", "question", "answer"),  # No explicit ID, use index
}


def convert_dataset(dataset_name: str, id_field: str, question_field: str, answer_field: str):
    """Convert a dataset to standardized format."""
    source_file = SOURCE_DIR / dataset_name / "test.jsonl"
    output_file = SOURCE_DIR / dataset_name / "standardized_data.jsonl"

    if not source_file.exists():
        print(f"Warning: {source_file} does not exist, skipping...")
        return

    print(f"Converting {dataset_name}...")
    converted_count = 0

    with open(source_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:
        for idx, line in enumerate(f_in):
            try:
                data = json.loads(line.strip())

                # Get task_id
                if id_field == "question":
                    # Use index as ID for datasets without explicit ID
                    task_id = f"{dataset_name}_{idx:04d}"
                else:
                    task_id = str(data.get(id_field, f"{dataset_name}_{idx:04d}"))

                # Get question and answer
                question = data.get(question_field, "")
                answer = data.get(answer_field, "")

                # Handle list-type answers (multiple acceptable answers)
                # Convert to "answer1, answer2" format for LLM judge
                if isinstance(answer, list):
                    if len(answer) == 1:
                        answer = answer[0]
                    else:
                        answer = ", ".join(str(a) for a in answer)

                # Create standardized record
                standardized = {
                    "task_id": task_id,
                    "task_question": question,
                    "ground_truth": answer,
                }

                f_out.write(json.dumps(standardized, ensure_ascii=False) + "\n")
                converted_count += 1

            except json.JSONDecodeError as e:
                print(f"  Warning: Failed to parse line {idx + 1}: {e}")
                continue

    print(f"  Converted {converted_count} records to {output_file}")


def main():
    print("=" * 60)
    print("Preparing Asearcher test data for MiroThinker benchmarks")
    print("=" * 60)

    for dataset_name, (id_field, question_field, answer_field) in DATASETS.items():
        convert_dataset(dataset_name, id_field, question_field, answer_field)

    print("\n" + "=" * 60)
    print("Done! All datasets have been converted.")
    print("=" * 60)


if __name__ == "__main__":
    main()

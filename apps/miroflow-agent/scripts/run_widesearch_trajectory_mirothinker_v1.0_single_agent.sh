#!/bin/bash

# Script to collect trajectories for widesearch benchmark
# Only saves rollout trajectories, no evaluation scoring

# Get script directory and change to miroflow-agent root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Parse environment variables, use defaults if not set
LLM_MODEL=${LLM_MODEL:-"qwen-3"}
BASE_URL=${BASE_URL:-"http://localhost:61002/v1"}

# Configuration parameters
NUM_RUNS=${NUM_RUNS:-4}                    # Number of parallel runs (each run processes all tasks)
NUM_TASKS=${NUM_TASKS:-null}                  # Number of tasks to process (use first N tasks from dataset, null for all)
MAX_CONCURRENT=${MAX_CONCURRENT:-10}        # Max concurrent tasks within each run (3-5 for 2xA100 80GB)
BENCHMARK_NAME="widesearch"
LLM_PROVIDER=${LLM_PROVIDER:-"qwen"}
AGENT_SET=${AGENT_SET:-"mirothinker_v1.0_keep5_widesearch"}
MAX_CONTEXT_LENGTH=${MAX_CONTEXT_LENGTH:-32768}
PASS_AT_K=${PASS_AT_K:-1}
TEMPERATURE=${TEMPERATURE:-1.0}
API_KEY=${API_KEY:-"xxx"}

# Proxy settings (same as run_test.sh)
export http_proxy=${http_proxy:-"http://127.0.0.1:1080"}
export https_proxy=${https_proxy:-"http://127.0.0.1:1080"}
export no_proxy=${no_proxy:-"localhost,127.0.0.1,0.0.0.0"}

# Set results directory with timestamp
TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
RESULTS_DIR="../../logs/${BENCHMARK_NAME}/${TIMESTAMP}_${LLM_PROVIDER}_${LLM_MODEL}_${AGENT_SET}_${NUM_TASKS}tasks_${NUM_RUNS}runs"

echo "=========================================="
echo "Widesearch Trajectory Collection"
echo "=========================================="
echo "LLM Model: $LLM_MODEL"
echo "LLM Provider: $LLM_PROVIDER"
echo "Agent Set: $AGENT_SET"
echo "Base URL: $BASE_URL"
echo "Number of runs: $NUM_RUNS"
echo "Number of tasks: $NUM_TASKS"
echo "Max concurrent per run: $MAX_CONCURRENT"
echo "Pass@K: $PASS_AT_K"
echo "Temperature: $TEMPERATURE"
echo "Results directory: $RESULTS_DIR"
echo "=========================================="

# Create results directory
mkdir -p "$RESULTS_DIR"

# Handle NUM_TASKS: convert to null for hydra if set to "null" or "all"
if [ "$NUM_TASKS" = "null" ] || [ "$NUM_TASKS" = "all" ]; then
    MAX_TASKS_ARG="benchmark.execution.max_tasks=null"
else
    MAX_TASKS_ARG="benchmark.execution.max_tasks=$NUM_TASKS"
fi

echo "Starting $NUM_RUNS run(s) of trajectory collection..."
echo "Results will be saved in: $RESULTS_DIR"

# Launch all parallel runs
for i in $(seq 1 $NUM_RUNS); do
    echo "=========================================="
    echo "Launching run $i/$NUM_RUNS"
    echo "Output log: $RESULTS_DIR/run_${i}_output.log"
    echo "=========================================="

    # Set specific identifier for this run
    RUN_ID="run_$i"

    # Run experiment (background execution)
    (
        uv run python benchmarks/common_benchmark.py \
            benchmark=$BENCHMARK_NAME \
            llm=qwen-3 \
            llm.provider=$LLM_PROVIDER \
            llm.model_name=$LLM_MODEL \
            llm.base_url=$BASE_URL \
            llm.async_client=true \
            llm.temperature=$TEMPERATURE \
            llm.max_context_length=$MAX_CONTEXT_LENGTH \
            llm.api_key=$API_KEY \
            $MAX_TASKS_ARG \
            benchmark.execution.max_concurrent=$MAX_CONCURRENT \
            benchmark.execution.pass_at_k=$PASS_AT_K \
            agent=$AGENT_SET \
            hydra.run.dir=${RESULTS_DIR}/$RUN_ID \
            2>&1 | tee "$RESULTS_DIR/${RUN_ID}_output.log"

        # Check if run was successful
        if [ $? -eq 0 ]; then
            echo "Run $i completed successfully"
            echo "Trajectories saved to ${RESULTS_DIR}/$RUN_ID/"
        else
            echo "Run $i failed!"
        fi
    ) &

    # Small delay between launches to avoid simultaneous requests
    sleep 2
done

echo "All $NUM_RUNS run(s) have been launched in parallel"
echo "Waiting for all runs to complete..."

# Wait for all background tasks to complete
wait

echo "=========================================="
echo "All $NUM_RUNS run(s) completed!"
echo "=========================================="
echo "Trajectory files are saved in: $RESULTS_DIR"
echo "Each task trajectory is saved as: task_<instance_id>_attempt-<N>_format-retry-<M>_<timestamp>.json"
echo ""
echo "To list all trajectory files:"
echo "  find $RESULTS_DIR -name 'task_*.json' | head -20"
echo "=========================================="

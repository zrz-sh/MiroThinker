#!/bin/bash

# Smoke test: Run tasks with average@2 (Multi-Agent Offline RAG version)
# Quick validation to ensure the multi-agent offline RAG pipeline works correctly

# Get script directory and change to miroflow-agent root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Parse environment variables, use defaults if not set
LLM_MODEL=${LLM_MODEL:-"qwen-3"}
BASE_URL=${BASE_URL:-"http://localhost:61002/v1"}
LLM_PROVIDER=${LLM_PROVIDER:-"qwen"}
AGENT_SET=${AGENT_SET:-"multi_agent_mirothinker_v1.0_8b_offline"}
MAX_CONTEXT_LENGTH=${MAX_CONTEXT_LENGTH:-32768}
TEMPERATURE=${TEMPERATURE:-1.0}
API_KEY=${API_KEY:-"xxx"}

# Offline RAG server configuration
RAG_SERVER_ADDR=${RAG_SERVER_ADDR:-"127.0.0.1:8000"}
export RAG_SERVER_ADDR

# Smoke test settings
NUM_TASKS=2                    # Number of tasks to test
MAX_CONCURRENT=2               # Max concurrent tasks per run
NUM_RUNS=2                     # Number of runs for average@2
BENCHMARK_NAME="nq"

# Proxy settings (ensure local RAG server is not proxied)
export http_proxy=${http_proxy:-"http://127.0.0.1:1080"}
export https_proxy=${https_proxy:-"http://127.0.0.1:1080"}
export no_proxy=${no_proxy:-"localhost,127.0.0.1,0.0.0.0,172.27.236.118"}

# Set results directory
TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
RESULTS_DIR="../../logs/smoke_test_multiagent_offline/${TIMESTAMP}_${BENCHMARK_NAME}_${NUM_TASKS}tasks_average_at_${NUM_RUNS}"

echo "=========================================="
echo "SMOKE TEST (Multi-Agent Offline RAG) - average@${NUM_RUNS}"
echo "=========================================="
echo "Benchmark: $BENCHMARK_NAME"
echo "Tasks: $NUM_TASKS"
echo "Runs: $NUM_RUNS (for average@${NUM_RUNS})"
echo "LLM Model: $LLM_MODEL"
echo "LLM Provider: $LLM_PROVIDER"
echo "Agent Set: $AGENT_SET"
echo "Base URL: $BASE_URL"
echo "RAG Server: $RAG_SERVER_ADDR"
echo "Results: $RESULTS_DIR"
echo "=========================================="

# Step 0: Check if RAG server is running
echo ""
echo "Step 0: Checking RAG server availability..."
if curl -s --max-time 5 "http://${RAG_SERVER_ADDR}/retrieve" \
    -X POST \
    -H "Content-Type: application/json" \
    -d '{"queries":["test"],"topk":1}' > /dev/null 2>&1; then
    echo "✓ RAG server is running at http://${RAG_SERVER_ADDR}"
else
    echo "✗ ERROR: RAG server is not responding at http://${RAG_SERVER_ADDR}"
    echo ""
    echo "Please start the RAG server first:"
    echo "  cd /mnt/project_rlinf/xzxuan/RLinf"
    echo "  bash examples/search_engine/qdrant_best_async_ip.sh"
    echo ""
    exit 1
fi

# Create results directory
mkdir -p "$RESULTS_DIR"

# Step 1: Run multiple independent runs
echo ""
echo "Step 1: Running $NUM_RUNS independent runs..."
echo ""

for i in $(seq 1 $NUM_RUNS); do
    RUN_DIR="${RESULTS_DIR}/run_${i}"
    mkdir -p "$RUN_DIR"

    echo "=========================================="
    echo "Starting Run $i/$NUM_RUNS"
    echo "=========================================="

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
        benchmark.execution.max_tasks=$NUM_TASKS \
        benchmark.execution.max_concurrent=$MAX_CONCURRENT \
        benchmark.execution.pass_at_k=1 \
        agent=$AGENT_SET \
        hydra.run.dir=${RUN_DIR} \
        2>&1 | tee "${RUN_DIR}/output.log"

    echo "Run $i completed."
    echo ""
done

# Step 2: Calculate average
echo ""
echo "=========================================="
echo "Step 2: Calculating average@${NUM_RUNS}"
echo "=========================================="

total_accuracy=0
count=0

for i in $(seq 1 $NUM_RUNS); do
    ACCURACY_FILE="${RESULTS_DIR}/run_${i}/benchmark_results_pass_at_1_accuracy.txt"

    if [ -f "$ACCURACY_FILE" ]; then
        ACCURACY=$(cat "$ACCURACY_FILE")
        echo "Run $i: $ACCURACY"

        # Extract numeric value
        NUMERIC_ACCURACY=$(echo "$ACCURACY" | sed 's/%//')
        total_accuracy=$(awk "BEGIN {print $total_accuracy + $NUMERIC_ACCURACY}")
        count=$((count + 1))
    else
        echo "Run $i: Results not found"
    fi
done

echo ""
echo "=========================================="
echo "SMOKE TEST RESULTS (Multi-Agent Offline RAG)"
echo "=========================================="

if [ $count -gt 0 ]; then
    average_accuracy=$(awk "BEGIN {printf \"%.2f\", $total_accuracy / $count}")
    echo "Average@${NUM_RUNS}: ${average_accuracy}%"
    echo "${average_accuracy}%" > "${RESULTS_DIR}/average_at_${NUM_RUNS}.txt"
else
    echo "No valid results found"
fi

echo ""
echo "Results dir: ${RESULTS_DIR}"
echo "=========================================="

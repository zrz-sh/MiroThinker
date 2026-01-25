#!/bin/bash

# Script to run all 7 Asearcher benchmarks with average@4 evaluation (Offline RAG version)
# average@4 = 4 independent runs, then average the accuracy
# Uses offline RAG server instead of online search
# Uses mirothinker_v1.0_keep5_offline agent setting
# Evaluates using LLM as a judge

# Get script directory and change to miroflow-agent root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Parse environment variables, use defaults if not set
LLM_MODEL=${LLM_MODEL:-"qwen-3"}
BASE_URL=${BASE_URL:-"http://localhost:61002/v1"}
LLM_PROVIDER=${LLM_PROVIDER:-"qwen"}
AGENT_SET=${AGENT_SET:-"mirothinker_v1.0_keep5_offline"}
MAX_CONTEXT_LENGTH=${MAX_CONTEXT_LENGTH:-32768}
TEMPERATURE=${TEMPERATURE:-1.0}
API_KEY=${API_KEY:-"xxx"}

# Offline RAG server configuration
RAG_SERVER_ADDR=${RAG_SERVER_ADDR:-"127.0.0.1:8000"}
export RAG_SERVER_ADDR

# Benchmark execution settings
NUM_TASKS=${NUM_TASKS:-null}          # Number of tasks to process (null for all)
MAX_CONCURRENT=${MAX_CONCURRENT:-500}    # Max concurrent tasks per run
NUM_RUNS=${NUM_RUNS:-4}                # Number of runs for average@4

# Proxy settings (ensure local RAG server is not proxied)
export http_proxy=${http_proxy:-"http://127.0.0.1:1080"}
export https_proxy=${https_proxy:-"http://127.0.0.1:1080"}
export no_proxy=${no_proxy:-"localhost,127.0.0.1,0.0.0.0,172.27.236.118"}

# Set results directory with timestamp
TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
BASE_RESULTS_DIR="../../logs/asearcher_benchmarks_offline/${TIMESTAMP}_${LLM_PROVIDER}_${LLM_MODEL}_${AGENT_SET}_average_at_${NUM_RUNS}"

# List of benchmarks to run
BENCHMARKS=(
    # "bamboogle"
    # "hotpotqa"
    # "2wikimultihopqa"
    # "musique"
    "nq"
    "popqa"
    "triviaqa"
)

echo "=========================================="
echo "Asearcher Benchmarks (Offline RAG) - average@${NUM_RUNS} Evaluation"
echo "=========================================="
echo "LLM Model: $LLM_MODEL"
echo "LLM Provider: $LLM_PROVIDER"
echo "Agent Set: $AGENT_SET"
echo "Base URL: $BASE_URL"
echo "RAG Server: $RAG_SERVER_ADDR"
echo "Number of tasks: $NUM_TASKS"
echo "Max concurrent per run: $MAX_CONCURRENT"
echo "Number of runs: $NUM_RUNS"
echo "Results base directory: $BASE_RESULTS_DIR"
echo "Benchmarks: ${BENCHMARKS[*]}"
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
mkdir -p "$BASE_RESULTS_DIR"

# Handle NUM_TASKS: convert to null for hydra if set to "null" or "all"
if [ "$NUM_TASKS" = "null" ] || [ "$NUM_TASKS" = "all" ]; then
    MAX_TASKS_ARG="benchmark.execution.max_tasks=null"
else
    MAX_TASKS_ARG="benchmark.execution.max_tasks=$NUM_TASKS"
fi

# Step 1: Run each benchmark with multiple runs
echo ""
echo "Step 1: Running benchmarks with offline RAG..."
echo ""

for BENCHMARK in "${BENCHMARKS[@]}"; do
    echo "=========================================="
    echo "Running benchmark: $BENCHMARK (Offline RAG)"
    echo "=========================================="

    BENCHMARK_DIR="${BASE_RESULTS_DIR}/${BENCHMARK}"
    mkdir -p "$BENCHMARK_DIR"

    # Run multiple independent runs for this benchmark
    for run_idx in $(seq 1 $NUM_RUNS); do
        RUN_DIR="${BENCHMARK_DIR}/run_${run_idx}"
        mkdir -p "$RUN_DIR"

        echo ""
        echo "--- $BENCHMARK: Run $run_idx/$NUM_RUNS ---"

        uv run python benchmarks/common_benchmark.py \
            benchmark=$BENCHMARK \
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
            benchmark.execution.pass_at_k=1 \
            agent=$AGENT_SET \
            hydra.run.dir=${RUN_DIR} \
            2>&1 | tee "${RUN_DIR}/output.log"

        echo "$BENCHMARK Run $run_idx completed."
    done

    # Calculate average for this benchmark
    echo ""
    echo "Calculating average@${NUM_RUNS} for $BENCHMARK..."

    total_accuracy=0
    count=0

    for run_idx in $(seq 1 $NUM_RUNS); do
        ACCURACY_FILE="${BENCHMARK_DIR}/run_${run_idx}/benchmark_results_pass_at_1_accuracy.txt"

        if [ -f "$ACCURACY_FILE" ]; then
            ACCURACY=$(cat "$ACCURACY_FILE")
            NUMERIC_ACCURACY=$(echo "$ACCURACY" | sed 's/%//')
            total_accuracy=$(awk "BEGIN {print $total_accuracy + $NUMERIC_ACCURACY}")
            count=$((count + 1))
            echo "  Run $run_idx: $ACCURACY"
        fi
    done

    if [ $count -gt 0 ]; then
        avg=$(awk "BEGIN {printf \"%.2f\", $total_accuracy / $count}")
        echo "${avg}%" > "${BENCHMARK_DIR}/average_at_${NUM_RUNS}.txt"
        echo "  $BENCHMARK average@${NUM_RUNS}: ${avg}%"
    fi

    echo ""
done

# Step 2: Summarize all results
echo "=========================================="
echo "Step 2: Summarizing results"
echo "=========================================="

SUMMARY_FILE="${BASE_RESULTS_DIR}/summary.txt"

echo "Asearcher Benchmarks Summary (Offline RAG)" > "$SUMMARY_FILE"
echo "===========================================" >> "$SUMMARY_FILE"
echo "Date: $(date)" >> "$SUMMARY_FILE"
echo "LLM Model: $LLM_MODEL" >> "$SUMMARY_FILE"
echo "Agent Set: $AGENT_SET" >> "$SUMMARY_FILE"
echo "RAG Server: $RAG_SERVER_ADDR" >> "$SUMMARY_FILE"
echo "Runs per benchmark: $NUM_RUNS" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Results (average@${NUM_RUNS}):" >> "$SUMMARY_FILE"
echo "------------------------------" >> "$SUMMARY_FILE"

total_avg=0
benchmark_count=0

for BENCHMARK in "${BENCHMARKS[@]}"; do
    AVG_FILE="${BASE_RESULTS_DIR}/${BENCHMARK}/average_at_${NUM_RUNS}.txt"

    if [ -f "$AVG_FILE" ]; then
        AVG=$(cat "$AVG_FILE")
        echo "$BENCHMARK: $AVG" >> "$SUMMARY_FILE"
        echo "$BENCHMARK: $AVG"

        NUMERIC_AVG=$(echo "$AVG" | sed 's/%//')
        total_avg=$(awk "BEGIN {print $total_avg + $NUMERIC_AVG}")
        benchmark_count=$((benchmark_count + 1))
    else
        echo "$BENCHMARK: Results not found" >> "$SUMMARY_FILE"
        echo "$BENCHMARK: Results not found"
    fi
done

# Calculate overall average
if [ $benchmark_count -gt 0 ]; then
    overall_avg=$(awk "BEGIN {printf \"%.2f\", $total_avg / $benchmark_count}")
    echo "" >> "$SUMMARY_FILE"
    echo "Overall average@${NUM_RUNS} across all benchmarks: ${overall_avg}%" >> "$SUMMARY_FILE"
    echo ""
    echo "Overall average@${NUM_RUNS} across all benchmarks: ${overall_avg}%"
fi

echo "" >> "$SUMMARY_FILE"
echo "Full results saved in: $BASE_RESULTS_DIR" >> "$SUMMARY_FILE"

echo ""
echo "=========================================="
echo "All benchmarks completed!"
echo "=========================================="
echo "Summary saved to: $SUMMARY_FILE"
echo "Full results in: $BASE_RESULTS_DIR"
echo "=========================================="

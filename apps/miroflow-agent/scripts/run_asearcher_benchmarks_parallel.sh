#!/bin/bash

# Script to run all 7 Asearcher benchmarks in PARALLEL with average@4 evaluation
# average@4 = 4 independent runs, then average the accuracy
# Uses mirothinker_v1.5_keep5_max200_widesearch agent setting
# Evaluates using LLM as a judge

# Get script directory and change to miroflow-agent root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Parse environment variables, use defaults if not set
LLM_MODEL=${LLM_MODEL:-"qwen-3"}
BASE_URL=${BASE_URL:-"http://localhost:61002/v1"}
LLM_PROVIDER=${LLM_PROVIDER:-"qwen"}
AGENT_SET=${AGENT_SET:-"mirothinker_v1.5_keep5_max200_widesearch"}
MAX_CONTEXT_LENGTH=${MAX_CONTEXT_LENGTH:-262144}
TEMPERATURE=${TEMPERATURE:-1.0}
API_KEY=${API_KEY:-"xxx"}

# Benchmark execution settings
NUM_TASKS=${NUM_TASKS:-null}          # Number of tasks to process (null for all)
MAX_CONCURRENT=${MAX_CONCURRENT:-3}    # Max concurrent tasks per run
NUM_RUNS=${NUM_RUNS:-4}                # Number of runs for average@4
NUM_PARALLEL_BENCHMARKS=${NUM_PARALLEL_BENCHMARKS:-2}  # Number of benchmarks to run in parallel

# Proxy settings
export http_proxy=${http_proxy:-"http://127.0.0.1:1080"}
export https_proxy=${https_proxy:-"http://127.0.0.1:1080"}
export no_proxy=${no_proxy:-"localhost,127.0.0.1,0.0.0.0,172.27.236.118"}

# Set results directory with timestamp
TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
BASE_RESULTS_DIR="../../logs/asearcher_benchmarks/${TIMESTAMP}_${LLM_PROVIDER}_${LLM_MODEL}_${AGENT_SET}_average_at_${NUM_RUNS}"

# List of benchmarks to run
BENCHMARKS=(
    "bamboogle"
    "hotpotqa"
    "2wikimultihopqa"
    "musique"
    "nq"
    "popqa"
    "triviaqa"
)

echo "=========================================="
echo "Asearcher Benchmarks - PARALLEL average@${NUM_RUNS} Evaluation"
echo "=========================================="
echo "LLM Model: $LLM_MODEL"
echo "LLM Provider: $LLM_PROVIDER"
echo "Agent Set: $AGENT_SET"
echo "Base URL: $BASE_URL"
echo "Number of tasks: $NUM_TASKS"
echo "Max concurrent per run: $MAX_CONCURRENT"
echo "Number of runs: $NUM_RUNS"
echo "Parallel benchmarks: $NUM_PARALLEL_BENCHMARKS"
echo "Results base directory: $BASE_RESULTS_DIR"
echo "Benchmarks: ${BENCHMARKS[*]}"
echo "=========================================="

# Step 1: Prepare data (convert test.jsonl to standardized_data.jsonl)
echo ""
echo "Step 1: Preparing data..."
python3 scripts/prepare_asearcher_data.py

if [ $? -ne 0 ]; then
    echo "Error: Data preparation failed!"
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

# Step 2: Run benchmarks in parallel (each benchmark runs NUM_RUNS times sequentially)
echo ""
echo "Step 2: Running benchmarks in parallel (max $NUM_PARALLEL_BENCHMARKS at a time)..."
echo ""

run_benchmark_with_runs() {
    local BENCHMARK=$1
    local BENCHMARK_DIR="${BASE_RESULTS_DIR}/${BENCHMARK}"
    mkdir -p "$BENCHMARK_DIR"

    echo "[$(date +%H:%M:%S)] Starting benchmark: $BENCHMARK ($NUM_RUNS runs)"

    # Run multiple independent runs for this benchmark
    for run_idx in $(seq 1 $NUM_RUNS); do
        local RUN_DIR="${BENCHMARK_DIR}/run_${run_idx}"
        mkdir -p "$RUN_DIR"

        echo "[$(date +%H:%M:%S)] $BENCHMARK: Run $run_idx/$NUM_RUNS"

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
            > "${RUN_DIR}/output.log" 2>&1
    done

    # Calculate average for this benchmark
    local total_accuracy=0
    local count=0

    for run_idx in $(seq 1 $NUM_RUNS); do
        local ACCURACY_FILE="${BENCHMARK_DIR}/run_${run_idx}/benchmark_results_pass_at_1_accuracy.txt"

        if [ -f "$ACCURACY_FILE" ]; then
            local ACCURACY=$(cat "$ACCURACY_FILE")
            local NUMERIC_ACCURACY=$(echo "$ACCURACY" | sed 's/%//')
            total_accuracy=$(awk "BEGIN {print $total_accuracy + $NUMERIC_ACCURACY}")
            count=$((count + 1))
        fi
    done

    if [ $count -gt 0 ]; then
        local avg=$(awk "BEGIN {printf \"%.2f\", $total_accuracy / $count}")
        echo "${avg}%" > "${BENCHMARK_DIR}/average_at_${NUM_RUNS}.txt"
        echo "[$(date +%H:%M:%S)] $BENCHMARK average@${NUM_RUNS}: ${avg}%"
    else
        echo "[$(date +%H:%M:%S)] Warning: $BENCHMARK has no valid results"
    fi
}

export -f run_benchmark_with_runs
export BASE_RESULTS_DIR LLM_PROVIDER LLM_MODEL BASE_URL TEMPERATURE MAX_CONTEXT_LENGTH API_KEY MAX_TASKS_ARG MAX_CONCURRENT NUM_RUNS AGENT_SET

# Run benchmarks with GNU parallel if available, otherwise use a simple loop
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel for execution..."
    printf '%s\n' "${BENCHMARKS[@]}" | parallel -j $NUM_PARALLEL_BENCHMARKS run_benchmark_with_runs {}
else
    echo "GNU parallel not found, using background jobs..."
    running=0
    for BENCHMARK in "${BENCHMARKS[@]}"; do
        run_benchmark_with_runs "$BENCHMARK" &
        running=$((running + 1))

        if [ $running -ge $NUM_PARALLEL_BENCHMARKS ]; then
            wait -n
            running=$((running - 1))
        fi
    done
    wait
fi

# Step 3: Summarize all results
echo ""
echo "=========================================="
echo "Step 3: Summarizing results"
echo "=========================================="

SUMMARY_FILE="${BASE_RESULTS_DIR}/summary.txt"

echo "Asearcher Benchmarks Summary" > "$SUMMARY_FILE"
echo "============================" >> "$SUMMARY_FILE"
echo "Date: $(date)" >> "$SUMMARY_FILE"
echo "LLM Model: $LLM_MODEL" >> "$SUMMARY_FILE"
echo "Agent Set: $AGENT_SET" >> "$SUMMARY_FILE"
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

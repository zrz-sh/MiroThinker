unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

NUM_GPUS=4
PORT=61002

# Downloading agent from HF (v1.5 recommended)
AGENT_PATH=/mnt/project_rlinf/xzxuan/model/miro-v1.0-8b

# Or use v1.0
# AGENT_PATH=miromind-ai/MiroThinker-v1.0-30B

python -m sglang.launch_server \
    --model-path $AGENT_PATH \
    --tp 1 \
    --dp $NUM_GPUS \
    --host 0.0.0.0 \
    --port $PORT \
    --trust-remote-code
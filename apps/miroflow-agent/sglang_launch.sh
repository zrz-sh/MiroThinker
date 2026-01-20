unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

NUM_GPUS=2
PORT=61002

# Downloading agent from HF (v1.5 recommended)
AGENT_PATH=/mnt/project_rlinf/zhangruize/ckpt/MiroThinker-v1.5-30B

# Or use v1.0
# AGENT_PATH=miromind-ai/MiroThinker-v1.0-30B

python -m sglang.launch_server \
    --model-path $AGENT_PATH \
    --tp $NUM_GPUS \
    --dp 1 \
    --host 0.0.0.0 \
    --port $PORT \
    --trust-remote-code
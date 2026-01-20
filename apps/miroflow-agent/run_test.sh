export http_proxy=http://127.0.0.1:1080
export https_proxy=http://127.0.0.1:1080
export no_proxy="localhost,127.0.0.1,0.0.0.0"

uv run python main.py llm=qwen-3 agent=mirothinker_v1.5_keep5_max400_widesearch llm.base_url=http://localhost:61002/v1
## TensorRT-Demo v0.13.0

### Step 1. Docker setup

```bash
export MODEL_PATH=/data
git clone https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct $MODEL_PATH/Meta-Llama-3-8B-Instruct
git clone https://github.com/MasterJH5574/tensorrt-demo.git

docker pull nvcr.io/nvidia/tritonserver:24.09-trtllm-python-py3
docker run --shm-size 32g -v $MODEL_PATH:/models -v $PWD/tensorrt-demo:/tensorrt-demo --workdir / -p 8123:8123 --gpus all -it $(docker image ls | grep 24.09 | awk '{print $3}') /bin/bash
```

### Step 2. Convert checkpoint and build engine

```bash
# Convert checkpoint
wget https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/v0.13.0/examples/llama/convert_checkpoint.py
python3 convert_checkpoint.py --model_dir /models/Meta-Llama-3-8B-Instruct --dtype float16 --tp_size 1 --output_dir llama3-8b
# Build engine
trtllm-build --checkpoint_dir=llama3-8b --output_dir=llama3-8b-engine --gpt_attention_plugin=float16 --gemm_plugin=float16 --remove_input_padding=enable --paged_kv_cache=enable --use_paged_context_fmha enable --multiple_profiles enable --max_batch_size=2048 --max_input_len=16384 --max_num_tokens=16384
```

### Step 3. Launch server

```bash
cp llama3-8b-engine/* /tensorrt-demo/triton_model_repo/tensorrt_llm/1/
wget https://raw.githubusercontent.com/triton-inference-server/tensorrtllm_backend/v0.13.0/scripts/launch_triton_server.py
python3 launch_triton_server.py --world_size=1 --model_repo=/tensorrt-demo/triton_model_repo --http_port 8123
```

### Step 4. Run benchmark

```bash
# Create conda env (if run outside the docker container)
conda create -n mlc_env python=3.11 -y
conda activate mlc_env

# If run outside the docker container
export MODEL_PATH=/data/Meta-Llama-3-8B-Instruct
# If run inside the docker container
export MODEL_PATH=/models/Meta-Llama-3-8B-Instruct

# Install MLC for `mlc_llm.bench`
python3 -m pip install --pre -U -f https://mlc.ai/wheels mlc-llm-cu123 mlc-ai-cu123

# Run benchmark
export SERVER_ADDR=127.0.0.1
export SERVER_PORT=8123
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
export SHAREGPT_PATH=$PWD/ShareGPT_V3_unfiltered_cleaned_split.json
python3 -m mlc_llm.bench --api-endpoint tensorrt-llm --dataset sharegpt --dataset-path $SHAREGPT_PATH --tokenizer $MODEL_PATH --num-request 500 --num-gpus 1 --num-concurrent-requests 1,4,8,10,16,20,30,64 --temperature 0.6 --top-p 0.9 --ignore-eos --apply-chat-template --host $SERVER_ADDR --port $SERVER_PORT -o mlc_benchmark.csv
```

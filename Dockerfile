FROM python:3.10-slim-bookworm

ENV TRANSFORMERS_CACHE=/home/aicrowd/hf_cache
ENV HF_HOME=/home/aicrowd/hf_cache

ENV HUGGINGFACE_TOKEN=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

RUN mkdir -p /root/.huggingface && \
    echo "{\"token\":\"$HUGGINGFACE_TOKEN\"}" > /root/.huggingface/token

# Python setup
RUN pip install --no-cache-dir -U pip==21.0.1
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN pip install --no-cache-dir vllm==0.7.3

# Code and working directory
WORKDIR /home/aicrowd
COPY agents/ ./agents/
COPY docs/ ./docs/
COPY offload_dir ./offload_dir
COPY aicrowd.json .
COPY crag_batch_iterator.py .
COPY crag_image_loader.py .
COPY crag_web_result_fetcher.py .
COPY local_evaluation.py .
COPY utils.py .

# Run main file
CMD ["python3", "local_evaluation.py"]
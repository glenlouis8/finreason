# vLLM serving image for the AWQ-quantized FinReason model.
# Built on the official vLLM image (CUDA + vLLM preinstalled).
FROM vllm/vllm-openai:latest

# model is pulled from the Hub at runtime; override with -e MODEL=...
ENV MODEL="glen-louis/finreason-qwen2.5-7b-awq"

EXPOSE 8000

# vllm/vllm-openai's entrypoint IS `vllm serve`; we pass args via CMD
CMD ["--model", "glen-louis/finreason-qwen2.5-7b-awq", \
     "--quantization", "awq", \
     "--max-model-len", "4096", \
     "--gpu-memory-utilization", "0.90", \
     "--port", "8000"]

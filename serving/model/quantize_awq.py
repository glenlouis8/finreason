"""
Merge the FinReason LoRA adapter into the base model, then AWQ-quantize
the merged model to 4-bit, then push to the Hub.

Run on a GPU box (RunPod, ~24-48GB VRAM recommended):
    pip install "autoawq" transformers peft accelerate huggingface_hub
    huggingface-cli login
    python quantize_awq.py

Why merge first: the published repo is a LoRA *adapter*, not a full model.
AWQ needs one standalone model, so we bake the adapter into the base
(`merge_and_unload`) before quantizing.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from awq import AutoAWQForCausalLM

BASE      = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER   = "glen-louis/finreason-qwen2.5-7b-dpo"
MERGED    = "finreason-merged"                       # local scratch dir
AWQ_OUT   = "finreason-qwen2.5-7b-awq"               # local out dir
HF_REPO   = "glen-louis/finreason-qwen2.5-7b-awq"    # destination on Hub

# ── 1. MERGE adapter into base -> one full fp16 model ──
print(">> loading base + adapter, merging...")
base = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.float16, device_map="auto"
)
model = PeftModel.from_pretrained(base, ADAPTER)
model = model.merge_and_unload()                     # bake adapter in permanently
model.save_pretrained(MERGED, safe_serialization=True)
AutoTokenizer.from_pretrained(BASE).save_pretrained(MERGED)
del base, model
torch.cuda.empty_cache()
print(">> merged model saved to", MERGED)

# ── 2. AWQ quantize the merged model (16-bit -> 4-bit) ──
print(">> AWQ quantizing...")
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
awq_model = AutoAWQForCausalLM.from_pretrained(MERGED, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MERGED)
awq_model.quantize(tokenizer, quant_config=quant_config)   # uses default calibration set
awq_model.save_quantized(AWQ_OUT)
tokenizer.save_pretrained(AWQ_OUT)
print(">> AWQ model saved to", AWQ_OUT)

# ── 3. PUSH to Hub ──
print(">> pushing to", HF_REPO)
awq_model.model.push_to_hub(HF_REPO)
tokenizer.push_to_hub(HF_REPO)
print(">> done. Serve with: vllm serve", HF_REPO, "--quantization awq")

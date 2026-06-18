import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel


def get_bnb_config(cfg: dict) -> BitsAndBytesConfig:
    q = cfg["quantization"]
    return BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
    )


def load_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(model_name: str, bnb_config: BitsAndBytesConfig) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    return model


def apply_lora(model: AutoModelForCausalLM, cfg: dict) -> AutoModelForCausalLM:
    q = cfg["qlora"]
    lora_config = LoraConfig(
        r=q["r"],
        lora_alpha=q["lora_alpha"],
        lora_dropout=q["lora_dropout"],
        target_modules=q["target_modules"],
        bias=q["bias"],
        task_type=q["task_type"],
    )
    return get_peft_model(model, lora_config)


def load_model_for_training(cfg: dict):
    model_name = cfg["model"]["name"]
    bnb_config = get_bnb_config(cfg)
    tokenizer = load_tokenizer(model_name)
    model = load_base_model(model_name, bnb_config)
    model = apply_lora(model, cfg)
    model.print_trainable_parameters()
    return model, tokenizer


def load_model_for_inference(model_name: str, checkpoint_path: str, cfg: dict):
    bnb_config = get_bnb_config(cfg)
    tokenizer = load_tokenizer(model_name)
    base_model = load_base_model(model_name, bnb_config)
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()
    return model, tokenizer


def load_base_only(model_name: str, cfg: dict):
    bnb_config = get_bnb_config(cfg)
    tokenizer = load_tokenizer(model_name)
    model = load_base_model(model_name, bnb_config)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]


def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 0.8,
    top_p: float = 0.95,
) -> list[str]:
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
    input_lengths = inputs["input_ids"].shape[1]
    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample, pad_token_id=tokenizer.eos_token_id)
    if do_sample:
        # Sampling needed for DPO pair mining: runs must diverge to yield correct+wrong chains.
        gen_kwargs.update(temperature=temperature, top_p=top_p)
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    tokenizer.padding_side = "right"
    return [tokenizer.decode(out[input_lengths:], skip_special_tokens=True) for out in outputs]

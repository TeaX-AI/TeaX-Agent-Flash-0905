#!/usr/bin/env python3
import os
import sys
import json
import argparse
import glob
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from datasets import load_dataset, concatenate_datasets

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["WANDB_DISABLED"] = "true"

# ==================== 系统提示词 ====================
SYSTEM_PROMPT = """You are TeaX Agent Flash, a capable AI assistant engineered for developer productivity. Your expertise spans code generation, debugging, tool orchestration, and technical Q&A.

You operate with precision and clarity. You do not guess. You do not improvise when uncertain. You ask.

Core Operating Principles

1. Reasoning Protocol
Before responding, you may reason internally. When you do:
- Open your reasoning with <think> and close with </think>.
- The first sentence inside <think> MUST begin with "we need to ..." — this anchors your reasoning in executable, task-oriented thinking.
- Keep reasoning concise and action-focused. One thought per sentence.
- Never leak <think> content into your final response.

2. Answer Boundary (CRITICAL for Ollama/Qwen)
- Your response MUST end with <|im_end|>.
- Do NOT generate any content after <|im_end|>.
- Do NOT role-play as the user, the system, or any other entity.
- Do NOT continue the conversation beyond answering the current user query.
If you find yourself generating dialogue for the user, STOP. That is a boundary violation.

3. Uncertainty Handling
- If information is missing, ambiguous, or unclear: call ask-user.
- Do NOT hallucinate. Do NOT guess. Do NOT assume.
- If you don't know, say you don't know — then ask.

4. Tool Use (DSH / DeepSeek Harness Style)
Available tools: bash, web_search, web_fetch, ask-user, read, write, edit.
Tool call format:
<tool_call>{"name": "<tool_name>", "arguments": {...}}</tool_call>
Tool discipline:
- ask-user: use when ANY uncertainty exists about user intent, file paths, or desired output.
- bash: prefer for file operations, process management, and system queries.
- read before write or edit — understand the current state first.
- web_search for information beyond your knowledge cutoff.

5. Output Format
- Code: use markdown fenced blocks with language annotation.
- Data: prefer JSON for structured output.
- Explanations: concise, technical, and in English unless the user specifies otherwise.
- Tool results: present clearly, with citations when applicable.

6. Mode & Effort
- Default to direct, non-thinking mode for simple Q&A.
- Use light reasoning for routine coding and agent workflows.
- Escalate to deeper reasoning only when the problem genuinely demands it — not for every query.

Hard Constraints (Ollama/Qwen Compatibility)
- End every response with <|im_end|> — prevents Qwen from "continuing" beyond your answer.
- Never generate user-side dialogue — stops the model from role-playing both sides.
- Never open a new assistant turn unprompted — one response per user query, no cascading.
- Keep thinking internal (<think> ... </think>) — thinking mode must not leak into visible output.

Meta-Instruction
You are an agent, not a chatbot that fills empty space. Your job is to solve the user's problem — then stop. Clarity over length. Action over explanation. Ask over guess. When in doubt, ask. When done, stop."""

parser = argparse.ArgumentParser()
parser.add_argument('--stage', type=int, required=True, choices=[1, 2, 3, 4, 5])
parser.add_argument('--base_model', type=str, default=None)
parser.add_argument('--output_model', type=str, default=None)
parser.add_argument('--max_steps', type=int, default=None)
args = parser.parse_args()

STAGE_CONFIG = {
    1: {
        "name": "MSAgent",
        "output_model": "checkpoints/TeaX-Agent-Flash-MSAgent",
        "dataset_paths": [".cache/msagent/train.jsonl"],
        "dataset_format": "json",
        "samples": 598185,
        "base_model": ".cache/model",
    },
    2: {
        "name": "ShareGPT",
        "output_model": "checkpoints/TeaX-Agent-Flash-MSAgent-ShareGPT",
        "dataset_paths": [".cache/sharegpt/sharegpt_jsonl/*.jsonl"],
        "dataset_format": "json",
        "samples": 230000,
        "base_model": "previous_model",
    },
    3: {
        "name": "DeepSeek-Hermes",
        "output_model": "checkpoints/TeaX-Agent-Flash-MSAgent-ShareGPT-DeepSeekHarmes",
        "dataset_paths": [".cache/deepseek-hermes/train.parquet"],
        "dataset_format": "parquet",
        "samples": 16431,
        "base_model": "previous_model",
    },
    4: {
        "name": "NextCoder",
        "output_model": "checkpoints/TeaX-Agent-Flash-MSAgent-ShareGPT-DeepSeekHarmes-NextCoder",
        "dataset_paths": [".cache/nextcoder/data.parquet"],
        "dataset_format": "parquet",
        "samples": 100000,
        "base_model": "previous_model",
    },
    5: {
        "name": "Merge",
        "output_model": "checkpoints/TeaX-Agent-Flash",
        "base_model": "checkpoints/TeaX-Agent-Flash-MSAgent-ShareGPT-DeepSeekHarmes-NextCoder",
        "is_merge": True,
    }
}

config = STAGE_CONFIG[args.stage]
base_model_path = args.base_model if args.base_model else config.get("base_model", ".cache/model")
output_model_path = args.output_model if args.output_model else config["output_model"]
max_steps = args.max_steps if args.max_steps else (config.get("samples", 0) // 4) if config.get("samples") else 10000

print(f"阶段 {args.stage}: {config['name']}")
print(f"基础模型: {base_model_path}")
print(f"输出模型: {output_model_path}")
print(f"最大步数: {max_steps}")

if args.stage == 5:
    print("合并模式：将 LoRA 适配器合并到基础模型，并分片保存")
    if base_model_path == "previous_model":
        base_model_path = "checkpoints/TeaX-Agent-Flash-MSAgent-ShareGPT-DeepSeekHarmes-NextCoder"
        if not Path(base_model_path).exists():
            print(f"错误: 找不到适配器目录 {base_model_path}")
            sys.exit(-1)

    print(f"加载基础模型: .cache/model")
    tokenizer = AutoTokenizer.from_pretrained(".cache/model", trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        ".cache/model",
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    print(f"加载 LoRA 适配器: {base_model_path}")
    model = PeftModel.from_pretrained(base_model, base_model_path)
    print("合并权重...")
    merged_model = model.merge_and_unload()
    print(f"保存合并模型到 {output_model_path}，并自动分片（每片 < 1.8GB）")
    os.makedirs(output_model_path, exist_ok=True)
    merged_model.save_pretrained(output_model_path, max_shard_size="1.8GB")
    tokenizer.save_pretrained(output_model_path)
    print("合并完成，分片保存成功！")
    sys.exit(0)

def fmt_msagent(sample, tokenizer):
    convs = sample.get("conversations", [])
    if not convs:
        return None
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for t in convs:
        role = t.get("from", "")
        content = t.get("value", "")
        if role == "system":
            continue
        if role == "user":
            messages.append({"role": "user", "content": content})
        elif role == "assistant":
            messages.append({"role": "assistant", "content": content})
        else:
            continue
    if len(messages) <= 1:
        return None
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

def fmt_sharegpt(sample, tokenizer):
    convs = sample.get("conversations", [])
    if not convs:
        return None
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for t in convs:
        role = t.get("from", "")
        content = t.get("value", "")
        if role == "human":
            messages.append({"role": "user", "content": content})
        elif role == "gpt":
            messages.append({"role": "assistant", "content": content})
        else:
            continue
    if len(messages) <= 1:
        return None
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

def fmt_deepseek_hermes(sample, tokenizer):
    convs = sample.get("conversations", [])
    if not convs:
        convs = sample.get("messages", [])
    if not convs:
        return None
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for t in convs:
        role = t.get("role", "")
        content = t.get("content", "")
        if role == "system":
            continue
        if role in ("user", "assistant"):
            messages.append({"role": role, "content": content})
        else:
            continue
    if len(messages) <= 1:
        return None
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

def fmt_nextcoder(sample, tokenizer):
    input_msgs = sample.get("input", [])
    response = sample.get("response", "")
    if not input_msgs or not response:
        return None
    user_content = None
    for m in input_msgs:
        if m.get("role") == "user":
            user_content = m.get("content")
            break
    if not user_content:
        return None
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": response}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

if base_model_path == "previous_model":
    prev_stage = args.stage - 1
    prev_output = STAGE_CONFIG[prev_stage]["output_model"]
    base_model_path = prev_output
    if not Path(base_model_path).exists():
        print(f"错误: 上一阶段输出 {base_model_path} 不存在")
        sys.exit(-1)

print(f"加载模型: {base_model_path}")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    dtype=torch.bfloat16,
    device_map="cpu",
    low_cpu_mem_usage=True,
    trust_remote_code=True
)
model.gradient_checkpointing_enable()

if not isinstance(model, PeftModel):
    print("应用新 LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
else:
    print("检测到已有 PEFT 模型，继续训练...")

model.print_trainable_parameters()

def get_fmt_fn(stage):
    if stage == 1:
        return fmt_msagent
    elif stage == 2:
        return fmt_sharegpt
    elif stage == 3:
        return fmt_deepseek_hermes
    elif stage == 4:
        return fmt_nextcoder
    else:
        raise ValueError("无效 stage")

fmt_fn = get_fmt_fn(args.stage)

all_datasets = []
for pattern in config["dataset_paths"]:
    files = glob.glob(pattern)
    if not files:
        print(f"警告: 未找到匹配 {pattern} 的文件")
        continue
    for f in files:
        if not Path(f).exists():
            continue
        ext = Path(f).suffix[1:]
        print(f"加载 {f}")
        try:
            if ext in ("jsonl", "json"):
                ds = load_dataset("json", data_files=f, split="train", streaming=True)
            elif ext == "parquet":
                ds = load_dataset("parquet", data_files=f, split="train", streaming=True)
            else:
                print(f"未知格式: {ext}，跳过")
                continue
            all_datasets.append(ds)
        except Exception as e:
            print(f"加载 {f} 失败: {e}")
            continue

if not all_datasets:
    print("错误: 没有加载到任何数据")
    sys.exit(-1)

if len(all_datasets) > 1:
    dataset = concatenate_datasets(all_datasets)
else:
    dataset = all_datasets[0]

print("数据集加载完成")

def tokenize_fn(ex):
    text = fmt_fn(ex, tokenizer)
    if not text:
        return None
    enc = tokenizer(text, truncation=True, max_length=2048, padding=False)
    return {
        "input_ids": enc["input_ids"],
        "labels": enc["input_ids"].copy(),
        "attention_mask": enc["attention_mask"]
    }

dataset = dataset.map(tokenize_fn, batched=False, remove_columns=dataset.column_names)
dataset = dataset.filter(lambda x: x["input_ids"] is not None and len(x["input_ids"]) > 0)

def custom_collator(batch):
    max_len = max(len(item["input_ids"]) for item in batch)
    padded_input_ids = []
    padded_labels = []
    padded_attention_mask = []
    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        padded_input_ids.append(item["input_ids"] + [tokenizer.pad_token_id] * pad_len)
        padded_labels.append(item["labels"] + [-100] * pad_len)
        padded_attention_mask.append(item["attention_mask"] + [0] * pad_len)
    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long),
        "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
    }

training_args = TrainingArguments(
    output_dir="./tmp_train",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=100,
    max_steps=max_steps,
    logging_steps=200,
    save_steps=500,
    save_total_limit=2,
    bf16=True,
    use_cpu=True,
    report_to="none",
    gradient_checkpointing=True,
    dataloader_num_workers=0,
    remove_unused_columns=False,
    eval_strategy="no",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=custom_collator,
)

print("开始训练...")
trainer.train()

print(f"保存模型到 {output_model_path}")
os.makedirs(output_model_path, exist_ok=True)
model.save_pretrained(output_model_path)
tokenizer.save_pretrained(output_model_path)
print("训练完成！")
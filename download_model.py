#!/usr/bin/env python3
"""
Download Qwen2.5-1.5B base model to .cache/model/
If the directory already exists and is non-empty, skip download.
"""
import os
import sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
CACHE_DIR = ".cache/model"

def download_model():
    cache_path = Path(CACHE_DIR)
    # 如果目录已存在且非空，跳过下载
    if cache_path.exists() and any(cache_path.iterdir()):
        print(f"✅ 模型已存在: {CACHE_DIR}")
        return True

    print(f"⏳ 正在从 HuggingFace 下载 {MODEL_NAME} ...")
    try:
        # 下载模型和分词器
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            cache_dir=CACHE_DIR,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=CACHE_DIR,
            trust_remote_code=True
        )
        # transformers 的 cache_dir 会自动保存，但我们需要确保文件在正确位置
        # 这里直接保存到 CACHE_DIR
        model.save_pretrained(CACHE_DIR)
        tokenizer.save_pretrained(CACHE_DIR)
        print(f"✅ 模型下载并保存到 {CACHE_DIR}")
        return True
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

if __name__ == "__main__":
    if download_model():
        sys.exit(0)
    else:
        sys.exit(-1)
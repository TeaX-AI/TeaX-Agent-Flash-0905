#!/usr/bin/env python3
import os
import sys
import json
import argparse
import shutil
from pathlib import Path

CHUNK_SIZE = int(1.8 * 1024 * 1024 * 1024)  # 1.8 GB

def split_file(file_path, output_dir, chunk_size=CHUNK_SIZE):
    file_path = Path(file_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_size = file_path.stat().st_size
    if file_size <= chunk_size:
        shutil.copy2(file_path, output_dir / file_path.name)
        return [file_path.name]

    parts = []
    with open(file_path, 'rb') as f:
        part_num = 1
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            part_name = f"{file_path.stem}.part{part_num:03d}{file_path.suffix}"
            part_path = output_dir / part_name
            with open(part_path, 'wb') as pf:
                pf.write(chunk)
            parts.append(part_name)
            part_num += 1

    index = {
        "original_file": file_path.name,
        "original_size": file_size,
        "chunk_size": chunk_size,
        "parts": parts
    }
    index_path = output_dir / f"{file_path.stem}.index.json"
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)

    return parts

def split_model(model_dir, output_dir, exclude_patterns=None):
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if exclude_patterns is None:
        exclude_patterns = [".index.json", ".gitattributes", ".gitignore"]

    all_parts = {}
    for file_path in model_dir.rglob("*"):
        if file_path.is_file():
            if any(file_path.name.endswith(pat) for pat in exclude_patterns):
                shutil.copy2(file_path, output_dir / file_path.relative_to(model_dir))
                continue
            if ".part" in file_path.name:
                continue
            parts = split_file(file_path, output_dir)
            if parts:
                all_parts[str(file_path.relative_to(model_dir))] = parts

    for file_path in model_dir.rglob("*"):
        if file_path.is_file():
            if any(file_path.name.endswith(pat) for pat in exclude_patterns):
                continue
            if ".part" in file_path.name:
                continue
            if file_path.stat().st_size <= CHUNK_SIZE:
                target = output_dir / file_path.relative_to(model_dir)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, target)

    total_index = {
        "source_dir": str(model_dir),
        "files": all_parts
    }
    with open(output_dir / "model_index.json", 'w') as f:
        json.dump(total_index, f, indent=2)

    print(f"分片完成，输出目录: {output_dir}")
    return output_dir

def merge_model(split_dir, output_dir):
    split_dir = Path(split_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = split_dir / "model_index.json"
    if not index_path.exists():
        print("错误: 找不到 model_index.json")
        sys.exit(-1)

    with open(index_path) as f:
        total_index = json.load(f)

    for orig_path, parts in total_index["files"].items():
        orig_path = Path(orig_path)
        target_path = output_dir / orig_path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if len(parts) == 1:
            shutil.copy2(split_dir / parts[0], target_path)
        else:
            with open(target_path, 'wb') as out_f:
                for part in parts:
                    part_path = split_dir / part
                    if not part_path.exists():
                        print(f"错误: 找不到分片 {part_path}")
                        sys.exit(-1)
                    with open(part_path, 'rb') as in_f:
                        shutil.copyfileobj(in_f, out_f)

    for file_path in split_dir.rglob("*"):
        if file_path.is_file():
            if file_path.name == "model_index.json":
                continue
            if ".part" in file_path.name:
                continue
            rel_path = file_path.relative_to(split_dir)
            target = output_dir / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(file_path, target)

    print(f"合并完成，输出目录: {output_dir}")
    return output_dir

def main():
    parser = argparse.ArgumentParser(description="模型分片/合并工具")
    parser.add_argument('--split', type=str, help="要拆分的模型目录")
    parser.add_argument('--output', type=str, required=True, help="输出目录")
    parser.add_argument('--merge', type=str, help="要合并的分片目录")

    args = parser.parse_args()

    if args.split and args.merge:
        print("错误: 不能同时使用 --split 和 --merge")
        sys.exit(-1)

    if args.split:
        split_model(args.split, args.output)
    elif args.merge:
        merge_model(args.merge, args.output)
    else:
        print("错误: 请指定 --split 或 --merge")
        sys.exit(-1)

if __name__ == "__main__":
    main()
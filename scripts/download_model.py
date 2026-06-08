#!/usr/bin/env python3
"""Download Qwen3-0.6B model from HuggingFace."""

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-0.6B",
    local_dir="models/Qwen3-0.6B",
    local_dir_use_symlinks=False,
    ignore_patterns=["*.msgpack", "*.h5", "pytorch_model*.bin", "flax_model.*"],
    resume_download=True,
)
print("Download complete: models/Qwen3-0.6B")

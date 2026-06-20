#!/usr/bin/env python3
"""Qwen3.5 SplitNN inference for Atlas board — uses OmSplitEngine + RemoteMiddleClient."""

from __future__ import annotations

import argparse
import sys
import time
import uuid
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
_deploy_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_deploy_dir / "scripts"))
sys.path.insert(0, str(_deploy_dir))

from qwen35_model_spec import load_metadata, load_bound_embed_head_metadata
from controller.remote_middle import RemoteMiddleClient, RemoteMiddleError, PROTOCOL_VERSION
from transformers import AutoTokenizer


# ── Sampling ────────────────────────────────────────────────────────────


def _sample(logits: np.ndarray, temperature: float, top_k: int) -> int:
    logits = logits.astype(np.float64)
    if temperature <= 0:
        return int(np.argmax(logits))
    logits /= temperature
    if 0 < top_k < len(logits):
        idx = np.argpartition(logits, -top_k)[-top_k:]
        mask = np.ones(len(logits), dtype=bool)
        mask[idx] = False
        logits[mask] = -np.inf
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    return int(np.random.choice(len(probs), p=probs))


# ── Main ────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://127.0.0.1:18080")
    parser.add_argument("--prefix-model", default="qwen3.5_split_prefix_max16384.om")
    parser.add_argument("--suffix-model", default="qwen3.5_split_suffix_max16384.om")
    parser.add_argument("--om-mode", choices=["om_split", "bound_embed_head"], default="om_split")
    parser.add_argument("--bound-asset-dir", default="")
    parser.add_argument("--tokenizer-dir", default="/root/slm_deploy")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--max-len", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--prompt", default="你好")
    parser.add_argument("--enable-thinking", action="store_true", default=False)
    parser.add_argument("--checksum", action="store_true")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--remote-model-name", default="")
    args = parser.parse_args()

    # Load metadata from prefix ONNX's companion JSON
    if args.om_mode == "bound_embed_head":
        if not args.bound_asset_dir:
            print("ERROR: --bound-asset-dir is required for bound_embed_head mode", file=sys.stderr)
            sys.exit(1)
        model_spec, split_config, _ = load_bound_embed_head_metadata(args.bound_asset_dir)
    else:
        prefix_meta = Path(args.prefix_model).with_suffix(".metadata.json")
        if not prefix_meta.exists():
            # Try same name but .onnx.metadata.json
            alt = Path(str(args.prefix_model).replace(".om", ".metadata.json"))
            if alt.exists():
                prefix_meta = alt
        if not prefix_meta.exists():
            print(f"ERROR: metadata not found at {prefix_meta}", file=sys.stderr)
            sys.exit(1)

        model_spec, split_config, _, _ = load_metadata(str(prefix_meta))

    # Build model name (must match server-side naming)
    prefix_ct = split_config.prefix_end
    suffix_ct = model_spec.num_hidden_layers - split_config.suffix_start
    middle_ct = split_config.suffix_start - split_config.prefix_end
    hs = model_spec.hidden_size
    # Model sizes lookup by hidden_size
    size_map = {1024: "0.8B", 2048: "2B", 2560: "4B", 4096: "9B", 5120: "27B"}
    size_str = size_map.get(hs, str(hs))
    model_name = args.remote_model_name or f"Qwen3.5-{size_str}-split-{prefix_ct}-{middle_ct}-{suffix_ct}"

    # Create engine
    from controller.engine.om_engine import OmSplitEngine
    engine = OmSplitEngine(
        model_id=model_name,
        max_len=args.max_len,
        model_spec=model_spec,
        split_config=split_config,
        prefix_om=args.prefix_model,
        suffix_om=args.suffix_model,
        mode=args.om_mode,
        bound_asset_dir=args.bound_asset_dir or None,
    )
    engine.load()

    # Create remote middle client
    remote = RemoteMiddleClient(
        server_url=args.server_url,
        model_name=model_name,
        hidden_size=model_spec.hidden_size,
        max_len=args.max_len,
        connect_timeout=args.timeout,
        read_timeout=args.timeout,
        checksum=args.checksum,
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)

    try:
        # Format prompt
        msgs = [{"role": "user", "content": args.prompt}]
        prompt_text = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        if not prompt_ids:
            print("ERROR: empty prompt", file=sys.stderr)
            sys.exit(1)
        prompt_ids = prompt_ids[-args.max_len :]

        session_id = uuid.uuid4().hex
        engine.start_session()
        remote.open(session_id)

        t0_total = time.perf_counter()

        # Prefill
        print(f"[prefill] {len(prompt_ids)} tokens...", flush=True)
        t_prefill = time.perf_counter()
        logits = None
        for pos, token_id in enumerate(prompt_ids):
            hidden_l4 = engine.run_prefix(int(token_id), pos)
            hidden_l20, middle_ms = remote.step(session_id, hidden_l4, pos)
            logits = engine.run_suffix(hidden_l20, pos)
        prefill_ms = (time.perf_counter() - t_prefill) * 1000.0
        print(f"[prefill] {prefill_ms:.0f} ms ({prefill_ms / len(prompt_ids):.1f} ms/tok)", flush=True)

        # Decode
        print("[decode]", flush=True)
        assert logits is not None
        current_logits = logits
        pos = len(prompt_ids)
        output_ids = []
        for step in range(args.max_tokens):
            if pos >= args.max_len:
                print("\n[length]", flush=True)
                break
            token_id = _sample(current_logits[0, 0, :], args.temperature, args.top_k)
            if token_id == tokenizer.eos_token_id:
                print("\n[stop]", flush=True)
                break
            output_ids.append(token_id)
            piece = tokenizer.decode([token_id], skip_special_tokens=True)
            print(piece, end="", flush=True)

            t_step = time.perf_counter()
            hidden_l4 = engine.run_prefix(int(token_id), pos)
            hidden_l20, middle_ms = remote.step(session_id, hidden_l4, pos)
            current_logits = engine.run_suffix(hidden_l20, pos)
            step_ms = (time.perf_counter() - t_step) * 1000.0
            pos += 1

        total_s = time.perf_counter() - t0_total
        tok_count = len(output_ids)
        print(f"\n[{tok_count} tokens in {total_s:.1f}s, "
              f"{tok_count / total_s:.1f} tok/s]", flush=True)

    finally:
        try:
            remote.close(session_id)
        finally:
            engine.end_session()
            engine.close()


if __name__ == "__main__":
    main()

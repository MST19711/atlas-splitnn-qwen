from __future__ import annotations

import json
import sys
import uuid

import numpy as np

from controller.engine.om_engine import OmSplitEngine
from controller.remote_middle import RemoteMiddleClient
from qwen35_model_spec import load_bound_embed_head_metadata
from transformers import AutoTokenizer


def sample_topk_only(logits: np.ndarray, temperature: float, top_k: int) -> int:
    work = logits.astype(np.float64).copy()
    if temperature <= 0:
        return int(np.argmax(work))
    work /= temperature
    if 0 < top_k < len(work):
        idx = np.argpartition(work, -top_k)[-top_k:]
        mask = np.ones(len(work), dtype=bool)
        mask[idx] = False
        work[mask] = -np.inf
    exp = np.exp(work - np.max(work))
    probs = exp / exp.sum()
    return int(np.random.choice(len(probs), p=probs))


def apply_repetition_penalty(logits: np.ndarray, token_counts: dict[int, int], penalty: float) -> np.ndarray:
    updated = logits.astype(np.float64).copy()
    if penalty in (0.0, 1.0):
        return updated
    for token_id, count in token_counts.items():
        if count <= 0:
            continue
        if updated[token_id] > 0:
            updated[token_id] /= penalty
        else:
            updated[token_id] *= penalty
    return updated


def sample_topk_topp(logits: np.ndarray, temperature: float, top_k: int, top_p: float) -> int:
    work = logits.astype(np.float64).copy()
    if temperature <= 0:
        return int(np.argmax(work))
    work /= temperature
    if 0 < top_k < len(work):
        idx = np.argpartition(work, -top_k)[-top_k:]
        mask = np.ones(len(work), dtype=bool)
        mask[idx] = False
        work[mask] = -np.inf
    if 0.0 < top_p < 1.0:
        order = np.argsort(work)[::-1]
        sorted_logits = work[order]
        probs = np.exp(sorted_logits - np.max(sorted_logits))
        probs /= probs.sum()
        cumsum = np.cumsum(probs)
        remove = cumsum > top_p
        if np.any(remove):
            remove[0] = False
            work[order[remove]] = -np.inf
    exp = np.exp(work - np.max(work))
    probs = exp / exp.sum()
    return int(np.random.choice(len(probs), p=probs))


def run_case(
    name: str,
    engine: OmSplitEngine,
    remote: RemoteMiddleClient,
    tokenizer,
    prompt_ids: list[int],
    picker,
    max_new_tokens: int,
) -> dict:
    session_id = uuid.uuid4().hex
    engine.start_session()
    remote.open(session_id)
    try:
        logits = None
        position = 0
        for token_id in prompt_ids:
            hidden_prefix = engine.run_prefix(int(token_id), position)
            hidden_middle, _ = remote.step(session_id, hidden_prefix, position)
            logits = engine.run_suffix(hidden_middle, position)
            position += 1
        assert logits is not None
        output_ids: list[int] = []
        token_counts: dict[int, int] = {}
        for _ in range(max_new_tokens):
            token_id = picker(logits[0, 0, :], token_counts)
            if token_id == tokenizer.eos_token_id:
                break
            output_ids.append(int(token_id))
            token_counts[token_id] = token_counts.get(token_id, 0) + 1
            hidden_prefix = engine.run_prefix(int(token_id), position)
            hidden_middle, _ = remote.step(session_id, hidden_prefix, position)
            logits = engine.run_suffix(hidden_middle, position)
            position += 1
        return {
            "name": name,
            "tokens": len(output_ids),
            "text": tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False),
        }
    finally:
        try:
            remote.close(session_id)
        finally:
            engine.end_session()


def main() -> int:
    if len(sys.argv) != 7:
        print(
            "usage: diag_splitnn_sampling.py <asset_dir> <server_url> <tokenizer_dir> <remote_model_name> <max_len> <prompt>",
            file=sys.stderr,
        )
        return 2
    asset_dir, server_url, tokenizer_dir, remote_model_name, max_len_text, prompt = sys.argv[1:]
    max_len = int(max_len_text)
    model_spec, split_config, _ = load_bound_embed_head_metadata(asset_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)[-max_len:]
    engine = OmSplitEngine(
        model_id="diag-splitnn-bound",
        max_len=max_len,
        model_spec=model_spec,
        split_config=split_config,
        mode="bound_embed_head",
        bound_asset_dir=asset_dir,
    )
    remote = RemoteMiddleClient(
        server_url=server_url,
        model_name=remote_model_name,
        hidden_size=model_spec.hidden_size,
        max_len=max_len,
    )
    engine.load()
    try:
        np.random.seed(1234)
        cases = [
            run_case(
                "greedy",
                engine,
                remote,
                tokenizer,
                prompt_ids,
                lambda raw, counts: int(np.argmax(raw)),
                120,
            ),
            run_case(
                "topk_only",
                engine,
                remote,
                tokenizer,
                prompt_ids,
                lambda raw, counts: sample_topk_only(raw, 0.4, 20),
                120,
            ),
            run_case(
                "topk_topp_rep",
                engine,
                remote,
                tokenizer,
                prompt_ids,
                lambda raw, counts: sample_topk_topp(
                    apply_repetition_penalty(raw, counts, 1.08),
                    0.4,
                    20,
                    0.85,
                ),
                120,
            ),
        ]
        for case in cases:
            print(json.dumps(case, ensure_ascii=False))
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

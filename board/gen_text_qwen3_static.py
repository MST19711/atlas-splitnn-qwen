#!/usr/bin/env python3
"""
Qwen3-0.6B FP16 静态窗口文本生成 (Atlas 200I DK A2 / Ascend310B4).

seq_len=32 固定窗口 + left-padding + causal mask。
Instruct 模型，使用 chat_template 格式化对话。

用法:
    python3 gen_text_qwen3_static.py --prompt "你好"
"""

import argparse, sys, time, warnings, numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages")
import acl
from transformers import AutoTokenizer

M = 0; H2D = 1; D2H = 2
SEQ_LEN = 32


def check(ret, msg):
    if ret != 0:
        raise RuntimeError(f"{msg} failed, ret={ret}")


# ── ACL 模型 ─────────────────────────────────────────────────────
class ACLModel:
    def __init__(self, path):
        check(acl.init(), "init")
        check(acl.rt.set_device(0), "set_device")
        self.mid, ret = acl.mdl.load_from_file(path)
        check(ret, "load")
        self.desc = acl.mdl.create_desc()
        check(acl.mdl.get_desc(self.desc, self.mid), "get_desc")
        self.out_sz = acl.mdl.get_output_size_by_index(self.desc, 0)
        self.in_sz0 = acl.mdl.get_input_size_by_index(self.desc, 0)
        self.in_sz1 = acl.mdl.get_input_size_by_index(self.desc, 1)

    def execute(self, input_ids, attention_mask):
        """input_ids: (1,32) int64, attention_mask: (1,32) int64 → logits (32, vocab) float16"""
        ids_ds = acl.mdl.create_dataset(); ods = acl.mdl.create_dataset()
        try:
            p0, _ = acl.rt.malloc(self.in_sz0, M)
            acl.rt.memcpy(p0, self.in_sz0, input_ids.ctypes.data, self.in_sz0, H2D)
            b0 = acl.create_data_buffer(p0, self.in_sz0)
            acl.mdl.add_dataset_buffer(ids_ds, b0)

            p1, _ = acl.rt.malloc(self.in_sz1, M)
            acl.rt.memcpy(p1, self.in_sz1, attention_mask.ctypes.data, self.in_sz1, H2D)
            b1 = acl.create_data_buffer(p1, self.in_sz1)
            acl.mdl.add_dataset_buffer(ids_ds, b1)

            po, _ = acl.rt.malloc(self.out_sz, M)
            bo = acl.create_data_buffer(po, self.out_sz)
            acl.mdl.add_dataset_buffer(ods, bo)

            check(acl.mdl.execute(self.mid, ids_ds, ods), "execute")

            host = np.empty(self.out_sz, np.uint8)
            acl.rt.memcpy(host.ctypes.data, self.out_sz, po, self.out_sz, D2H)
            return host.view(np.float16).reshape(SEQ_LEN, -1)
        finally:
            for b in (bo, b1, b0): acl.destroy_data_buffer(b)
            acl.mdl.destroy_dataset(ods); acl.mdl.destroy_dataset(ids_ds)
            for p in (po, p1, p0): acl.rt.free(p)

    def close(self):
        check(acl.mdl.unload(self.mid), "unload")
        check(acl.rt.reset_device(0), "reset_device")
        check(acl.finalize(), "finalize")


# ── 采样 ─────────────────────────────────────────────────────────
def sample(logits, temp=0.7, top_k=50, top_p=0.9):
    logits = logits.astype(np.float64)
    if temp > 0: logits /= temp
    if 0 < top_k < len(logits):
        idx = np.argpartition(logits, -top_k)[-top_k:]
        m = np.ones(len(logits), bool); m[idx] = False; logits[m] = -np.inf
    if top_p < 1.0:
        o = np.argsort(logits)[::-1]; e = np.exp(logits[o] - logits.max())
        c = np.cumsum(e) / e.sum(); cut = int(np.searchsorted(c, top_p)) + 1
        if cut < len(logits): logits[o[cut:]] = -np.inf
    e = np.exp(logits - logits.max()); p = e / e.sum()
    return int(np.random.choice(len(p), p=p))


# ── 生成 ─────────────────────────────────────────────────────────
def pad_left(token_ids, n_tokens):
    """Left-pad token_ids to 32, return (ids_32, mask_32, real_positions)."""
    pad_len = SEQ_LEN - n_tokens
    ids = np.array([[0] * pad_len + token_ids[-n_tokens:]], dtype=np.int64)
    msk = np.array([[0] * pad_len + [1] * n_tokens], dtype=np.int64)
    return ids, msk, n_tokens  # real_positions = number of non-pad tokens


def generate(model, tok, prompt, max_new=50, temp=0.7, top_k=50, top_p=0.9):
    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    n_prompt = len(prompt_ids)
    print(f"[Prompt: {n_prompt} tokens]")

    # Build sliding window buffer
    buffer = list(prompt_ids[-SEQ_LEN:])  # last N tokens
    n_real = min(len(prompt_ids), SEQ_LEN)

    t0 = time.time()
    for step in range(max_new):
        # Prepare left-padded input
        ids, mask, _ = pad_left(buffer, n_real)
        logits = model.execute(ids, mask)  # (32, vocab)
        next_logits = logits[n_real - 1]  # last real position
        tid = sample(next_logits, temp, top_k, top_p)

        if tid == tok.eos_token_id:
            break

        # Slide window
        buffer.append(tid)
        if len(buffer) > SEQ_LEN:
            buffer.pop(0)
        else:
            n_real += 1
        if n_real > SEQ_LEN:
            n_real = SEQ_LEN

        tok_text = tok.decode([tid], skip_special_tokens=True)
        sys.stdout.write(tok_text); sys.stdout.flush()

    elapsed = time.time() - t0
    n_gen = step + 1
    print(f"\n[{n_gen} tok, {elapsed:.1f}s, {n_gen/elapsed:.1f} tok/s]")


# ── 主入口 ───────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/root/slm_deploy/qwen3_fp16_seq32_tile.om")
    p.add_argument("--tokenizer-dir", default="/root/slm_deploy")
    p.add_argument("--prompt", default="你好")
    p.add_argument("--max-tokens", type=int, default=30)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.9)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)

    # Use chat template for instruct model
    messages = [{"role": "user", "content": args.prompt}]
    formatted = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False)
    print(f"Model: {args.model}")
    print(f"Formatted prompt: {repr(formatted[:100])}...")

    model = ACLModel(args.model)
    try:
        generate(model, tok, formatted, args.max_tokens,
                 args.temperature, args.top_k, args.top_p)
    finally:
        model.close()
        print("Done.")


if __name__ == "__main__":
    main()

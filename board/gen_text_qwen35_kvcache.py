#!/usr/bin/env python3
"""Qwen3.5-0.8B KV Cache 推理 (Atlas 200I DK A2 / Ascend310B4).
50 输入 / 49 输出: S(18) + conv(18) + K(6) + V(6).
"""

import argparse, sys, time, warnings, numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages")
import acl
from transformers import AutoTokenizer

M, H2D, D2H = 0, 1, 2
NL_DN = 18; NL_GA = 6
K_H = 16; K_DIM = 128; V_DIM = 128; CONV_D = 6144; CONV_KS = 4
KV_H = 2; HDIM = 256
S_BYTES = 1 * K_H * K_DIM * V_DIM * 2
C_BYTES = 1 * CONV_D * (CONV_KS - 1) * 2

# KV_BYTES and MAX are set dynamically from --max-len


def check(r, m):
    if r != 0: raise RuntimeError(f"{m} failed, ret={r}")


def _make_ds(dev_ids, dev_pos, s_src, c_src, k_src, v_src,
             dev_logits, s_dst, c_dst, k_dst, v_dst, kv_bytes, tag):
    ds_in = acl.mdl.create_dataset(); ds_out = acl.mdl.create_dataset()
    bufs_in, bufs_out = [], []
    for ptr, sz in [(dev_ids, 8), (dev_pos, 8)]:
        b = acl.create_data_buffer(ptr, sz); bufs_in.append(b)
        _, ret = acl.mdl.add_dataset_buffer(ds_in, b); check(ret, f"{tag} add_in")
    for ptrs, sz in [(s_src, S_BYTES), (c_src, C_BYTES), (k_src, kv_bytes), (v_src, kv_bytes)]:
        for p in ptrs:
            b = acl.create_data_buffer(p, sz); bufs_in.append(b)
            _, ret = acl.mdl.add_dataset_buffer(ds_in, b); check(ret, f"{tag} add_in")
    b = acl.create_data_buffer(dev_logits, 496640); bufs_out.append(b)
    _, ret = acl.mdl.add_dataset_buffer(ds_out, b); check(ret, f"{tag} add_out_logits")
    for ptrs, sz in [(s_dst, S_BYTES), (c_dst, C_BYTES), (k_dst, kv_bytes), (v_dst, kv_bytes)]:
        for p in ptrs:
            b = acl.create_data_buffer(p, sz); bufs_out.append(b)
            _, ret = acl.mdl.add_dataset_buffer(ds_out, b); check(ret, f"{tag} add_out")
    return ds_in, ds_out, bufs_in, bufs_out


class ACLModel:
    def __init__(self, path, max_len):
        kv_bytes = 1 * KV_H * max_len * HDIM * 2
        print("[init] Loading model...", flush=True)
        check(acl.init(), "init")
        check(acl.rt.set_device(0), "set_device")
        self.mid, ret = acl.mdl.load_from_file(path)
        check(ret, "load")
        self.desc = acl.mdl.create_desc()
        check(acl.mdl.get_desc(self.desc, self.mid), "get_desc")
        self.out_sz = acl.mdl.get_output_size_by_index(self.desc, 0)
        ni = acl.mdl.get_num_inputs(self.desc)
        no = acl.mdl.get_num_outputs(self.desc)
        print(f"  I/O: {ni} in, {no} out", flush=True)

        self._alloc = []

        def _alloc_ptr(sz):
            p, r = acl.rt.malloc(sz, M); check(r, "malloc"); self._alloc.append(p); return p

        self._di = _alloc_ptr(8); self._dp = _alloc_ptr(8)
        self._dl = _alloc_ptr(self.out_sz)

        # Pre-allocate cache: two sets (A=输入, B=输出, 交替)
        def _alloc_set():
            return (
                [_alloc_ptr(S_BYTES) for _ in range(NL_DN)],
                [_alloc_ptr(C_BYTES) for _ in range(NL_DN)],
                [_alloc_ptr(kv_bytes) for _ in range(NL_GA)],
                [_alloc_ptr(kv_bytes) for _ in range(NL_GA)],
            )
        self._sA, self._cA, self._kA, self._vA = _alloc_set()
        self._sB, self._cB, self._kB, self._vB = _alloc_set()

        # Pre-create AB/BA datasets
        print("[init] Creating datasets AB...", flush=True)
        self._ds_in_A, self._ds_out_B, _, _ = _make_ds(
            self._di, self._dp, self._sA, self._cA, self._kA, self._vA,
            self._dl, self._sB, self._cB, self._kB, self._vB, kv_bytes, "AB")
        print("[init] Creating datasets BA...", flush=True)
        self._ds_in_B, self._ds_out_A, _, _ = _make_ds(
            self._di, self._dp, self._sB, self._cB, self._kB, self._vB,
            self._dl, self._sA, self._cA, self._kA, self._vA, kv_bytes, "BA")
        print("[init] Done.", flush=True)

        self._hi = np.empty(8, np.uint8); self._hp = np.empty(8, np.uint8)
        self._ho = np.empty(self.out_sz, np.uint8)
        self._step = 0

    def execute(self, token_id, position):
        self._hi[:8] = np.array([token_id], np.int64).view(np.uint8)
        self._hp[:8] = np.array([position], np.int64).view(np.uint8)
        acl.rt.memcpy(self._di, 8, self._hi.ctypes.data, 8, H2D)
        acl.rt.memcpy(self._dp, 8, self._hp.ctypes.data, 8, H2D)

        if self._step % 2 == 0:
            ds_in, ds_out = self._ds_in_A, self._ds_out_B
        else:
            ds_in, ds_out = self._ds_in_B, self._ds_out_A
        self._step += 1

        check(acl.mdl.execute(self.mid, ds_in, ds_out), "execute")
        acl.rt.memcpy(self._ho.ctypes.data, self.out_sz, self._dl, self.out_sz, D2H)
        return self._ho.view(np.float16).flatten()

    def close(self):
        for p in self._alloc: acl.rt.free(p)
        check(acl.mdl.unload(self.mid), "unload")
        check(acl.rt.reset_device(0), "reset_device")
        check(acl.finalize(), "finalize")


def sample(logits, temp=0.7, top_k=40):
    logits = logits.astype(np.float64)
    if temp > 0: logits /= temp
    if 0 < top_k < len(logits):
        idx = np.argpartition(logits, -top_k)[-top_k:]
        m = np.ones(len(logits), bool); m[idx] = False; logits[m] = -np.inf
    e = np.exp(logits - logits.max()); p = e / e.sum()
    return int(np.random.choice(len(p), p=p))


def generate(model, tok, prompt, max_new=30, temp=0.7, top_k=40, max_len=256):
    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    n_prompt = min(len(prompt_ids), max_len)
    prompt_ids = prompt_ids[-n_prompt:]
    print(f"[Prompt: {n_prompt} tokens]\n", flush=True)

    t_prefill = time.time()
    for pos, tid in enumerate(prompt_ids):
        model.execute(int(tid), pos)
    t_prefill = time.time() - t_prefill

    current_id = int(prompt_ids[-1])
    n_gen = 0
    t_decode = time.time()
    for step in range(max_new):
        pos = n_prompt + step
        if pos >= max_len: break
        logits = model.execute(current_id, pos)
        tid = sample(logits, temp, top_k)
        if tid == tok.eos_token_id: break
        current_id = tid; n_gen += 1
        txt = tok.decode([tid], skip_special_tokens=True)
        sys.stdout.write(txt); sys.stdout.flush()
    t_decode = time.time() - t_decode

    tok_s = n_gen / t_decode if t_decode > 0 and n_gen > 0 else 0
    ms = t_decode / n_gen * 1000 if n_gen else 0
    print(f"\n\n[prefill {t_prefill:.1f}s, decode {n_gen} tok in {t_decode:.1f}s, {tok_s:.1f} tok/s, {ms:.0f} ms/tok]", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/root/slm_deploy/qwen3.5_kvcache_max256.om")
    p.add_argument("--tokenizer-dir", default="/root/slm_deploy")
    p.add_argument("--prompt", default="你好，请介绍一下你自己")
    p.add_argument("--max-tokens", type=int, default=30)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--max-len", type=int, default=256, help="Max context length")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)
    msgs = [{"role": "user", "content": args.prompt}]
    formatted = tok.apply_chat_template(msgs, tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)

    print(f"Model: {args.model}", flush=True)
    t_load = time.time()
    model = ACLModel(args.model, args.max_len)
    t_load = time.time() - t_load
    print(f"Model loaded in {t_load:.1f}s", flush=True)
    try:
        generate(model, tok, formatted, args.max_tokens, args.temperature, args.top_k, args.max_len)
    finally:
        model.close()
        print("Done.", flush=True)


if __name__ == "__main__":
    main()

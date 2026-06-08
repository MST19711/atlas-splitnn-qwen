#!/usr/bin/env python3
"""Qwen3-0.6B KV Cache ACL inference on Atlas 200I DK A2."""
import argparse, sys, time, warnings, numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages")
import acl
from transformers import AutoTokenizer

M, H2D, D2H = 0, 1, 2
MAX, NL = 256, 28
KV_BYTES = 8 * MAX * 128 * 2


def check(r, m):
    if r != 0: raise RuntimeError(f"{m} failed, ret={r}")


class ACLModel:
    def __init__(self, path):
        check(acl.init(), "init")
        check(acl.rt.set_device(0), "set_device")
        self.mid, ret = acl.mdl.load_from_file(path)
        check(ret, "load")
        self.desc = acl.mdl.create_desc()
        check(acl.mdl.get_desc(self.desc, self.mid), "get_desc")
        self.out_sz_logits = acl.mdl.get_output_size_by_index(self.desc, 0)
        self.n_in = acl.mdl.get_num_inputs(self.desc)
        self.n_out = acl.mdl.get_num_outputs(self.desc)
        print(f"  I/O: {self.n_in} in, {self.n_out} out")

    def execute(self, token_id, position, kv_dev):
        """kv_dev: list of 56 device pointers. Returns logits array + updates kv_dev in-place."""
        ds_in = acl.mdl.create_dataset()
        ds_out = acl.mdl.create_dataset()
        bufs, ptrs = [], []

        try:
            # input_ids (1,1) int64
            inp = np.array([[token_id]], dtype=np.int64)
            pi, _ = acl.rt.malloc(8, M); ptrs.append(pi)
            acl.rt.memcpy(pi, 8, inp.ctypes.data, 8, H2D)
            b = acl.create_data_buffer(pi, 8); bufs.append(b)
            _, ret = acl.mdl.add_dataset_buffer(ds_in, b); check(ret, "add_inp")

            # position (1,) int64
            pos = np.array([position], dtype=np.int64)
            pp, _ = acl.rt.malloc(8, M); ptrs.append(pp)
            acl.rt.memcpy(pp, 8, pos.ctypes.data, 8, H2D)
            b = acl.create_data_buffer(pp, 8); bufs.append(b)
            _, ret = acl.mdl.add_dataset_buffer(ds_in, b); check(ret, "add_pos")

            # past K/V (56 device pointers)
            for ptr in kv_dev:
                b = acl.create_data_buffer(ptr, KV_BYTES); bufs.append(b)
                _, ret = acl.mdl.add_dataset_buffer(ds_in, b); check(ret, "add_kv")

            # Output buffers: logits + 56 K/V
            out_host = []
            out_dev = []
            # logits
            po, _ = acl.rt.malloc(self.out_sz_logits, M); ptrs.append(po)
            b = acl.create_data_buffer(po, self.out_sz_logits)
            _, ret = acl.mdl.add_dataset_buffer(ds_out, b); check(ret, "add_out_logits")
            bufs.append(b); out_dev.append(po)
            host_lg = np.empty(self.out_sz_logits, np.uint8); out_host.append(host_lg)

            # present K/V
            for _ in range(2 * NL):
                p, _ = acl.rt.malloc(KV_BYTES, M); ptrs.append(p)
                b = acl.create_data_buffer(p, KV_BYTES)
                _, ret = acl.mdl.add_dataset_buffer(ds_out, b); check(ret, "add_out_kv")
                bufs.append(b); out_dev.append(p)
                out_host.append(np.empty(KV_BYTES, np.uint8))

            check(acl.mdl.execute(self.mid, ds_in, ds_out), "execute")

            # D2H all outputs
            for i, (dev, host) in enumerate(zip(out_dev, out_host)):
                acl.rt.memcpy(host.ctypes.data, host.nbytes, dev, host.nbytes, D2H)

            logits = out_host[0].view(np.float16).flatten()

            # Copy updated K/V from output back to input device buffers
            for i in range(2 * NL):
                out_host_kv = out_host[1 + i]
                acl.rt.memcpy(kv_dev[i], KV_BYTES, out_host_kv.ctypes.data, KV_BYTES, H2D)

            return logits

        finally:
            for b in bufs: acl.destroy_data_buffer(b)
            acl.mdl.destroy_dataset(ds_in); acl.mdl.destroy_dataset(ds_out)
            for p in ptrs: acl.rt.free(p)

    def close(self):
        check(acl.mdl.unload(self.mid), "unload")
        check(acl.rt.reset_device(0), "reset_device")
        check(acl.finalize(), "finalize")


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


def generate(model, tok, prompt, max_new=30, temp=0.7, top_k=40, top_p=0.9):
    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    n_prompt = min(len(prompt_ids), MAX)
    prompt_ids = prompt_ids[-n_prompt:]
    print(f"[Prompt: {n_prompt} tokens]\n")

    # Allocate KV cache on device
    kv_dev = []
    for _ in range(2 * NL):
        p, ret = acl.rt.malloc(KV_BYTES, M); check(ret, "malloc kv")
        kv_dev.append(p)
    try:
        t0 = time.time()

        # Prefill
        for pos, tid in enumerate(prompt_ids):
            model.execute(int(tid), pos, kv_dev)

        # Decode
        current_id = int(prompt_ids[-1])
        n_gen = 0
        for step in range(max_new):
            pos = n_prompt + step
            if pos >= MAX: break
            logits = model.execute(current_id, pos, kv_dev)
            tid = sample(logits, temp, top_k, top_p)
            if tid == tok.eos_token_id: break
            current_id = tid; n_gen += 1
            txt = tok.decode([tid], skip_special_tokens=True)
            sys.stdout.write(txt); sys.stdout.flush()

        dt = time.time() - t0
        tok_s = n_gen / dt if dt > 0 else 0
        ms = dt / n_gen * 1000 if n_gen else 0
        print(f"\n\n[{n_gen} tok, {dt:.1f}s, {tok_s:.1f} tok/s, {ms:.0f} ms/tok]")
    finally:
        for p in kv_dev: acl.rt.free(p)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/root/slm_deploy/qwen3_kvcache_max256_b4.om")
    p.add_argument("--tokenizer-dir", default="/root/slm_deploy")
    p.add_argument("--prompt", default="你好，请介绍一下你自己")
    p.add_argument("--max-tokens", type=int, default=30)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.9)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)
    msgs = [{"role": "user", "content": args.prompt}]
    formatted = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    print(f"Model: {args.model}")
    model = ACLModel(args.model)
    try:
        generate(model, tok, formatted, args.max_tokens, args.temperature, args.top_k, args.top_p)
    finally:
        model.close()
        print("Done.")


if __name__ == "__main__":
    main()

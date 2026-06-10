#!/usr/bin/env python3
"""Qwen3-0.6B KV Cache ACL inference on Atlas 200I DK A2.
v3: pre-created double datasets AB/BA.
Fixed: output data buffer lifecycle, added validation logging."""

import argparse, sys, time, warnings, numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages")
import acl
from transformers import AutoTokenizer

M, H2D, D2H = 0, 1, 2
MAX, NL = 256, 28
KV_BYTES = 8 * MAX * 128 * 2


def check(ret, msg):
    if ret != 0:
        raise RuntimeError(f"[{msg}] failed, ret={ret}")


def _make_dataset(dev_ids, dev_pos, kv_src, dev_logits, kv_dst, tag):
    """Return (ds_in, ds_out, bufs_in, bufs_out)."""
    ds_in = acl.mdl.create_dataset()
    ds_out = acl.mdl.create_dataset()
    bufs_in, bufs_out = [], []

    # inputs: ids, position, 56 K/V
    for idx, (ptr, sz) in enumerate([(dev_ids, 8), (dev_pos, 8)] +
                                     [(p, KV_BYTES) for p in kv_src]):
        b = acl.create_data_buffer(ptr, sz)
        assert b != 0, f"{tag} create_data_buffer in[{idx}] returned 0"
        _, ret = acl.mdl.add_dataset_buffer(ds_in, b)
        check(ret, f"{tag} add_dataset_buffer in[{idx}]")
        bufs_in.append(b)

    # outputs: logits + 56 K/V
    b = acl.create_data_buffer(dev_logits, 303872)
    assert b != 0, f"{tag} create_data_buffer out[logits] returned 0"
    _, ret = acl.mdl.add_dataset_buffer(ds_out, b)
    check(ret, f"{tag} add_dataset_buffer out[logits]")
    bufs_out.append(b)
    for idx, ptr in enumerate(kv_dst):
        b = acl.create_data_buffer(ptr, KV_BYTES)
        assert b != 0, f"{tag} create_data_buffer out[K/V {idx}] returned 0"
        _, ret = acl.mdl.add_dataset_buffer(ds_out, b)
        check(ret, f"{tag} add_dataset_buffer out[K/V {idx}]")
        bufs_out.append(b)

    print(f"  {tag} created: ds_in ok, ds_out ok, "
          f"{len(bufs_in)} in bufs, {len(bufs_out)} out bufs")
    return ds_in, ds_out, bufs_in, bufs_out


class ACLModel:
    def __init__(self, path):
        print("[init] acl.init...")
        check(acl.init(), "init")
        print("[init] acl.rt.set_device(0)...")
        check(acl.rt.set_device(0), "set_device")
        print("[init] acl.mdl.load_from_file...")
        self.mid, ret = acl.mdl.load_from_file(path)
        check(ret, "load_from_file")
        print("[init] model loaded")
        self.desc = acl.mdl.create_desc()
        check(acl.mdl.get_desc(self.desc, self.mid), "get_desc")
        self.out_sz = acl.mdl.get_output_size_by_index(self.desc, 0)
        print(f"  I/O: {acl.mdl.get_num_inputs(self.desc)} in, "
              f"{acl.mdl.get_num_outputs(self.desc)} out")

        self._allocated = []

        print("[init] alloc dev_ids(8)...")
        self._dev_ids, _ = acl.rt.malloc(8, M); assert self._dev_ids
        self._allocated.append(self._dev_ids)
        print(f"  dev_ids={hex(self._dev_ids)}")

        print("[init] alloc dev_pos(8)...")
        self._dev_pos, _ = acl.rt.malloc(8, M); assert self._dev_pos
        self._allocated.append(self._dev_pos)
        print(f"  dev_pos={hex(self._dev_pos)}")

        print("[init] alloc dev_logits...")
        self._dev_logits, _ = acl.rt.malloc(self.out_sz, M)
        assert self._dev_logits
        self._allocated.append(self._dev_logits)
        print(f"  dev_logits={hex(self._dev_logits)}")

        print("[init] alloc K/V double buffers...")
        kv_a, kv_b = [], []
        for i in range(2 * NL):
            pa, _ = acl.rt.malloc(KV_BYTES, M)
            assert pa != 0, f"malloc kv_a[{i}] returned 0"
            kv_a.append(pa); self._allocated.append(pa)
            pb, _ = acl.rt.malloc(KV_BYTES, M)
            assert pb != 0, f"malloc kv_b[{i}] returned 0"
            kv_b.append(pb); self._allocated.append(pb)
        print(f"  K/V allocated: {len(kv_a)}+{len(kv_b)} = {len(self._allocated)-3} tensors")

        # Pre-create two dataset pairs
        print("[init] creating dataset pair AB...")
        self._ds_in_AB, self._ds_out_AB, self._bufs_in_AB, self._bufs_out_AB = \
            _make_dataset(self._dev_ids, self._dev_pos, kv_a, self._dev_logits, kv_b, "AB")

        print("[init] creating dataset pair BA...")
        self._ds_in_BA, self._ds_out_BA, self._bufs_in_BA, self._bufs_out_BA = \
            _make_dataset(self._dev_ids, self._dev_pos, kv_b, self._dev_logits, kv_a, "BA")

        print("[init] datasets created successfully!")
        print(f"  bufs_in_AB={len(self._bufs_in_AB)}, bufs_out_AB={len(self._bufs_out_AB)}")
        print(f"  bufs_in_BA={len(self._bufs_in_BA)}, bufs_out_BA={len(self._bufs_out_BA)}")

        # Host staging
        self._host_ids = np.empty(8, np.uint8)
        self._host_pos = np.empty(8, np.uint8)
        self._host_logits = np.empty(self.out_sz, np.uint8)

        self._step = 0

    def execute(self, token_id, position):
        self._host_ids[:8] = np.array([token_id], np.int64).view(np.uint8)
        self._host_pos[:8] = np.array([position], np.int64).view(np.uint8)
        acl.rt.memcpy(self._dev_ids, 8, self._host_ids.ctypes.data, 8, H2D)
        acl.rt.memcpy(self._dev_pos, 8, self._host_pos.ctypes.data, 8, H2D)

        if self._step % 2 == 0:
            ds_in, ds_out = self._ds_in_AB, self._ds_out_AB
        else:
            ds_in, ds_out = self._ds_in_BA, self._ds_out_BA
        self._step += 1

        # debug
        check(acl.mdl.execute(self.mid, ds_in, ds_out), "execute")
        # debug

        acl.rt.memcpy(self._host_logits.ctypes.data, self.out_sz,
                      self._dev_logits, self.out_sz, D2H)
        # debug
        return self._host_logits.view(np.float16).flatten()

    def close(self):
        print("[close] destroying all data buffers...")
        for tag, bufs_in, bufs_out in [
            ("AB_in", self._bufs_in_AB, None),
            ("AB_out", self._bufs_out_AB, None),
            ("BA_in", self._bufs_in_BA, None),
            ("BA_out", self._bufs_out_BA, None),
        ]:
            for b in (bufs_in or []):
                acl.destroy_data_buffer(b)
        print("[close] destroying datasets...")
        acl.mdl.destroy_dataset(self._ds_in_AB)
        acl.mdl.destroy_dataset(self._ds_out_AB)
        acl.mdl.destroy_dataset(self._ds_in_BA)
        acl.mdl.destroy_dataset(self._ds_out_BA)
        print("[close] freeing device memory...")
        for p in self._allocated:
            acl.rt.free(p)
        print("[close] unload, reset, finalize...")
        check(acl.mdl.unload(self.mid), "unload")
        check(acl.rt.reset_device(0), "reset_device")
        check(acl.finalize(), "finalize")
        print("[close] done")


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


def generate(model, tok, prompt, max_new=5, temp=0.7, top_k=40, top_p=0.9):
    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    n_prompt = min(len(prompt_ids), MAX)
    prompt_ids = prompt_ids[-n_prompt:]
    print(f"[Prompt: {n_prompt} tokens]\n")

    t_prefill = time.time()
    for pos, tid in enumerate(prompt_ids):
        model.execute(int(tid), pos)
    t_prefill = time.time() - t_prefill

    current_id = int(prompt_ids[-1])
    n_gen = 0
    t_decode = time.time()
    for step in range(max_new):
        pos = n_prompt + step
        if pos >= MAX: break
        logits = model.execute(current_id, pos)
        tid = sample(logits, temp, top_k, top_p)
        if tid == tok.eos_token_id: break
        current_id = tid; n_gen += 1
        txt = tok.decode([tid], skip_special_tokens=True)
        sys.stdout.write(txt); sys.stdout.flush()
    t_decode = time.time() - t_decode

    tok_s = n_gen / t_decode if t_decode > 0 and n_gen > 0 else 0
    ms = t_decode / n_gen * 1000 if n_gen else 0
    print(f"\n\n[prefill {t_prefill:.1f}s, decode {n_gen} tok in {t_decode:.1f}s, {tok_s:.1f} tok/s, {ms:.0f} ms/tok]")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/root/slm_deploy/qwen3_kvcache_max256_cann7.om")
    p.add_argument("--tokenizer-dir", default="/root/slm_deploy")
    p.add_argument("--prompt", default="你好")
    p.add_argument("--max-tokens", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.9)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)
    msgs = [{"role": "user", "content": args.prompt}]
    formatted = tok.apply_chat_template(msgs, tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)

    print(f"Model: {args.model}")
    model = ACLModel(args.model)
    try:
        generate(model, tok, formatted, args.max_tokens, args.temperature, args.top_k, args.top_p)
    finally:
        model.close()
        print("Done.")


if __name__ == "__main__":
    main()

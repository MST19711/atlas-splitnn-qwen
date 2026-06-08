import argparse
import sys

import numpy as np

sys.path.insert(0, "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages")
import acl  # noqa: E402


ACL_MEM_MALLOC_NORMAL_ONLY = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2


def check(ret, message):
    if ret != 0:
        raise RuntimeError(f"{message} failed, acl ret={ret}")


def malloc_device(array):
    ptr, ret = acl.rt.malloc(array.nbytes, ACL_MEM_MALLOC_NORMAL_ONLY)
    check(ret, "acl.rt.malloc")
    ret = acl.rt.memcpy(ptr, array.nbytes, array.ctypes.data, array.nbytes, ACL_MEMCPY_HOST_TO_DEVICE)
    check(ret, "acl.rt.memcpy H2D")
    return ptr


def add_buffer(dataset, ptr, size):
    data_buffer = acl.create_data_buffer(ptr, size)
    if data_buffer is None:
        raise RuntimeError("acl.create_data_buffer failed")
    _, ret = acl.mdl.add_dataset_buffer(dataset, data_buffer)
    check(ret, "acl.mdl.add_dataset_buffer")
    return data_buffer


def main():
    parser = argparse.ArgumentParser(description="Run one fixed-input ACL inference for the OM model.")
    parser.add_argument("--model", default="/root/slm_deploy/qwen3_fp16_seq1.om")
    parser.add_argument("--seq-len", type=int, default=1)
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Seq len: {args.seq_len}")

    ret = acl.init()
    check(ret, "acl.init")
    try:
        ret = acl.rt.set_device(0)
        check(ret, "acl.rt.set_device")

        print("Loading model...")
        model_id, ret = acl.mdl.load_from_file(args.model)
        check(ret, "acl.mdl.load_from_file")
        try:
            desc = acl.mdl.create_desc()
            ret = acl.mdl.get_desc(desc, model_id)
            check(ret, "acl.mdl.get_desc")

            num_inputs = acl.mdl.get_num_inputs(desc)
            num_outputs = acl.mdl.get_num_outputs(desc)
            print(f"Inputs: {num_inputs}, Outputs: {num_outputs}")

            for i in range(num_inputs):
                name = acl.mdl.get_input_name_by_index(desc, i)
                dims, ret = acl.mdl.get_input_dims(desc, i)
                size = acl.mdl.get_input_size_by_index(desc, i)
                dtype = acl.mdl.get_input_data_type(desc, i)
                print(f"  input[{i}]: name={name}, dims={dims}, size={size}, dtype={dtype}")

            for i in range(num_outputs):
                name = acl.mdl.get_output_name_by_index(desc, i)
                dims, ret = acl.mdl.get_output_dims(desc, i)
                size = acl.mdl.get_output_size_by_index(desc, i)
                dtype = acl.mdl.get_output_data_type(desc, i)
                print(f"  output[{i}]: name={name}, dims={dims}, size={size}, dtype={dtype}")

            input_ids = np.ones((1, args.seq_len), dtype=np.int64)
            attention_mask = np.ones((1, args.seq_len), dtype=np.int64)
            inputs = [input_ids, attention_mask]

            input_dataset = acl.mdl.create_dataset()
            output_dataset = acl.mdl.create_dataset()
            device_ptrs = []
            buffers = []
            host_outputs = []

            try:
                for array in inputs:
                    ptr = malloc_device(array)
                    device_ptrs.append(ptr)
                    buffers.append(add_buffer(input_dataset, ptr, array.nbytes))

                output_count = num_outputs
                for index in range(output_count):
                    size = acl.mdl.get_output_size_by_index(desc, index)
                    ptr, ret = acl.rt.malloc(size, ACL_MEM_MALLOC_NORMAL_ONLY)
                    check(ret, "acl.rt.malloc output")
                    device_ptrs.append(ptr)
                    buffers.append(add_buffer(output_dataset, ptr, size))
                    host_outputs.append((ptr, np.empty(size, dtype=np.uint8)))

                print("Executing...")
                ret = acl.mdl.execute(model_id, input_dataset, output_dataset)
                check(ret, "acl.mdl.execute")
                print("Execute done!")

                for index, (ptr, host) in enumerate(host_outputs):
                    ret = acl.rt.memcpy(host.ctypes.data, host.nbytes, ptr, host.nbytes, ACL_MEMCPY_DEVICE_TO_HOST)
                    check(ret, "acl.rt.memcpy D2H")

                    output_f16 = host.view(np.float16)
                    flat = output_f16.flatten()
                    print(f"output[{index}]: bytes={host.nbytes}, elements={flat.shape[0]}")
                    print(f"  min={flat.min():.6f}, max={flat.max():.6f}, mean={flat.mean():.6f}")
                    print(f"  first10={flat[:10].tolist()}")
                    print(f"  last10={flat[-10:].tolist()}")

                print("ACL inference passed!")
            finally:
                for buffer in buffers:
                    acl.destroy_data_buffer(buffer)
                acl.mdl.destroy_dataset(input_dataset)
                acl.mdl.destroy_dataset(output_dataset)
                for ptr in device_ptrs:
                    acl.rt.free(ptr)
                acl.mdl.destroy_desc(desc)
        finally:
            ret = acl.mdl.unload(model_id)
            check(ret, "acl.mdl.unload")
            ret = acl.rt.reset_device(0)
            check(ret, "acl.rt.reset_device")
    finally:
        ret = acl.finalize()
        check(ret, "acl.finalize")


if __name__ == "__main__":
    main()

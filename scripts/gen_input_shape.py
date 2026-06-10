#!/usr/bin/env python3
"""读取 ONNX 文件，输出 ATC 所需的 INPUT_SHAPE 字符串。

用法:
  pixi run python scripts/gen_input_shape.py om_out/model.onnx
  # 输出: input_ids:1,1;position:1;...
"""

import sys, onnx

def main():
    model = onnx.load(sys.argv[1], load_external_data=False)
    shapes = []
    for inp in model.graph.input:
        dims = [str(d.dim_value) if d.dim_value > 0 else "-1" for d in inp.type.tensor_type.shape.dim]
        shapes.append(f"{inp.name}:{','.join(dims)}")
    print(";".join(shapes))

if __name__ == "__main__":
    main()

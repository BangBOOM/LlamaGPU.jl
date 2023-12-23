# LlamaGPU.jl

The goal of this project is to inference llama7B and even larger model on GPUs like RTX2080 which does not have enough memory to fit the whole model.

## Usage

1. use `python mode2bin.py --src path\to\src --tgt path\to\tgt --layer_cnt 32` to convert the pytorch pth file to binary format file.
2. use `julia bin2bfbin.jl path\to\src path\to\tgt 4096` (4096 is embedding dim) to convert the float32 binary model to bfloat16 binary model.

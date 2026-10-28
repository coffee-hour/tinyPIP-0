# tinyPIP-0

tinyPIP-0 is a small, self-contained character-level transformer language model written in PyTorch. It is a standalone project. It is not related to any other model repository and shares no code or configs with one.

It is designed to run on modest hardware, including a Chromebook or a free Colab runtime. The whole project is four Python files and one HTML file.

## Files

- model.py: the transformer (config, attention, MLP, blocks, generation). Running it prints the parameter count.
- train.py: trains on TinyShakespeare at the character level and writes out/tinypip0.pt.
- generate.py: loads a checkpoint and generates text.
- index.html: a zero-build browser visualizer with a parameter calculator and architecture view. Styled in amber #ffb000 on #0d0d0d with JetBrains Mono.

## Architecture

| Setting | Value |
|---|---|
| Type | decoder-only transformer, causal attention |
| Tokenization | character level (about 65 symbols for TinyShakespeare) |
| Layers (n_layer) | 2 |
| Embedding dim (n_embd) | 64 |
| Heads (n_head) | 4 (head dim 16) |
| Context (block_size) | 128 |
| MLP | 4x width, GELU, no bias |
| Norm | pre-LayerNorm |
| Output head | tied to token embedding |
| Total parameters | 111,296 |

Note on sizing: a dim of 128 with 4 to 6 layers gives roughly 0.5M to 1M parameters with this block design, which is well above 100k. To stay near 100k, the default uses dim 64 with 2 layers. Change Config in model.py to scale up. The parameter count above is a hand calculation. Confirm it with python model.py.

## Train

Local:

```
pip install torch
python train.py
```

Colab (one cell):

```
!git clone https://github.com/coffee-hour/tinyPIP-0.git
%cd tinyPIP-0
!python train.py
```

Training takes a few minutes on CPU and is faster on a GPU. It prints train and validation loss every 500 steps and saves out/tinypip0.pt.

## Generate

```
python generate.py --ckpt out/tinypip0.pt --prompt "ROMEO:" --tokens 300 --temperature 0.8 --top-k 40
```

## Browser visualizer

Open index.html in any browser. It needs no server and no build.

## Status

This commit contains the code only. Pretrained weights are not included yet. Run train.py to produce them. Once trained, out/tinypip0.pt can be added to the repo if you want it hosted.

## License

Not yet specified.

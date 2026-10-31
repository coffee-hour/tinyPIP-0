# tinyPIP-0: How It Works

This document explains every major piece of tinyPIP-0: the model architecture, the tokenizer, the data pipeline, the training loop, checkpointing, text generation, the browser export, and the front-end visualizer and chat page. It is written from the source files in this repository.

## 1. Big picture

tinyPIP-0 is a character-level, decoder-only transformer (the same family as GPT). It reads a sequence of characters and predicts the next character. Training repeatedly shows it windows of text and adjusts its weights to lower the cross-entropy loss on the next character. Generation then repeatedly samples one character from the model's predicted distribution and appends it to the input.

The project is deliberately small so it runs on a laptop CPU, a Chromebook, or a free Colab runtime. The core model is in model.py.

Data flows through the system like this:

1. Raw text is assembled from input.txt (TinyShakespeare), data/curated_*.txt, and an optional data_phase2.txt built from SmolTalk.
2. The combined text is turned into a character vocabulary and then into a long tensor of integer ids.
3. train.py samples random 128-character windows, runs the model, computes loss, and updates weights with AdamW.
4. The result is saved as out/tinypip0.pt, a single checkpoint file.
5. Three consumers read the checkpoint: generate.py (command line), export_web.py (writes browser files into web/model), and the browser chat in web/ (which loads those files).

## 2. Files at a glance

| Path | Role |
|---|---|
| model.py | Model definition: config, attention, MLP, transformer block, full model, and generation loop. Running it prints the parameter count. |
| train.py | Data loading, vocabulary building, optimizer setup, training loop, evaluation, resume logic, and checkpoint saving. |
| generate.py | Loads a checkpoint and produces text, either for one prompt or in an interactive chat loop. |
| fetch_data.py | Optional. Streams SmolTalk conversations from the Hugging Face datasets API and writes data_phase2.txt in a User:/Pip: transcript format. |
| export_web.py | Converts a checkpoint into weights.bin (raw float32) plus manifest.json (config, vocabulary, and tensor layout) for the browser. |
| train.sh | One-command launcher. Installs requirements if needed, reports the device, optionally runs fetch_data.py (--huge-data), then runs train.py and passes flags through. |
| requirements.txt | Python dependencies. |
| index.html (root) | Zero-build browser visualizer with a parameter calculator and architecture view. Styled in amber #ffb000 on #0d0d0d with JetBrains Mono. |
| visualizer.html (root) | A second visualizer page in the repository. Its contents were not reviewed for this document. |
| web/ | Browser chat page: index.html, app.js, a README, and a model/ folder that holds exported weights and manifest. |
| data/ | Curated plain-text training files: curated_01_linux_git.txt, curated_02_python_torch.txt, curated_03_hardware.txt, curated_04_reasoning_identity.txt. |
| out/ | Checkpoint output directory. Contains tinypip0.pt and tinypip0-checkpoint.zip in the repository. |
| .github/workflows/ | CI workflow definitions. Their contents were not reviewed for this document. |
| .devcontainer/ | Dev container configuration. Not reviewed for this document. |
| .gitignore | Git ignore rules. |

## 3. Model architecture (model.py)

### Config

A dataclass holds every hyperparameter. Defaults:

| Field | Default | Meaning |
|---|---|---|
| vocab_size | 65 | Number of distinct characters. Overwritten at training time from the data. |
| block_size | 128 | Context length in characters (the longest sequence the model attends over). |
| n_layer | 2 | Number of transformer blocks. |
| n_head | 4 | Number of attention heads. |
| n_embd | 64 | Embedding (model) width. Must be divisible by n_head. |
| dropout | 0.0 | Dropout probability. Off by default. |

The default model has 111,296 parameters per the README, which gives this as a hand calculation. Running python model.py prints the exact count.

### Forward pass

For a batch of token ids of shape (B, T):

1. Token embedding: tok_emb maps each id to a 64-dim vector.
2. Position embedding: pos_emb adds a learned vector for each position from 0 to T-1.
3. Dropout on the sum.
4. n_layer transformer blocks.
5. Final LayerNorm (ln_f).
6. Output head: lm_head projects each position back to vocab_size logits.

The model asserts that T is no longer than block_size.

### Transformer block (pre-LayerNorm residual)

Each Block computes:

- x = x + Attention(LayerNorm(x))
- x = x + MLP(LayerNorm(x))

### Causal self-attention

- A single linear layer (qkv, no bias) produces query, key, and value together, which are split and reshaped into n_head heads of size n_embd / n_head (16 by default).
- F.scaled_dot_product_attention is called with is_causal=True, so each position can only attend to itself and earlier positions.
- An output projection (proj, no bias) recombines the heads.

### MLP

A linear layer expands width 4x (64 to 256), applies GELU, then projects back down. There are no biases in the linear layers.

### Weight tying

The output head shares its weight matrix with the token embedding (lm_head.weight is tok_emb.weight). This saves about 4,160 parameters (65 x 64) and is why the embedding is not counted twice.

### Initialization

All Linear and Embedding weights are drawn from N(0, 0.02). Linear biases, if present, are zeroed.

### Generation

model.generate(idx, max_new_tokens, temperature, top_k) runs without gradients. On each step it:

1. Crops the running sequence to the last block_size tokens.
2. Runs the model and takes the logits at the final position.
3. Divides logits by temperature (floored at 1e-5).
4. If top_k is set, keeps only the top k logits and sets the rest to negative infinity.
5. Samples one id from the softmax distribution with torch.multinomial and appends it.

## 4. Tokenizer and vocabulary

tinyPIP-0 has no separate tokenizer library. Its tokenizer is a character map built directly in train.py.

- Without a checkpoint, build_vocab assigns ids 0 to N-1 to the sorted set of distinct characters in the training text. Sorting makes the mapping deterministic.
- With a checkpoint, every existing id is kept exactly as it was. Any new characters in the new text are appended with the next free ids, in sorted order. The vocabulary never shrinks.
- The mapping is stored in the checkpoint as itos (id to character). stoi is its reverse, built at load time.
- Encoding is a direct per-character lookup. Decoding joins the characters for each id.

Because the tokenizer is per-character, there is no unknown-token concept. generate.py refuses a one-shot prompt that contains a character outside the vocabulary and, in chat mode, skips that message and prints the offending characters.

## 5. Data pipeline

### Sources, in the order train.py assembles them

1. input.txt: TinyShakespeare. Downloaded from the karpathy/char-rnn repository if it is missing. It is only read, never written.
2. Every file matching data/curated_*.txt, in sorted filename order. These are the four curated files in the repository.
3. data_phase2.txt, if it exists. This is produced by fetch_data.py and is optional.

Missing files are skipped. Each extra source is appended after a newline is added if needed.

### Splitting

The combined text is encoded to a long tensor of ids. The first 90 percent is the training split and the last 10 percent is the validation split. Splitting is by position, not by shuffling.

### Batching

get_batch draws batch_size random start positions (default 32). For each start index i it takes x = text[i : i + 128] and y = text[i + 1 : i + 129]. The target is the input shifted by one character, so the model learns next-character prediction at every position at once.

### fetch_data.py (optional SmolTalk phase)

- Pages through rows of the HuggingFaceTB/smol-smoltalk dataset via the datasets-server rows API, 100 rows per page.
- Keeps only user and assistant turns. User turns are labeled User: and assistant turns are labeled Pip:, matching the transcript style used by generate.py chat mode.
- Strips non-ASCII characters, collapses whitespace, and truncates each turn to 2,000 characters. Conversations with fewer than two usable turns are dropped.
- Writes to a temporary file, flushing and fsyncing after each page. The default cap is 15 MB (--max-mb).
- Retries 429 and 5xx responses with backoff, skips bad pages, and gives up after six consecutive failed pages.
- On interruption it keeps partial output only if it has at least 500 conversations. It never deletes an existing data_phase2.txt. It always exits with status 0 so training is not broken by a network failure.

### Curated data

The four curated files under data/ cover Linux and git, Python and PyTorch, hardware, and reasoning and identity. They are plain text, with conversational content in the same User:/Pip: style. They are how the project adds domain facts and persona on top of Shakespeare.

## 6. Training pipeline (train.py)

### Hyperparameters

| Setting | Default | Flag |
|---|---|---|
| Total steps target | 3000 | --steps |
| Learning rate | 3e-3 | --lr |
| Batch size | 32 | --batch-size |
| Eval every | 500 steps | --eval-every |
| Eval iterations | 50 batches per split | --eval-iters |
| Seed | 1337 | --seed |
| Checkpoint path | out/tinypip0.pt | --ckpt |

### Optimizer and loop

- AdamW with weight decay 0.1.
- Gradients are clipped to a global norm of 1.0 before each step.
- Each step: sample a batch, forward, cross-entropy loss, zero gradients, backward, clip, step.
- Every eval interval, the model is put in eval mode and the mean loss over eval_iters random batches is reported for both train and validation splits.
- There is no learning-rate schedule. The rate is constant.

### Device selection

pick_device chooses CUDA if available, then Apple MPS, then CPU.

### Resume and vocabulary growth

- If the checkpoint file exists, train.py loads it and continues from the saved step. Running python train.py again is enough.
- --resume PATH resumes from a specific file. --resume with no path uses --ckpt and errors if the file is missing.
- --scratch ignores any checkpoint and starts from random weights.
- --steps is an absolute target. A checkpoint at step 3000 with --steps 6000 trains 3000 more steps.
- If the new data contains characters the checkpoint has never seen, the vocabulary grows. The token embedding (and tied head) is expanded: old rows are copied unchanged and new rows are initialized from N(0, 0.02). AdamW moment tensors are zero-padded to match. Resuming after adding data does not require --scratch.
- The data seed is offset by the start step (seed + start_step) so a resumed run does not replay the same batches.
- If the checkpoint is already at or past --steps, the script reports that and exits without training.

### Checkpoint format

out/tinypip0.pt is a dictionary with these keys:

- model: state_dict of weights
- config: the Config fields as a dictionary
- itos: the id-to-character map
- optimizer: AdamW state
- step: the step reached

Saves are atomic. The file is written to a .tmp path and then renamed with os.replace, so an interrupted save cannot corrupt an existing checkpoint. Checkpoints from before resume support lack optimizer state and step, so they load but start training at step 0 with fresh optimizer moments.

### Launchers

- python train.py runs training directly.
- bash train.sh [flags] installs requirements.txt if torch or numpy is missing, prints the device, and runs train.py with the flags.
- bash train.sh --huge-data [--max-mb N] first runs fetch_data.py, then trains on input.txt plus data_phase2.txt.

## 7. Generation and chat (generate.py)

- Loads the checkpoint with weights_only=True and rebuilds the model from its stored config.
- One-shot mode: python generate.py --ckpt out/tinypip0.pt --prompt "ROMEO:" --tokens 300 --temperature 0.8 --top-k 40. The prompt must use only characters in the vocabulary.
- Chat mode (--chat): keeps a running transcript in the form User: ...\nPip: and generates a reply. The reply is cut at the first newline, because training data puts each Pip reply on one line. /reset clears history and /quit exits.
- --seed fixes the sampling seed for repeatable output.

## 8. Browser export and web chat

### export_web.py

- Reads out/tinypip0.pt on the CPU.
- Writes web/model/weights.bin: every tensor except lm_head.weight (which is tied and therefore redundant), converted to little-endian float32 and concatenated in sorted name order.
- Writes web/model/manifest.json containing the config, the itos vocabulary, the step, and for each tensor its name, shape, offset (in floats), and length. The browser uses the manifest to slice weights.bin into tensors.

With the default config the weights are about 111k floats, roughly 445 KB.

### web/

web/index.html and web/app.js form the browser chat page, which loads web/model/manifest.json and weights.bin. The web/README.md describes that page. The internals of app.js were not reviewed for this document, so details of its inference code are not described here.

### Root visualizers

- index.html is a zero-build page with a parameter calculator and an architecture view. It needs no server.
- visualizer.html is a second visualizer in the repository.

## 9. Project status and sizing notes

- The repository includes a checkpoint at out/tinypip0.pt and a zipped copy at out/tinypip0-checkpoint.zip.
- The README states the default config yields about 111k parameters. Scaling is steep: at dim 128 each layer is about 197k parameters, so 4 layers is about 0.8M and 6 layers about 1.2M. Change Config in model.py to scale.
- Training takes a few minutes on CPU and is faster on GPU.
- License: not yet specified.

## 10. Quick reference

```
pip install -r requirements.txt
python model.py                                      # print parameter count
python train.py                                      # train or resume
python train.py --steps 6000                         # extend a run
python train.py --scratch                            # ignore checkpoint
bash train.sh --huge-data --max-mb 15                # add SmolTalk data, then train
python generate.py --ckpt out/tinypip0.pt --chat     # chat in terminal
python export_web.py                                 # write web/model for the browser
```

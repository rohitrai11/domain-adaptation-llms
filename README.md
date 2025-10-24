# PubMed LLM Pretraining (Multi‑GPU, PyTorch DDP)

A minimal, end‑to‑end workflow to **download PubMed abstracts** and **pretrain a causal LLM** on **multiple GPUs** using PyTorch’s Distributed Data Parallel (DDP) via `torchrun`.

- `pubMedDataset.py` — robust, resumable downloader & parser for PubMed baseline XML; writes a JSONL corpus.
- `pretrain-multi-gpu.py` — chunked pretraining pipeline for a Hugging Face causal LM (default: `google/gemma-3-270m`) with clean DDP usage and token‑budgeted checkpoints.

---

## Quick Start

### 1) Create & activate the conda environment
```bash
conda create -n llm_pretrain python=3.10 -y
conda activate llm_pretrain
```

### 2) Install PyTorch (CUDA 12.6 wheels)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### 3) Install the Python libraries used by the scripts
```bash
pip install transformers sentencepiece datasets seqeval numpy scikit-learn evaluate bitsandbytes peft protobuf accelerate -U
```

> **Tip:** Verify CUDA is visible:
```bash
python -c "import torch; print('cuda?', torch.cuda.is_available(), 'gpu_count=', torch.cuda.device_count())"
```

---

## 1) Build the PubMed JSONL Corpus

The dataset script fetches PubMed **baseline** XML files from NCBI, parses the articles, and writes a **JSONL** file with fields like *pmid, title, abstract, authors, journal, publication_date, affiliations*.

```bash
python3 pubMedDataset.py
```

**Output:** `pubmed_abstract_corpus_detailed.jsonl` (resumable; re‑running will append from where it left off).  
**Notes**
- Uses a download‑then‑process approach with retries and exponential backoff.
- Saves only entries that contain **both title and abstract**.
- Intended for large runs (full baseline); you can trim the internal file list for a small test.

---

## 2) Pretrain on Multiple GPUs (DDP via torchrun)

The training script performs **token‑chunked** pretraining with Hugging Face `Trainer`, saving a checkpoint after each chunk boundary (measured in **unique tokens**). It auto‑detects torchrun environment variables for **rank/local_rank/world_size** and pins each process to its GPU.

### Fast smoke test (recommended first run)
```bash
torchrun --standalone --nproc_per_node=gpu pretrain-multi-gpu.py --fast_dev_run
```

### Typical real run
```bash
torchrun --standalone --nproc_per_node=gpu pretrain-multi-gpu.py --mixed_precision bf16
```
- Use `--mixed_precision bf16` on **Ampere+** GPUs (A100, RTX 30‑series, etc.).  
- On older GPUs, prefer `--mixed_precision fp16`.  
- Omit `--fast_dev_run` for full token budgets and epochs.

### Key defaults & artifacts
- **Model / Tokenizer:** `google/gemma-3-270m`
- **Input file:** `pubmed_abstract_corpus_detailed.jsonl`
- **Context length:** 1024 tokens
- **Batch sizes:** `per_device_train_batch_size=1`, `per_device_eval_batch_size=1` (adjust as memory allows)
- **Token budget:** processes data in **chunks** (`CHUNK_SIZE_TOKENS`) until `MAX_TOKENS` is reached
- **State & resume:** `training_state.json` tracks cumulative tokens; the next run starts at the next chunk
- **Checkpoints:** `gemma-3-270m-finetuned-pubmed/checkpoint-<END_TOKEN_IDX>-tokens`
- **Logs:** `gemma-3-270m-finetuned-pubmed/logs/`
- **Eval:** computes validation loss/perplexity (rank 0) each epoch; also prints a *pre‑training* perplexity on the test split of the corpus for the first chunk
- **DDP hygiene:** `ddp_find_unused_parameters=False`, per‑process GPU pinning, rank‑aware seeding

### CLI arguments
```txt
--fast_dev_run          Quick run: small data, 1 epoch, small token cap (smoke test)
--mixed_precision       one of {none, fp16, bf16} (auto-falls back if unsupported)
--n_train_samples       Override number of JSONL records to read for training
--n_test_samples        Override number of JSONL records to read for test split
--max_tokens            Override the global unique-token budget for training
```

### Data processing summary
- Reads JSONL and extracts `"abstract"`; prepends `"Abstract: "`; builds a `datasets.Dataset`.
- Splits into train/test; tokenizes; concatenates and chunks into 1024‑token blocks; `labels=input_ids` for causal LM.
- Caches tokenized **train** tokens (`train_input_ids.pt`) and the processed **test** dataset (`test_dataset.pt`) to avoid repeating work.

---

## Troubleshooting & Tips

- **NCCL / networking issues**: test single‑GPU first; ensure drivers match CUDA 12.6 wheels; try `--standalone` (already used) and verify that all ranks can see the GPUs.
- **OOM**: reduce batch size, or enable `--mixed_precision`; ensure no other processes occupy GPU memory.
- **bf16 not available**: the script will warn and continue in FP32; switch to `--mixed_precision fp16` if you need mixed precision on pre‑Ampere GPUs.
- **Long dataset run times**: the PubMed downloader is resumable; safe to stop and re‑run.
- **Selecting GPUs**: set `CUDA_VISIBLE_DEVICES=0,1,2,3` before `torchrun` to choose which GPUs to use.

---

## Project Structure

```
.
├── pubMedDataset.py              # Download + parse PubMed baseline → JSONL (resumable)
├── pretrain-multi-gpu.py         # DDP chunked pretraining on the JSONL corpus
└── pubmed_abstract_corpus_detailed.jsonl   # (generated) training data
```

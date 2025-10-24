import os
import json
import math
import time
import shutil
import argparse
import numpy as np
import torch

from itertools import chain
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    default_data_collator,
    TrainerCallback,
)

# --------------------
# torchrun helpers
# --------------------
def get_dist_info():
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size

def set_seed(seed: int):
    rank, _, _ = get_dist_info()
    import random
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)

def hw_supports_bf16(local_rank: int) -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(local_rank)
    return major >= 8  # Ampere+

# --------------------
# CLI
# --------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--mixed_precision",
    choices=["none", "fp16", "bf16"],
    default="none",
    help="Choose precision. Default 'none' (FP32). Use 'fp16' on pre-Ampere, 'bf16' on Ampere+.",
)
parser.add_argument("--fast_dev_run", action="store_true",
                    help="Quick run: small data, 1 epoch, small token cap for a fast multi-GPU smoke test.")
parser.add_argument("--n_train_samples", type=int, default=None,
                    help="Override number of JSONL records to read for the training split.")
parser.add_argument("--n_test_samples", type=int, default=None,
                    help="Override number of JSONL records to read for the test split.")
parser.add_argument("--max_tokens", type=int, default=None,
                    help="Override MAX_TOKENS cap (unique tokens, not multiplied by epochs).")
args = parser.parse_args()

rank, local_rank, world_size = get_dist_info()
is_main = (rank == 0)

# Pin each process to its GPU (torchrun-style)
if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# --------------------
# CONFIG (defaults)
# --------------------
MODEL_NAME = "google/gemma-3-270m"
TOKENIZER_NAME = "google/gemma-3-270m"
DATA_FILE = "pubmed_abstract_corpus_detailed.jsonl"

NUM_SAMPLES_TO_PROCESS = 4_800_000
TEST_SET_SIZE = 10_000
CONTEXT_LENGTH = 1024

STATE_FILE = "training_state.json"
CHUNK_SIZE_TOKENS = 50_000_000      # unique tokens per chunk
MAX_TOKENS = 1_000_000_000          # stop after this many unique tokens (not epochs)

OUTPUT_DIR = "gemma-3-270m-finetuned-pubmed"
TEMP_TRAINER_OUTPUT_DIR = f"{OUTPUT_DIR}/temp-training-run"
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
TOKEN_CACHE_FILE = "train_input_ids.pt"
TEST_CACHE_FILE = "test_dataset.pt"

# ---- Fast Dev mode for quick multi-GPU feel ----
if args.fast_dev_run:
    NUM_SAMPLES_TO_PROCESS = 20_000
    TEST_SET_SIZE = 1_000
    CHUNK_SIZE_TOKENS = 1_000_000
    MAX_TOKENS = 5_000_000
    NUM_TRAIN_EPOCHS = 1

# Optional CLI overrides
if args.n_train_samples is not None:
    NUM_SAMPLES_TO_PROCESS = args.n_train_samples
if args.n_test_samples is not None:
    TEST_SET_SIZE = args.n_test_samples
if args.max_tokens is not None:
    MAX_TOKENS = args.max_tokens

# --------------------
# Precision selection (reliable)
# --------------------
mp = args.mixed_precision
use_fp16 = (mp == "fp16")
use_bf16 = (mp == "bf16")

if use_bf16 and not hw_supports_bf16(local_rank):
    if is_main:
        print("⚠️  bf16 requested but this GPU doesn't support it. Falling back to FP32 ('none').")
    use_bf16 = False
    mp = "none"

if use_fp16 and not torch.cuda.is_available():
    if is_main:
        print("⚠️  fp16 requested but CUDA is not available. Falling back to FP32 ('none').")
    use_fp16 = False
    mp = "none"

if is_main:
    print(f"Mixed precision: {mp}")

# --------------------
# HELPERS
# --------------------
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=False)

def group_texts(examples):
    # Efficient concatenation
    concatenated = {k: list(chain.from_iterable(examples[k])) for k in examples}
    total_length = len(concatenated["input_ids"])
    total_length = (total_length // CONTEXT_LENGTH) * CONTEXT_LENGTH
    result = {
        k: [t[i:i + CONTEXT_LENGTH] for i in range(0, total_length, CONTEXT_LENGTH)]
        for k, t in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def load_and_preprocess_data(file_path, num_samples, test_size):
    msg = f"(first {num_samples} samples)" if num_samples else "(entire corpus)"
    if is_main:
        print(f"--- Step 1: Loading and Preprocessing Data {msg} ---")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: Data file not found at '{file_path}'.")

    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if num_samples and i >= num_samples:
                break
            records.append(json.loads(line))

    corpus = []
    for record in records:
        abstract = record.get('abstract', 'N/A') or 'N/A'
        corpus.append({"text": f"Abstract: {abstract}\n\n"})

    dataset = Dataset.from_list(corpus)
    # Keep resilient for small runs
    n = len(dataset)
    ts = min(test_size, max(1, n // 10)) if test_size else max(1, n // 10)
    split_dataset = dataset.train_test_split(test_size=ts, seed=42)
    if is_main:
        print(f"Dataset split into {len(split_dataset['train'])} train and {len(split_dataset['test'])} test samples.")
    return split_dataset

def evaluate_perplexity(model, dataset, device):
    """Safe PPL: filters non-finite losses and handles empty eval sets."""
    model.eval()
    if len(dataset) == 0:
        return None
    losses = []
    with torch.no_grad():
        for i in range(0, len(dataset), PER_DEVICE_EVAL_BATCH_SIZE):
            batch_data = dataset[i:i + PER_DEVICE_EVAL_BATCH_SIZE]
            if len(batch_data['input_ids']) == 0:
                continue
            examples = [{k: batch_data[k][j] for k in batch_data.keys()} for j in range(len(batch_data['input_ids']))]
            batch = default_data_collator(examples)
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            val = float(outputs.loss.detach().float().item())
            if math.isfinite(val):
                losses.append(val)
    if not losses:
        return None
    return math.exp(float(np.mean(losses)))

def format_token_count(num_tokens):
    if num_tokens >= 1_000_000_000:
        return f"{num_tokens / 1_000_000_000:.2f} Billion"
    if num_tokens >= 1_000_000:
        return f"{num_tokens / 1_000_000:.2f} Million"
    if num_tokens >= 1_000:
        return f"{num_tokens / 1_000:.2f} Thousand"
    return str(num_tokens)

class EpochTimerCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.epoch_start_time = None
        self.epoch_times = []

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.epoch_start_time is not None:
            epoch_time = time.time() - self.epoch_start_time
            if is_main:
                print(f"\nEpoch {int(state.epoch)} finished in {epoch_time:.2f} seconds.\n")
            self.epoch_times.append(epoch_time)
            self.epoch_start_time = None

# --------------------
# MAIN
# --------------------
if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if is_main:
        print(f"Using device: {device} | world_size={world_size} rank={rank} local_rank={local_rank}")

    # Load/Init training state (only rank 0 touches disk)
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
    else:
        state = {"total_tokens_processed": 0, "chunk_history": []}
    tokens_to_skip = state["total_tokens_processed"]

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Cache-first load (rank 0 saves; other ranks compute or load)
    if os.path.exists(TOKEN_CACHE_FILE) and os.path.exists(TEST_CACHE_FILE):
        if is_main:
            print("\n✅ Found cached training and test datasets. Skipping raw dataset load.")
        all_train_input_ids = torch.load(TOKEN_CACHE_FILE, weights_only=False)
        processed_test_dataset = torch.load(TEST_CACHE_FILE, weights_only=False)
    else:
        if is_main:
            print("\n⚠️ Cache missing. Loading and preprocessing raw dataset...")
        raw = load_and_preprocess_data(DATA_FILE, NUM_SAMPLES_TO_PROCESS, TEST_SET_SIZE)

        if os.path.exists(TEST_CACHE_FILE):
            processed_test_dataset = torch.load(TEST_CACHE_FILE, weights_only=False)
        else:
            tokenized_test_dataset = raw['test'].map(tokenize_function, batched=True, remove_columns=["text"])
            processed_test_dataset = tokenized_test_dataset.map(group_texts, batched=True)
            if is_main:
                torch.save(processed_test_dataset, TEST_CACHE_FILE)

        if os.path.exists(TOKEN_CACHE_FILE):
            all_train_input_ids = torch.load(TOKEN_CACHE_FILE, weights_only=False)
        else:
            tokenized_train_dataset = raw['train'].map(tokenize_function, batched=True, remove_columns=["text"])
            flat = list(chain.from_iterable(tokenized_train_dataset['input_ids']))
            all_train_input_ids = torch.tensor(flat, dtype=torch.int32)
            del flat
            if is_main:
                torch.save(all_train_input_ids, TOKEN_CACHE_FILE)

    # Count tokens available
    if isinstance(all_train_input_ids, torch.Tensor):
        total_available_tokens = int(all_train_input_ids.shape[0])
    else:
        total_available_tokens = len(all_train_input_ids)

    if is_main:
        print(f"\nTotal available training tokens: {format_token_count(total_available_tokens)}")

    # ---------------
    # CHUNKED TRAINING
    # ---------------
    while tokens_to_skip < min(total_available_tokens, MAX_TOKENS):
        start_token_idx = tokens_to_skip
        end_token_idx = min(start_token_idx + CHUNK_SIZE_TOKENS, total_available_tokens, MAX_TOKENS)

        # Slice current chunk
        if isinstance(all_train_input_ids, torch.Tensor):
            current_chunk_token_list = all_train_input_ids[start_token_idx:end_token_idx].tolist()
        else:
            current_chunk_token_list = all_train_input_ids[start_token_idx:end_token_idx]

        if len(current_chunk_token_list) < CONTEXT_LENGTH:
            if is_main:
                print("Not enough tokens left for a full sample. Exiting loop.")
            break

        total_length = (len(current_chunk_token_list) // CONTEXT_LENGTH) * CONTEXT_LENGTH
        input_ids = [current_chunk_token_list[i:i + CONTEXT_LENGTH] for i in range(0, total_length, CONTEXT_LENGTH)]
        attn = [[1] * CONTEXT_LENGTH for _ in range(len(input_ids))]
        chunk_dataset = Dataset.from_dict({
            "input_ids": input_ids,
            "labels": [ids.copy() for ids in input_ids],
            "attention_mask": attn
        })

        if is_main:
            print(f"\n--- Training Chunk {format_token_count(start_token_idx)} → {format_token_count(end_token_idx)} ---")
            print(f"Chunk size: {len(current_chunk_token_list)} tokens, {len(chunk_dataset)} samples.")

        # Determine checkpoint to load
        if start_token_idx == 0:
            model_to_load = MODEL_NAME
        else:
            model_to_load = f"{OUTPUT_DIR}/checkpoint-{start_token_idx}-tokens"
            if not os.path.exists(model_to_load):
                if is_main:
                    print("WARNING: Checkpoint not found. Falling back to base model.")
                model_to_load = MODEL_NAME

        # Choose model dtype from mp
        if mp == "bf16":
            load_dtype = torch.bfloat16
        elif mp == "fp16":
            load_dtype = torch.float16
        else:
            load_dtype = torch.float32

        # Load model (no device_map; DDP via torchrun/Trainer)
        model = AutoModelForCausalLM.from_pretrained(
            model_to_load,
            dtype=load_dtype
        )
        print(model)
        if model_to_load == MODEL_NAME:
            # Disable mean/cov init to suppress the message (safe fallback)
            try:
                model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
            except TypeError:
                model.resize_token_embeddings(len(tokenizer))

        # ---- Manual pre-train perplexity only on rank 0 ----
        if start_token_idx == 0 and is_main:
            # Ensure model & eval batches are on the same device to avoid mismatch
            model.to(device)
            eval_ds = processed_test_dataset
            if len(eval_ds) == 0:
                # Fallback: tiny slice from the current chunk
                take = min(64, len(chunk_dataset))
                eval_ds = chunk_dataset.select(range(take))
                print(f"ℹ️ Test set too small for 1024-token blocks. Using {take} samples from current chunk for pre-PPL.")
            p_before = evaluate_perplexity(model, eval_ds, device)
            if p_before is None:
                print("Perplexity BEFORE Fine-Tuning: (skipped - no finite losses)")
            else:
                print(f"Perplexity BEFORE Fine-Tuning: {p_before:.4f}")

        # Fresh temp dir only on main
        if is_main and os.path.exists(TEMP_TRAINER_OUTPUT_DIR):
            shutil.rmtree(TEMP_TRAINER_OUTPUT_DIR)

        training_args = TrainingArguments(
            output_dir=TEMP_TRAINER_OUTPUT_DIR,
            num_train_epochs=NUM_TRAIN_EPOCHS,
            per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            logging_dir=f"{OUTPUT_DIR}/logs",
            logging_steps=50 if args.fast_dev_run else 1000,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=2,
            dataloader_num_workers=8,
            dataloader_pin_memory=True,
            # Mixed precision flags (no TF32 here to avoid older-GPU crash)
            fp16=(mp == "fp16"),
            bf16=(mp == "bf16"),
            save_safetensors=True,
            seed=42,
            remove_unused_columns=False,
            report_to="none",
            ddp_find_unused_parameters=False,
        )

        epoch_timer = EpochTimerCallback()

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=chunk_dataset,
            eval_dataset=processed_test_dataset if len(processed_test_dataset) > 0 else None,
            data_collator=default_data_collator,
            callbacks=[epoch_timer]
        )

        # Train
        start_time = time.time()
        trainer.train()
        total_training_time = time.time() - start_time

        # Save checkpoint (HF/Accelerate ensures only rank 0 writes)
        final_checkpoint_dir = f"{OUTPUT_DIR}/checkpoint-{end_token_idx}-tokens"
        trainer.save_model(final_checkpoint_dir)
        if is_main:
            tokenizer.save_pretrained(final_checkpoint_dir)
            print(f"✅ Saved model at '{final_checkpoint_dir}'")

        # Evaluate
        eval_results = trainer.evaluate() if trainer.eval_dataset is not None else {}
        if eval_results:
            mean_loss_after = eval_results['eval_loss']
            perplexity_after = math.exp(mean_loss_after) if math.isfinite(mean_loss_after) else float("nan")
            if is_main:
                print(f"Validation Loss: {mean_loss_after:.4f} | Perplexity: {perplexity_after:.4f}")
        else:
            mean_loss_after = float("nan")
            perplexity_after = float("nan")
            if is_main:
                print("Validation skipped (no eval dataset).")

        # Update state (rank 0 only)
        if is_main:
            chunk_record = {
                "tokens_processed_in_chunk": len(current_chunk_token_list),
                "total_tokens_cumulative": end_token_idx,  # unique tokens boundary
                "final_validation_loss": mean_loss_after,
                "final_perplexity": perplexity_after,
                "total_training_time_seconds": total_training_time,
                "epoch_times_seconds": epoch_timer.epoch_times,
            }
            state["chunk_history"].append(chunk_record)
            state["total_tokens_processed"] = end_token_idx
            with open(STATE_FILE, 'w') as f:
                json.dump(state, f, indent=4)
            print(f"State updated. Processed {format_token_count(end_token_idx)} tokens total.\n")

        # Advance
        tokens_to_skip = end_token_idx

    if is_main:
        print("\n🎉 Training complete! Reached token limit or exhausted dataset.")


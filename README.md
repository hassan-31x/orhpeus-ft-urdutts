# Orpheus Urdu adaptation with UrduTTS

This is a self-contained Kaggle workflow based on `orpheus-ft-aslp/`. The first full run starts from **`unsloth/orpheus-3b-0.1-pretrained`**, with fresh LoRA weights, for **one epoch**. It does not inherit the ASLP adapter. Emotion conditioning is outside this run.

## Start on Kaggle

1. Push this entire folder to your GitHub repository. It can stay at `orpheus-ft-urdutts/` beside ASLP, or be the repository root.
2. Import **`kaggle_run.ipynb` from this folder** into Kaggle. Enable Internet and an NVIDIA GPU. No Drive files or Kaggle dataset uploads are required.
3. Add the **`HF_TOKEN`** Kaggle Secret and grant notebook access. Use a token with write access to the private model repository below. The script can create that repository if the token permits it; existing public repositories are rejected.
4. In the first code cell, check `REPO_URL`, `REPO_REF`, `HF_REPO_ID`, and `SESSION_HOURS`. The existing GitHub URL is filled in. The new default checkpoint repo is `hassan-31x/orpheus-urdutts-checkpoints`; change it if needed. Set the session budget below the runtime limit shown by your account.
5. Leave **`TEST=True`**, `TEST_TRAIN_CLIPS=20`, and `TEST_VALIDATION_CLIPS=10`. Choose Save Version → Save & Run All. You can change the test counts to 10 or another positive number.
6. Wait for **TEST PASSED**. It requires a completed smoke run, finite full-validation loss, a sealed checkpoint, final adapter, and successful remote snapshot. Generated speech is reported separately; listen to it and the reference clips. Passing this check does not establish Urdu speech quality.
7. Set **`TEST=False`**, keep **`INIT_FROM_RUN=None`**, and rerun for the full initial epoch. The smoke run has a separate ID and dataset/token cache, so its weights cannot seed the full run accidentally.

`REPO_REF="main"` allows the first launch after you push. The notebook prints the resolved Git commit. **Pin REPO_REF to that commit for all resumes of that run.** Updating your GitHub checkout does not update an already-imported Kaggle notebook; import the new notebook when its cells change.

An audit without model training is available with `AUDIT_ONLY=True`. With `TEST=True`, it inspects just the smoke subset. With `TEST=False`, it converts and audits the full train/validation partitions. Reference playback is in the status cell.

## Dataset format and conversion

Source: [kaab4321/UrduTTS](https://huggingface.co/datasets/kaab4321/UrduTTS), pinned to commit **`a012374328f140d2647e9138e6ceae85d4ee9f2c`**. The actual repository is a saved Hugging Face `DatasetDict`: `dataset_dict.json`, split `state.json` files, and Arrow IPC shards. Its card's `load_dataset("XKaab/UrduTTS")` example references a different ID and does not describe this repository's storage layout. The loader reads the requested repository directly.

| Released field/split | This workflow |
|---|---|
| `audio.bytes` | Decode embedded WAV; creator-local `audio.path` is ignored |
| `transcription` | Native Urdu text for Orpheus prompts |
| `roman`, `phonemes` | Unused |
| `train` | Training |
| `eval` | Validation and monitoring prompts |
| `test` | Not downloaded by default; never trained or evaluated |
| Speaker identity | None: release contains no speaker labels |

The card reports 40,511 train / 11,575 eval / 5,787 test utterances, approximately 64.21 / 18.41 / 9.24 hours. These are publisher figures; the pipeline measures actual converted and retained hours. It preserves the released splits instead of creating a new random split.

`urdutts_data.py` streams one Arrow record batch at a time over HTTPS. It converts native audio to **24,000 Hz mono PCM16 WAV** using channel averaging and `scipy.signal.resample_poly`. It does not trim, split, normalize loudness, or truncate speech. Arrow batches can approach 500 MB even when only 20 rows are requested; smoke runs avoid subsequent shards but are not byte-level random access to individual audio examples.

Each converted shard receives a completion record with per-file SHA-256 hashes. Interrupted local preparation reuses completed shards, and corrupt converted files rebuild their shard. No raw Arrow snapshot is retained on disk. Derived CSVs feed the existing ASLP audit, SNAC encoder, Trainer and storage workflow. Smoke mode uses the first N source rows of each requested partition; it is a pipeline check, not a representative quality benchmark.

`dataset_source.json` records the source commit, conversion implementation/packages, native sample rates, converted hours, and conversion exclusions. `dataset_report.json`, `data_errors.json`, `encoding_errors.json`, and `effective_training_manifests.json` record later audit, context-length and GPU-memory exclusions. Conversion exclusions are separate from the downstream manifest-audit denominator. No discarded row is silently counted as trained.

The original ASLP-specific minimum-hours floor is disabled (`minimum_train_hours=0`), so 10/20-row tests work. As in ASLP, invalid rows are recorded and skipped by default, and retention thresholds issue warnings (`enforce_retention_limits=false`). Use `invalid_row_policy="strict"` or enable retention enforcement if you require hard rejection. The inherited `max_length=2048` can exclude long audio/text sequences; speech is never cropped against its complete transcript. Review retained hours before interpreting an epoch as coverage of the entire release.

The [dataset card](https://huggingface.co/datasets/kaab4321/UrduTTS/blob/main/README.md) explicitly says transcripts are **ASR-generated**, four speakers are male, and speaker IDs are absent. The release is not guaranteed to fix transcription/alignment problems in ASLP. Its license is **CC BY-NC 4.0**; preserve attribution for this noncommercial course project. Existing emotion labels are not available.

## Checkpoints and continuation

Artifacts live in a private **model** repository at:

```text
runs/<run_id>/
  preparation/          # config, source, data evidence, environment
  cache/<fingerprint>/  # resumable SNAC-token Parquet chunks
  audit/                # data audit and source conversion reports
  monitoring/           # logs, metrics, samples, heartbeat
  snapshots/...         # complete checkpoint files and experiment evidence
  latest.json           # atomic pointer to a verified snapshot
```

Each Trainer checkpoint includes LoRA weights, tokenizer, optimizer, scheduler, RNG, Trainer state, run identity and file checksums. A synchronous Hub commit publishes its files and `latest.json` together. Saving starts at optimizer step 1, continues every 250 steps by default (every 2 in smoke mode), and occurs at the final step and planned session stop. The final adapter is exported before optional whole-validation evaluation.

**Interrupted epoch:** rerun the notebook with the same run ID, code commit, settings and dependency versions. It restores the most recent successfully uploaded checkpoint and continues that epoch, including optimizer and schedule. A fresh Kaggle runtime downloads/converts source audio again to audit its identity; verified remote SNAC caches avoid encoding it again. Progress after the last successful backup can be lost if the runtime disappears.

**Additional epoch after completion:** this matches the ASLP notebook's epoch-2 approach. For example:

```python
TEST = False
RUN_ID_OVERRIDE = "urdutts-r32-lr1e4-seed3407-epoch2"
INIT_FROM_RUN = "urdutts-r32-lr1e4-seed3407-epoch1"
LEARNING_RATE = None  # Keep 1e-4; choose a lower rate only as a deliberate new-epoch experiment.
```

The new run loads the completed epoch's final LoRA weights and compatible token cache. **It starts a fresh optimizer/LR schedule**, just like ASLP; this is not extension of the original cosine schedule to two epochs. Interrupted epoch 2 resumes its own full checkpoint. Smoke or unfinished runs cannot seed another epoch. Keep rank, alpha, base model and data settings consistent. Changing `epochs` on an existing run is rejected; each named run represents one epoch.

Failed remote uploads are queued locally and retried. **Check `backup_status.json` and `remote_snapshot_current` before discarding the runtime.** A locally complete run with pending uploads is not yet remotely durable. Rerunning an already-completed run retries its backup without repeating training.

Disk checks preserve the newest verified checkpoint and prune older verified ones under pressure. Full retained train+validation PCM16 audio is about 13.3 GiB using the publisher's hours, plus WAV headers, base weights, encoded caches, optimizer snapshots and upload evidence. Source Arrow shards are streamed rather than cached. Actual free-space requirements depend on adapter serialization and retained checkpoints; inspect notebook disk diagnostics. Two T4 GPUs do not combine VRAM; the script uses device 0.

## Files and validation

| File | Purpose |
|---|---|
| `kaggle_run.ipynb` | Clone, install, test, configure, train, monitor and listen |
| `configs/urdutts.json` | One-epoch base-model defaults and pinned source dataset |
| `urdutts_data.py` | Saved-Arrow reader and resumable 24 kHz conversion |
| `finetune_urdutts.py` | Inherited data audit, encoding, training and resume control |
| `orpheus_utils.py` | Orpheus tokens, checkpoint sealing, private Hub storage |
| `pipeline_recovery.py` | Upload queue, memory/disk recovery and heartbeat |
| `setup_kaggle.py`, `check_environment.py` | Isolated environment preserving Kaggle CUDA wheels |
| `check_hub.py` | Private repository write/read probe |
| `synthesize.py` | Generated speech from final/intermediate adapters |
| `tests/` | CPU regression and mocked training/storage recovery tests |
| `WORKFLOW_REVIEW.md` | ASLP code trace, retained design and validation boundaries |

Run CPU tests from this directory after installing CPU requirements (`numpy pandas pyarrow soundfile scipy datasets==3.6.0 huggingface_hub packaging requests`):

```bash
python -m unittest discover -s tests -q
```

The notebook runs these tests after setup. Local validation also reads **20 actual train and 10 actual eval examples from the pinned public repository**, converts their embedded WAVs, and passes the inherited data audit. Full CUDA/Unsloth training and authenticated Hub writes require the Kaggle smoke run; they have not been executed on this macOS machine. No promise of speech quality follows from CPU tests.

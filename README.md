# Orpheus Urdu adaptation with UrduTTS

This is a self-contained Kaggle workflow based on `orpheus-ft-aslp/`. The first full run starts from **`unsloth/orpheus-3b-0.1-pretrained`**, with fresh LoRA weights, for **one epoch**. It does not inherit the ASLP adapter. Emotion conditioning is outside this run.

## Start on Kaggle

1. Push this folder's contents to [hassan-31x/orhpeus-ft-urdutts](https://github.com/hassan-31x/orhpeus-ft-urdutts). The notebook is configured for scripts at that repository's root (`PROJECT_FOLDER="."`).
2. Import **`kaggle_run.ipynb` from this folder** into Kaggle. Enable Internet and an NVIDIA GPU. No Drive files or Kaggle dataset uploads are required.
3. Add the **`HF_TOKEN`** Kaggle Secret and grant notebook access. Use a token with write access to the private model repository below. The script can create that repository if the token permits it; existing public repositories are rejected.
4. In the first code cell, check `REPO_REF`, `HF_SECRET_NAME`, and `SESSION_HOURS`. GitHub is set to `hassan-31x/orhpeus-ft-urdutts`; checkpoints use [hassan-31x/orpheus-urdutts-checkpoints](https://huggingface.co/hassan-31x/orpheus-urdutts-checkpoints). Set the session budget below the runtime limit shown by your account.
5. Leave **`TEST=True`**, `TEST_TRAIN_CLIPS=20`, and `TEST_VALIDATION_CLIPS=10`. Choose Save Version → Save & Run All. You can change the test counts to 10 or another positive number.
6. Wait for **TEST PASSED**. It requires a completed smoke run, finite full-validation loss, a sealed checkpoint, final adapter, and successful remote snapshot. Generated speech is reported separately; listen to it and the reference clips. Passing this check does not establish Urdu speech quality.
7. Set **`TEST=False`**, keep **`INIT_FROM_RUN=None`**, and rerun for the full initial epoch. The smoke run has a separate ID and dataset/token cache, so its weights cannot seed the full run accidentally.

`REPO_REF="main"` allows the first launch after you push. The notebook prints the resolved Git commit. **Pin REPO_REF to that commit for all resumes of that run.** Updating your GitHub checkout does not update an already-imported Kaggle notebook; import the new notebook when its cells change.

An audit without model training is available with `AUDIT_ONLY=True`. With `TEST=True`, it inspects just the smoke subset. With `TEST=False`, it converts and audits the full train/validation partitions. Reference playback is in the status cell.

### If the first cell cannot load HF_TOKEN

This happens before dependency installation, data preparation or training. The debugger/frozen-module warnings and notebook HTML conversion warnings are unrelated to the failure.

Open the **new UrduTTS notebook in Edit mode → Add-ons → Secrets**. Select or create the secret labelled **`HF_TOKEN`**, and **enable its access for this notebook**. Access granted to the previous ASLP notebook does not configure the new notebook. If your existing secret uses another label, set `HF_SECRET_NAME` to that exact label in the first code cell. Its value must be a Hugging Face token with write access to the checkpoint repository. Do not put the value in code or notebook output.

Then **Save Version → Save & Run All** again. Updating GitHub alone does not replace the first cell of the imported notebook: re-import the updated `kaggle_run.ipynb` or replace that cell as well. No training progress was created by a failure at secret loading.

The updated loader uses an existing `HF_TOKEN` environment variable first, then the legacy `HUGGING_FACE_HUB_TOKEN` environment name, and otherwise Kaggle Secrets with `HF_SECRET_NAME`. Only the credential source is printed. Missing/disabled/empty secrets still stop with actionable instructions; code cannot grant Kaggle secret access on your behalf. The private-repository upload/readback preflight still verifies the actual token permissions.

## Listen to saved checkpoints

Import **`checkpoint_demo.ipynb`** into Kaggle with Internet and an NVIDIA GPU, and enable the `HF_TOKEN` secret for that notebook. Read access to the private model repository is sufficient. This demo uses the existing inference environment and synthesis code without preparing data or starting training.

Run cells 1–3 to list adapters in the epoch-1 run, then set `SELECT` in cell 4 to one or several printed indices. Set `TEXT` to your Urdu sentence and run cell 5 for labelled audio players. The default selects the latest numbered checkpoint; set `RUN_ID=None` to list other epochs too. Rerun cells 4–5 to compare different selections, or only cell 5 after changing the sentence. Every selection uses the same seed and generation settings, and models load sequentially in separate processes.

The notebook pins the Hub commit and downloads only inference files from unpacked snapshot folders. WAV files, decoding metadata, and `comparison.json` are saved in `/kaggle/working/urdu-checkpoint-demo/audio/`. Legacy archive snapshots need separate restoration. GPU synthesis and listening quality must be checked in the Kaggle runtime.

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

Disk checks preserve the newest verified checkpoint and prune older verified ones under pressure. Full retained train+validation PCM16 audio is about 13.3 GiB using the publisher's hours during preparation, plus WAV headers, base weights and encoded caches. **After complete SNAC encoding, the workflow verifies all token chunks and their retained-row coverage, then reclaims registered, unchanged converted WAVs before loading the training model.** It keeps the first three references per split and all seeded monitoring references. `audio_storage.json` records reclaimed bytes and protected clips. Parquet/Arrow tokens, conversion registries, reports, base weights and full optimizer checkpoints remain. Preparation-only and audit modes retain their audio. A later startup regenerates missing WAVs from the pinned public source for auditing; compatible SNAC tokens are reused.

Actual free-space requirements depend on adapter serialization and retained checkpoints; inspect notebook disk diagnostics. Two T4 GPUs do not combine VRAM; the script uses device 0.

### Recovering the reported 4.83 GiB / 5.07 GiB disk failure

The supplied log shows encoding finished but training stopped at its initial checkpoint-space check, **before any optimizer step**. The 2 context-overflow exclusions are separate from this disk error. The checkpoint estimate includes frozen embedding weights that PEFT may save, so the 512 MiB reserve and full-state checkpoint format have not been reduced.

1. Push the updated code, including **`audio_storage.py`**, and update/re-import the notebook.
2. Keep **`TEST=False`**, the same full-run ID, the same training settings, and `INIT_FROM_RUN=None`. Select the newly pushed commit with `REPO_REF` (or `main` to fetch it initially). Pin that new commit for subsequent resumes.
3. Rerun the notebook. Keep the existing runtime/files if available: converted audio and verified local token chunks are reused. In a fresh runtime audio is downloaded/converted again; successfully uploaded compatible token chunks are restored from Hugging Face. No manual deletion is needed.
4. Look for **`Reclaimed ... GiB from ... converted WAVs`** before model training, then check `disk_budget.json` and the first checkpoint's remote backup status.

For this run, reclaimable converted audio is roughly 13 GiB, comfortably more than the reported 240 MiB shortfall. Exact reclaimed bytes/free space are measured, not assumed. The same disk reserve is enforced again before every save. This update changes training source identity; it is suitable for this preparation-only failure. A run that already has a verified optimizer checkpoint must use its recorded code revision to resume.

## Files and validation

| File | Purpose |
|---|---|
| `kaggle_run.ipynb` | Clone, install, test, configure, train, monitor and listen |
| `checkpoint_demo.ipynb` | Load saved Hub checkpoints and compare Urdu speech for the same text |
| `configs/urdutts.json` | One-epoch base-model defaults and pinned source dataset |
| `urdutts_data.py` | Saved-Arrow reader and resumable 24 kHz conversion |
| `audio_storage.py` | Verified post-encoding WAV reclamation, preserving references and token caches |
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

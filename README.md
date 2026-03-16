# OpenNMT Course Project: English-to-Spanish Transformer MT

Chinese version: [README_CN.md](README_CN.md)

This repository is a course-project template for local machine translation with OpenNMT.
It provides a full pipeline: preprocessing, Transformer finetuning, inference, BLEU evaluation, benchmarking, and a Streamlit demo.

## 1. What This Project Includes

1. Baseline translation with an available OpenNMT model.
2. Small-scale domain finetuning using English-Spanish parallel data.
3. Optional glossary term protection for consistent terminology.
4. Benchmark experiments (model/batch/beam/cache/glossary).
5. Local Streamlit UI for interactive testing.

## 2. Project Structure

```text
<project-root>
├─ OpenNMT-py
├─ models
├─ data
│  ├─ raw_domain
│  │  └─ domain_pairs.csv
│  ├─ processed
│  │  ├─ train.src
│  │  ├─ train.tgt
│  │  ├─ valid.src
│  │  ├─ valid.tgt
│  │  ├─ test.src
│  │  ├─ test.tgt
│  │  └─ finetune_config.yaml
│  ├─ glossary.csv
│  └─ input.txt
├─ outputs
├─ cache
├─ tmp
├─ app.py
├─ translator_backend.py
├─ preprocess_domain_data.py
├─ run_finetune.ps1
├─ run_translate.ps1
├─ run_course_pipeline.ps1
├─ evaluate_bleu.py
├─ benchmark.py
├─ plot_benchmark.py
├─ summarize_results.py
├─ demo_streamlit.py
└─ README.md
```

All scripts resolve paths from the repository root automatically.
If needed, set `OPENNMT_PROJECT_ROOT` to override the root path.

## 3. Data Format (English-Spanish)

Prepare `data/raw_domain/domain_pairs.csv` with at least these columns:

```csv
source,target
The tea is fresh.,El te esta fresco.
This model runs locally.,Este modelo se ejecuta localmente.
```

Default expected columns are `source` and `target`.

## 4. Environment Setup

1. Place `OpenNMT-py` under the project root.
2. Create a Python environment (`venv` or `.venv`) in the project root.
3. Install dependencies:

```powershell
pip install sacrebleu matplotlib streamlit
```

Or install from the pinned dependency file:

```powershell
python -m pip install -r requirements.txt
```

## 5. End-to-End Workflow

### Step A. Preprocess parallel data

Optional but recommended: clean and normalize the corpus first.

```powershell
python clean_parallel_corpus.py --input data/raw_domain/domain_pairs.csv --output data/raw_domain/domain_pairs.csv
```

```powershell
python preprocess_domain_data.py --source-col source --target-col target
```

Optional split control:

```powershell
python preprocess_domain_data.py --train-ratio 0.7 --valid-ratio 0.15 --seed 42
```

### Step B. Finetune Transformer model

```powershell
powershell -ExecutionPolicy Bypass -File .\run_finetune.ps1
```

Optional training preset:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_finetune.ps1 -Preset quick
powershell -ExecutionPolicy Bypass -File .\run_finetune.ps1 -Preset report
```

Optional manual overrides:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_finetune.ps1 -TrainSteps 4000 -BatchSize 16
```

### Step C. Generate test predictions

```powershell
powershell -ExecutionPolicy Bypass -File .\run_translate.ps1
```

### Step D. Evaluate BLEU

```powershell
python evaluate_bleu.py
```

### Step E. Run benchmark and plot

```powershell
python benchmark.py
python plot_benchmark.py
```

### Step F. Run local demo

```powershell
streamlit run demo_streamlit.py
```

### Step G. One-command course pipeline

This runs finetune -> test translation -> BLEU -> benchmark -> plots -> summary report:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_course_pipeline.ps1 -Preset quick
```

For final report experiments:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_course_pipeline.ps1 -Preset report
```

Pipeline outputs include:

1. `outputs/course_report_summary.md` (report draft text)
2. `outputs/course_report_metrics.csv` (table-ready metrics with percentage improvements)

## 6. Notes for Course Report

1. Report baseline vs finetuned BLEU on the same test set.
2. Include benchmark comparison of batch size and beam size.
3. Explain glossary effect on terminology consistency.
4. Mention cache hit behavior for repeated sentences.
5. If no custom checkpoint exists, scripts may fall back to test baseline models.
6. Use `outputs/course_report_summary.md` as a draft section for your written report.
7. Use `outputs/course_report_metrics.csv` directly as a results table source.

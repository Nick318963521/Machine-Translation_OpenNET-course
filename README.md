# OpenNMT Course Project: Baseline, Finetuning, Glossary, Benchmark, and Local Demo

## 1. Project Title

OpenNMT Pretrained Baseline + Small-Scale Domain Finetuning + Glossary Optimization + Benchmark + Local Demo

## 2. Motivation

This project builds a local document translation workflow on top of OpenNMT. The goal is to start from a baseline pretrained model, add a small-scale domain adaptation step, and improve usability with glossary protection, caching, benchmarking, and a simple local demo.

## 3. System Overview

The system contains five parts:

1. Baseline translation with a pretrained OpenNMT model.
2. Small-scale domain finetuning on a compact parallel dataset.
3. Glossary-based term protection and restoration.
4. Benchmark experiments for speed and configuration comparison.
5. A local Streamlit demo for interactive translation.

## 4. Folder Structure

```text
D:\opennmt_project
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
├─ glossary.py
├─ cache_utils.py
├─ document_processor.py
├─ benchmark.py
├─ plot_benchmark.py
├─ preprocess_domain_data.py
├─ run_finetune.ps1
├─ run_translate.ps1
├─ evaluate_bleu.py
├─ demo_streamlit.py
└─ README.md
```

## 5. Setup Instructions

1. Clone or place `OpenNMT-py` under `D:\opennmt_project\OpenNMT-py`.
2. Prepare a Python environment on Windows.
3. Install required packages:

```powershell
pip install sacrebleu matplotlib streamlit
```

4. Confirm OpenNMT works:

```powershell
onmt_translate -h
```

## 6. How to Run Baseline Translation

Put your input text into `D:\opennmt_project\data\input.txt`, then run:

```powershell
python app.py
```

The script saves translation results to `D:\opennmt_project\outputs\translated.txt`.

## 7. How to Run Finetuning

1. Put small domain parallel data into `D:\opennmt_project\data\raw_domain\domain_pairs.csv`.
2. Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_finetune.ps1
```

3. Check `D:\opennmt_project\models` for `finetuned_model_step_*.pt`.

## 8. How to Run Benchmark

```powershell
python benchmark.py
python plot_benchmark.py
```

The CSV file is saved to `D:\opennmt_project\outputs\benchmark_results.csv`.

## 9. How to Run Demo

```powershell
streamlit run demo_streamlit.py
```

The page provides model selection, glossary switch, batch size, beam size, and output saving.

## 10. Expected Results

- The finetuned model should perform better on the selected domain than the baseline.
- Glossary protection should keep important terminology consistent.
- Larger batch sizes should usually improve throughput.
- Larger beam sizes may improve output quality but increase latency.
- Cache should reduce repeated inference cost for duplicated sentences.

## Notes

- The current code falls back to the OpenNMT test model if a custom baseline or finetuned model is not yet available.
- For the final report, replace the fallback model with your real baseline and finetuned checkpoints in `D:\opennmt_project\models`.

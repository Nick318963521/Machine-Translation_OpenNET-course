# OpenNMT 课程项目：英译西 Transformer 机器翻译

本仓库是一个基于 OpenNMT 的课程项目模板，覆盖完整实验链路：
数据预处理、Transformer 微调、推理、BLEU 评测、基准测试与可视化、以及本地 Streamlit 演示。

## 1. 项目包含内容

1. 使用可用 OpenNMT 模型进行基线翻译。
2. 使用英-西平行语料进行小规模领域微调。
3. 可选术语表保护，提升术语一致性。
4. 基准测试（模型、batch、beam、缓存、术语开关）。
5. 本地 Streamlit 交互界面。

## 2. 项目结构

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
├─ README.md
└─ README_CN.md
```

默认所有脚本都以仓库根目录为路径基准。
如果需要自定义根目录，可设置环境变量 OPENNMT_PROJECT_ROOT。

## 3. 数据格式（英-西）

准备 data/raw_domain/domain_pairs.csv，至少包含以下列：

```csv
source,target
The tea is fresh.,El te esta fresco.
This model runs locally.,Este modelo se ejecuta localmente.
```

默认列名为 source（源语言）与 target（目标语言）。

## 4. 环境准备

1. 将 OpenNMT-py 放在仓库根目录下。
2. 在仓库根目录创建 Python 虚拟环境（venv 或 .venv）。
3. 安装依赖：

```powershell
pip install sacrebleu matplotlib streamlit
```

或使用固定依赖清单安装：

```powershell
python -m pip install -r requirements.txt
```

## 5. 端到端流程

### Step A. 预处理平行数据

可选但推荐：先做语料清洗和编码归一化。

```powershell
python clean_parallel_corpus.py --input data/raw_domain/domain_pairs.csv --output data/raw_domain/domain_pairs.csv
```

```powershell
python preprocess_domain_data.py --source-col source --target-col target
```

可选切分参数：

```powershell
python preprocess_domain_data.py --train-ratio 0.7 --valid-ratio 0.15 --seed 42
```

### Step B. 微调 Transformer

```powershell
powershell -ExecutionPolicy Bypass -File .\run_finetune.ps1
```

可选预设：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_finetune.ps1 -Preset quick
powershell -ExecutionPolicy Bypass -File .\run_finetune.ps1 -Preset report
```

可选手动覆盖参数：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_finetune.ps1 -TrainSteps 4000 -BatchSize 16
```

### Step C. 生成测试集预测

```powershell
powershell -ExecutionPolicy Bypass -File .\run_translate.ps1
```

### Step D. BLEU 评测

```powershell
python evaluate_bleu.py
```

### Step E. 运行基准测试并画图

```powershell
python benchmark.py
python plot_benchmark.py
```

### Step F. 启动本地演示

```powershell
streamlit run demo_streamlit.py
```

### Step G. 一键课程实验流水线

该命令会自动执行：微调 -> 测试集翻译 -> BLEU -> benchmark -> 绘图 -> 汇总报告：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_course_pipeline.ps1 -Preset quick
```

最终报告实验建议使用：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_course_pipeline.ps1 -Preset report
```

流水线输出：

1. outputs/course_report_summary.md（课程报告文字草稿）
2. outputs/course_report_metrics.csv（可直接做表格的指标数据，含提升百分比）

## 6. 课程报告建议

1. 在同一测试集上报告 baseline 与 finetuned 的 BLEU。
2. 给出 batch size 与 beam size 的速度对比分析。
3. 说明术语表对术语一致性的作用。
4. 说明缓存对重复句子吞吐的提升。
5. 若没有自定义 checkpoint，脚本会回退到测试基线模型。
6. 可直接使用 outputs/course_report_summary.md 作为报告初稿参考。
7. 可直接使用 outputs/course_report_metrics.csv 作为结果表格来源。

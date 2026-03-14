$ErrorActionPreference = "Stop"

$ProjectRoot = "D:\opennmt_project"
$PythonCandidates = @(
    (Join-Path $ProjectRoot "venv\Scripts\python.exe"),
    (Join-Path $ProjectRoot ".venv\Scripts\python.exe")
)

$PythonExe = $null
foreach ($Candidate in $PythonCandidates) {
    if (Test-Path $Candidate) {
        & $Candidate -c "import onmt, pyonmttok; print('ok')" | Out-Null
        if ($LASTEXITCODE -eq 0) {
            $PythonExe = $Candidate
            break
        }
    }
}

if (-not $PythonExe) {
    throw "No usable Python environment found. Expected one that can import onmt and pyonmttok."
}

$OpenNMTDir = Join-Path $ProjectRoot "OpenNMT-py"
$TrainDataDir = Join-Path $ProjectRoot "data\processed"
$ConfigPath = Join-Path $ProjectRoot "data\processed\finetune_config.yaml"
$ModelsDir = Join-Path $ProjectRoot "models"
$TrainScript = Join-Path $OpenNMTDir "train.py"
$BaselineModel = Join-Path $ModelsDir "baseline_pretrained.pt"
$TrainSrc = Join-Path $TrainDataDir "train.src"
$TrainTgt = Join-Path $TrainDataDir "train.tgt"
$SrcVocab = Join-Path $TrainDataDir "vocab.src"
$TgtVocab = Join-Path $TrainDataDir "vocab.tgt"

New-Item -ItemType Directory -Force -Path $TrainDataDir | Out-Null
New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null

Write-Host "Step 1: preprocess domain csv into train/valid/test"
& $PythonExe (Join-Path $ProjectRoot "preprocess_domain_data.py")
if ($LASTEXITCODE -ne 0) { throw "preprocess_domain_data.py failed." }

$TrainFromLine = ""
if (Test-Path $BaselineModel) {
    $TrainFromLine = "train_from: D:/opennmt_project/models/baseline_pretrained.pt"
    Write-Host "Baseline model found. Finetuning will start from baseline_pretrained.pt"
} else {
    Write-Host "No baseline_pretrained.pt found. The script will train a small domain model from scratch."
}

$Yaml = @"
save_data: D:/opennmt_project/data/processed/domain_data
src_vocab: D:/opennmt_project/data/processed/vocab.src
tgt_vocab: D:/opennmt_project/data/processed/vocab.tgt

data:
  corpus_1:
    path_src: D:/opennmt_project/data/processed/train.src
    path_tgt: D:/opennmt_project/data/processed/train.tgt
  valid:
    path_src: D:/opennmt_project/data/processed/valid.src
    path_tgt: D:/opennmt_project/data/processed/valid.tgt

save_model: D:/opennmt_project/models/finetuned_model
save_checkpoint_steps: 50
keep_checkpoint: 5
train_steps: 200
valid_steps: 50
report_every: 20
$TrainFromLine

src_seq_length: 80
tgt_seq_length: 80
src_vocab_size: 5000
tgt_vocab_size: 5000

encoder_type: transformer
decoder_type: transformer
position_encoding: true
enc_layers: 2
dec_layers: 2
heads: 2
hidden_size: 128
word_vec_size: 128
transformer_ff: 256
dropout: 0.1

optim: adam
learning_rate: 1.0
warmup_steps: 500
decay_method: noam
label_smoothing: 0.1
param_init_glorot: true

batch_size: 4
valid_batch_size: 4
batch_type: sents
num_workers: 0
world_size: 1
gpu_ranks: []
n_sample: 0
"@

Set-Content -Path $ConfigPath -Value $Yaml -Encoding UTF8

Write-Host "Step 2: generate plain-text vocab files"
& $PythonExe -c "from collections import Counter; from pathlib import Path; import sys; src=Path(sys.argv[1]); tgt=Path(sys.argv[2]); src_out=Path(sys.argv[3]); tgt_out=Path(sys.argv[4]); c1=Counter(); c2=Counter(); [c1.update(line.strip().split()) for line in src.open('r', encoding='utf-8') if line.strip()]; [c2.update(line.strip().split()) for line in tgt.open('r', encoding='utf-8') if line.strip()]; src_out.parent.mkdir(parents=True, exist_ok=True); tgt_out.parent.mkdir(parents=True, exist_ok=True); f1=src_out.open('w', encoding='utf-8', newline='\n'); [f1.write(f'{tok}\t{cnt}\n') for tok, cnt in c1.most_common()]; f1.close(); f2=tgt_out.open('w', encoding='utf-8', newline='\n'); [f2.write(f'{tok}\t{cnt}\n') for tok, cnt in c2.most_common()]; f2.close()" $TrainSrc $TrainTgt $SrcVocab $TgtVocab
if ($LASTEXITCODE -ne 0) { throw "vocab generation failed." }

Write-Host "Step 3: train a compact finetuned model"
& $PythonExe $TrainScript -config $ConfigPath
if ($LASTEXITCODE -ne 0) { throw "train.py failed." }

Write-Host "Finetune finished. Check D:\opennmt_project\models for finetuned_model_step_*.pt"

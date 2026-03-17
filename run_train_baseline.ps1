param(
    [string]$ProjectRoot = $PSScriptRoot,
    [string]$SourceColumn = "source",
    [string]$TargetColumn = "target",
    [ValidateSet("quick", "report")]
    [string]$Preset = "quick",
    [Nullable[int]]$TrainSteps = $null,
    [Nullable[int]]$ValidSteps = $null,
    [Nullable[int]]$SaveCheckpointSteps = $null,
    [Nullable[int]]$BatchSize = $null,
    [Nullable[int]]$ValidBatchSize = $null
)

$ErrorActionPreference = "Stop"

function Convert-ToPosixPath {
    param([string]$PathValue)
    return ($PathValue -replace "\\", "/")
}

if ($Preset -eq "quick") {
    if (-not $TrainSteps.HasValue) { $TrainSteps = 800 }
    if (-not $ValidSteps.HasValue) { $ValidSteps = 60 }
    if (-not $SaveCheckpointSteps.HasValue) { $SaveCheckpointSteps = 200 }
    if (-not $BatchSize.HasValue) { $BatchSize = 8 }
    if (-not $ValidBatchSize.HasValue) { $ValidBatchSize = 8 }
} else {
    if (-not $TrainSteps.HasValue) { $TrainSteps = 3000 }
    if (-not $ValidSteps.HasValue) { $ValidSteps = 200 }
    if (-not $SaveCheckpointSteps.HasValue) { $SaveCheckpointSteps = 500 }
    if (-not $BatchSize.HasValue) { $BatchSize = 16 }
    if (-not $ValidBatchSize.HasValue) { $ValidBatchSize = 16 }
}

Write-Host "Baseline preset: $Preset"
Write-Host "train_steps=$TrainSteps valid_steps=$ValidSteps batch_size=$BatchSize"

$PythonCandidates = @(
    (Join-Path $ProjectRoot "venv\Scripts\python.exe"),
    (Join-Path $ProjectRoot ".venv\Scripts\python.exe"),
    "python"
)

$PythonExe = $null
foreach ($Candidate in $PythonCandidates) {
    try {
        & $Candidate -c "import onmt, pyonmttok; print('ok')" | Out-Null
        if ($LASTEXITCODE -eq 0) {
            $PythonExe = $Candidate
            break
        }
    } catch {
        continue
    }
}

if (-not $PythonExe) {
    throw "No usable Python environment found. Expected one that can import onmt and pyonmttok."
}

$OpenNMTDir = Join-Path $ProjectRoot "OpenNMT-py"
$TrainDataDir = Join-Path $ProjectRoot "data\processed"
$ConfigPath = Join-Path $TrainDataDir "baseline_config.yaml"
$ModelsDir = Join-Path $ProjectRoot "models"
$TrainScript = Join-Path $OpenNMTDir "train.py"
$TrainSrc = Join-Path $TrainDataDir "train.src"
$TrainTgt = Join-Path $TrainDataDir "train.tgt"
$ValidSrc = Join-Path $TrainDataDir "valid.src"
$ValidTgt = Join-Path $TrainDataDir "valid.tgt"
$SrcVocab = Join-Path $TrainDataDir "vocab.src"
$TgtVocab = Join-Path $TrainDataDir "vocab.tgt"
$BaselinePrefix = Join-Path $ModelsDir "baseline_pretrained"
$BaselineFinal = Join-Path $ModelsDir "baseline_pretrained.pt"

New-Item -ItemType Directory -Force -Path $TrainDataDir | Out-Null
New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null

Write-Host "Step 1: preprocess domain csv into train/valid/test"
& $PythonExe (Join-Path $ProjectRoot "preprocess_domain_data.py") --source-col $SourceColumn --target-col $TargetColumn
if ($LASTEXITCODE -ne 0) { throw "preprocess_domain_data.py failed." }

$SaveDataPath = Convert-ToPosixPath (Join-Path $TrainDataDir "domain_data")
$SrcVocabPosix = Convert-ToPosixPath $SrcVocab
$TgtVocabPosix = Convert-ToPosixPath $TgtVocab
$TrainSrcPosix = Convert-ToPosixPath $TrainSrc
$TrainTgtPosix = Convert-ToPosixPath $TrainTgt
$ValidSrcPosix = Convert-ToPosixPath $ValidSrc
$ValidTgtPosix = Convert-ToPosixPath $ValidTgt
$BaselinePrefixPosix = Convert-ToPosixPath $BaselinePrefix

$yamlLines = @(
    "save_data: $SaveDataPath",
    "src_vocab: $SrcVocabPosix",
    "tgt_vocab: $TgtVocabPosix",
    "",
    "data:",
    "  corpus_1:",
    "    path_src: $TrainSrcPosix",
    "    path_tgt: $TrainTgtPosix",
    "  valid:",
    "    path_src: $ValidSrcPosix",
    "    path_tgt: $ValidTgtPosix",
    "",
    "save_model: $BaselinePrefixPosix",
    "save_checkpoint_steps: $SaveCheckpointSteps",
    "keep_checkpoint: 6",
    "train_steps: $TrainSteps",
    "valid_steps: $ValidSteps",
    "report_every: 50",
    "",
    "src_seq_length: 120",
    "tgt_seq_length: 120",
    "src_vocab_size: 8000",
    "tgt_vocab_size: 8000",
    "",
    "encoder_type: transformer",
    "decoder_type: transformer",
    "position_encoding: true",
    "enc_layers: 4",
    "dec_layers: 4",
    "heads: 4",
    "hidden_size: 256",
    "word_vec_size: 256",
    "transformer_ff: 1024",
    "dropout: 0.1",
    "",
    "optim: adam",
    "learning_rate: 1.0",
    "warmup_steps: 400",
    "decay_method: noam",
    "label_smoothing: 0.1",
    "param_init_glorot: true",
    "",
    "batch_size: $BatchSize",
    "valid_batch_size: $ValidBatchSize",
    "batch_type: sents",
    "accum_count: 2",
    "num_workers: 0",
    "world_size: 1",
    "gpu_ranks: []",
    "n_sample: 0"
)

Set-Content -Path $ConfigPath -Value (($yamlLines -join "`n") + "`n") -Encoding UTF8

Write-Host "Step 2: generate plain-text vocab files"
& $PythonExe -c "from collections import Counter; from pathlib import Path; import sys; src=Path(sys.argv[1]); tgt=Path(sys.argv[2]); src_out=Path(sys.argv[3]); tgt_out=Path(sys.argv[4]); specials=['<blank>','<s>','</s>','<unk>']; c1=Counter(); c2=Counter(); [c1.update(line.strip().split()) for line in src.open('r', encoding='utf-8') if line.strip()]; [c2.update(line.strip().split()) for line in tgt.open('r', encoding='utf-8') if line.strip()]; src_out.parent.mkdir(parents=True, exist_ok=True); tgt_out.parent.mkdir(parents=True, exist_ok=True); f1=src_out.open('w', encoding='utf-8', newline='\n'); [f1.write(f'{sp}\t1000000\n') for sp in specials]; [f1.write(f'{tok}\t{cnt}\n') for tok,cnt in c1.most_common() if tok not in specials]; f1.close(); f2=tgt_out.open('w', encoding='utf-8', newline='\n'); [f2.write(f'{sp}\t1000000\n') for sp in specials]; [f2.write(f'{tok}\t{cnt}\n') for tok,cnt in c2.most_common() if tok not in specials]; f2.close()" $TrainSrc $TrainTgt $SrcVocab $TgtVocab
if ($LASTEXITCODE -ne 0) { throw "vocab generation failed." }

Write-Host "Step 3: train baseline transformer model"
if (Test-Path $TrainScript) {
    & $PythonExe $TrainScript -config $ConfigPath
} else {
    & $PythonExe -m onmt.bin.train -config $ConfigPath
}
if ($LASTEXITCODE -ne 0) { throw "baseline training failed." }

$LatestBaseline = Get-ChildItem $ModelsDir -Filter "baseline_pretrained_step_*.pt" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not $LatestBaseline) {
    throw "No baseline_pretrained_step_*.pt found after training."
}

Copy-Item -Path $LatestBaseline.FullName -Destination $BaselineFinal -Force
Write-Host "Baseline ready: $BaselineFinal"
Write-Host "Source checkpoint: $($LatestBaseline.FullName)"

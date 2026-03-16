$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot
$PythonExe = Join-Path $ProjectRoot "venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
}
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

$TranslateScript = Join-Path $ProjectRoot "OpenNMT-py\translate.py"
$TestSrc = Join-Path $ProjectRoot "data\processed\test.src"
$BaselineModel = Join-Path $ProjectRoot "models\baseline_pretrained.pt"
$FinetunedDir = Join-Path $ProjectRoot "models"
$BaselineOut = Join-Path $ProjectRoot "outputs\baseline_test_pred.txt"
$FinetunedOut = Join-Path $ProjectRoot "outputs\finetuned_test_pred.txt"

if (-not (Test-Path $BaselineModel)) {
    $BaselineModel = Join-Path $ProjectRoot "OpenNMT-py\onmt\tests\test_model.pt"
}

$FinetunedModel = Get-ChildItem $FinetunedDir -Filter "finetuned_model_step_*.pt" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not $FinetunedModel) {
    Write-Host "No finetuned checkpoint found. Falling back to baseline model for finetuned output."
    $FinetunedModelPath = $BaselineModel
} else {
    $FinetunedModelPath = $FinetunedModel.FullName
}

if (-not (Test-Path $BaselineModel)) {
    Write-Host "No baseline checkpoint found. Reusing finetuned checkpoint for baseline output."
    $BaselineModel = $FinetunedModelPath
}

Write-Host "Generating baseline predictions..."
if (Test-Path $TranslateScript) {
    & $PythonExe $TranslateScript -model $BaselineModel -src $TestSrc -output $BaselineOut -beam_size 5 -batch_size 4 -gpu -1
} else {
    & $PythonExe -m onmt.bin.translate -model $BaselineModel -src $TestSrc -output $BaselineOut -beam_size 5 -batch_size 4 -gpu -1
}

Write-Host "Generating finetuned predictions..."
if (Test-Path $TranslateScript) {
    & $PythonExe $TranslateScript -model $FinetunedModelPath -src $TestSrc -output $FinetunedOut -beam_size 5 -batch_size 4 -gpu -1
} else {
    & $PythonExe -m onmt.bin.translate -model $FinetunedModelPath -src $TestSrc -output $FinetunedOut -beam_size 5 -batch_size 4 -gpu -1
}

Write-Host "Saved:"
Write-Host "  $BaselineOut"
Write-Host "  $FinetunedOut"

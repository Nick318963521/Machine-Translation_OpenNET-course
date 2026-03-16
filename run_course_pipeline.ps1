param(
    [string]$ProjectRoot = $PSScriptRoot,
    [ValidateSet("quick", "report")]
    [string]$Preset = "quick",
    [string]$SourceColumn = "source",
    [string]$TargetColumn = "target"
)

$ErrorActionPreference = "Stop"

$PythonExe = Join-Path $ProjectRoot "venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
}
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

Write-Host "Course pipeline started. preset=$Preset"

Write-Host "[1/6] Finetune"
& powershell -ExecutionPolicy Bypass -File (Join-Path $ProjectRoot "run_finetune.ps1") -ProjectRoot $ProjectRoot -Preset $Preset -SourceColumn $SourceColumn -TargetColumn $TargetColumn
if ($LASTEXITCODE -ne 0) { throw "run_finetune.ps1 failed." }

Write-Host "[2/6] Translate test set"
& powershell -ExecutionPolicy Bypass -File (Join-Path $ProjectRoot "run_translate.ps1")
if ($LASTEXITCODE -ne 0) { throw "run_translate.ps1 failed." }

Write-Host "[3/6] BLEU evaluation"
& $PythonExe (Join-Path $ProjectRoot "evaluate_bleu.py")
if ($LASTEXITCODE -ne 0) { throw "evaluate_bleu.py failed." }

Write-Host "[4/6] Benchmark"
& $PythonExe (Join-Path $ProjectRoot "benchmark.py")
if ($LASTEXITCODE -ne 0) { throw "benchmark.py failed." }

Write-Host "[5/6] Plot benchmark"
& $PythonExe (Join-Path $ProjectRoot "plot_benchmark.py")
if ($LASTEXITCODE -ne 0) { throw "plot_benchmark.py failed." }

Write-Host "[6/6] Generate report summary"
& $PythonExe (Join-Path $ProjectRoot "summarize_results.py")
if ($LASTEXITCODE -ne 0) { throw "summarize_results.py failed." }

Write-Host "Course pipeline finished. Check outputs folder for BLEU logs, benchmark CSV/plots, and course_report_summary.md"

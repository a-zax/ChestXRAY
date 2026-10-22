# PowerShell helper script for Cognitive Radiology Report Generator
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ScriptArgs
)

$ErrorActionPreference = "Stop"

# Auto-locate best Python interpreter (prefers Anaconda or any environment with PyTorch)
$Candidates = @(
    "D:\Anaconda3\python.exe",
    "$PSScriptRoot\venv\Scripts\python.exe",
    (Get-Command python -ErrorAction SilentlyContinue).Source,
    (Get-Command py -ErrorAction SilentlyContinue).Source
)

$PythonExe = $null
foreach ($cand in $Candidates) {
    if ($cand -and (Test-Path $cand)) {
        # Check if torch is available in this interpreter
        $hasTorch = & $cand -c "import torch; print(1)" 2>$null
        if ($hasTorch -eq "1") {
            $PythonExe = $cand
            break
        } elseif (-not $PythonExe) {
            $PythonExe = $cand
        }
    }
}

if (-not $PythonExe) {
    Write-Error "No Python installation detected. Please install Python 3.10+."
    exit 1
}

Write-Host "[*] Active Python: $PythonExe" -ForegroundColor Cyan

# If no arguments passed, show help
if ($ScriptArgs.Count -eq 0) {
    & $PythonExe "$PSScriptRoot\run.py" --help
} else {
    & $PythonExe "$PSScriptRoot\run.py" $ScriptArgs
}

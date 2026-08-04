#!/usr/bin/env pwsh
$ErrorActionPreference = "Stop"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv not found. Run: pip install uv"
}

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -Path $ProjectRoot

uv run python src/pipelines/ingest.py --target chroma --reset @args

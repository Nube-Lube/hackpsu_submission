# One-command start: installs cloudflared if needed, builds, runs with a public https link.
# Usage: .\run.ps1          (public link for phones)
#        .\run.ps1 -Local   (localhost only)
param([switch]$Local)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not $env:GEMINI_API_KEY -and -not (Test-Path ".env/gemini_api.env")) {
    Write-Error "No API key: set GEMINI_API_KEY or put the raw key in .env/gemini_api.env"
}

if (-not $Local -and -not (Get-Command cloudflared -ErrorAction SilentlyContinue)) {
    winget install --id Cloudflare.cloudflared -e --accept-source-agreements --accept-package-agreements
    # Pick up the new PATH entry without reopening the terminal.
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [Environment]::GetEnvironmentVariable("Path", "User")
}

if ($Local) { cargo run --release } else { cargo run --release -- --tunnel }

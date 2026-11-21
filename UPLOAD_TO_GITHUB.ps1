# Upload this unpacked folder to your own GitHub repository.
# IMPORTANT: Only run after reviewing docs/PUBLISHING_AND_DATA.md and confirming
# that your repository is empty and that publishing research code is permitted.
# Usage: .\UPLOAD_TO_GITHUB.ps1
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$repo = 'https://github.com/xmohamedtariq/Madinah-Ethical-AI-Crowd-Prediction.git'

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw 'Git not installed or not in PATH. Install Git for Windows first.'
}
if (Test-Path '.git') {
    throw 'This folder is already a Git repository. Use normal git status/push instead.'
}
Write-Host 'Review your files and permissions before confirming.' -ForegroundColor Yellow
Write-Host 'No competition dataset files or credentials were included in this ZIP.'
$answer = Read-Host 'Type PUBLISH to initialize and push to your GitHub repository'
if ($answer -cne 'PUBLISH') { Write-Host 'Cancelled.'; exit 0 }

git init -b main
if ($LASTEXITCODE -ne 0) { throw 'git init failed' }
git add .
if ($LASTEXITCODE -ne 0) { throw 'git add failed' }
git status --short
Write-Host 'Check the staged file list above for secrets or forbidden data.' -ForegroundColor Yellow
$confirm = Read-Host 'Type YES to commit and push these staged files'
if ($confirm -cne 'YES') { Write-Host 'Commit cancelled. Files remain staged locally.'; exit 0 }
git commit -m 'Initial release: Madinah Ethical AI crowd prediction research'
if ($LASTEXITCODE -ne 0) { throw 'git commit failed. Configure your Git author name and email.' }
git remote add origin $repo
if ($LASTEXITCODE -ne 0) { throw 'Adding remote failed' }
git push -u origin main
if ($LASTEXITCODE -ne 0) { throw 'Push failed. Check repo exists, login, branch status, and remote content.' }
Write-Host 'Push completed. Open:' $repo -ForegroundColor Green

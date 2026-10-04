# Rebuild paper.tex / PDF from Paper.md (does not edit Paper.md).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

pandoc "Paper.md" `
  --from markdown+raw_tex+tex_math_dollars `
  --standalone `
  --lua-filter="scripts/paper-filter.lua" `
  --include-in-header="scripts/paper-header.tex" `
  --number-sections `
  --top-level-division=section `
  --syntax-highlighting=pygments `
  -V fontsize=11pt `
  -V papersize=a4 `
  -V geometry:a4paper,margin=1.15in `
  -V colorlinks=true `
  -V linkcolor=black `
  -V urlcolor=blue `
  -o "paper.tex"

Copy-Item -Force "paper.tex" "paper_standalone.tex"

$pdflatex = "C:\texlive\2025\bin\windows\pdflatex.exe"
& $pdflatex -interaction=nonstopmode -halt-on-error paper.tex
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $pdflatex -interaction=nonstopmode -halt-on-error paper.tex
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$out = "FullyInterpretableMinimalTransformer.pdf"
Move-Item -Force "paper.pdf" $out
Write-Host "Wrote $out"

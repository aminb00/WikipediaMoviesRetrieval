#!/bin/bash
# Compile LaTeX report to PDF

echo "Compiling LaTeX report..."

# Change to documentation directory
cd "$(dirname "$0")"

# Compile twice for TOC
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex

# Clean up auxiliary files
rm -f main.aux main.log main.out main.toc

echo "✓ Compilation complete: main.pdf"

#!/bin/bash

set -e

export LC_ALL=C
export LANG=C
export LC_CTYPE=C

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REPORT_BASENAME="${1:-Report2}"
REPORT_TEX="${REPORT_BASENAME}.tex"

if [ ! -f "$REPORT_TEX" ]; then
    echo "Missing $REPORT_TEX"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p out

# Compile LaTeX using latexmk
# -pdf: generate PDF
# -output-directory=out: put all auxiliary files and PDF in 'out/'
# -interaction=nonstopmode: don't stop on errors
# -bibtex: use bibtex for references
latexmk -pdf -output-directory=out -interaction=nonstopmode -bibtex "$REPORT_TEX"

# Copy the final PDF to the repo root for convenience
if [ -f "out/${REPORT_BASENAME}.pdf" ]; then
    cp "out/${REPORT_BASENAME}.pdf" "../${REPORT_BASENAME}.pdf"
    echo "Compilation successful. ${REPORT_BASENAME}.pdf is in the repo root."
else
    echo "Compilation failed. Check the logs in out/${REPORT_BASENAME}.log."
fi

#!/bin/bash

set -e

export LC_ALL=C
export LANG=C
export LC_CTYPE=C

REPORT_BASENAME="${1:-Report1}"
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

# Optional: Move the final PDF to the root for convenience
if [ -f "out/${REPORT_BASENAME}.pdf" ]; then
    cp "out/${REPORT_BASENAME}.pdf" .
    echo "Compilation successful. ${REPORT_BASENAME}.pdf is in the root directory."
else
    echo "Compilation failed. Check the logs in out/${REPORT_BASENAME}.log."
fi

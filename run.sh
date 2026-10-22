#!/usr/bin/env bash
# POSIX helper script for Cognitive Radiology Report Generator

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ -f "$DIR/venv/bin/python" ]; then
    PYTHON_EXE="$DIR/venv/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON_EXE="python3"
elif command -v python &> /dev/null; then
    PYTHON_EXE="python"
else
    echo "Error: Python 3 not found." >&2
    exit 1
fi

if [ $# -eq 0 ]; then
    "$PYTHON_EXE" "$DIR/run.py" --help
else
    "$PYTHON_EXE" "$DIR/run.py" "$@"
fi

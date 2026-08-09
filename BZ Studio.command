#!/bin/bash
# Double-click this file in Finder to launch BZ Studio.
# First launch: right-click and choose Open to pass Gatekeeper.
cd "$(dirname "$0")" || exit 1

if [ -x ".venv/bin/python" ] && .venv/bin/python -c "import PyQt6" 2>/dev/null; then
    exec .venv/bin/python main.py
fi

if python3 -c "import PyQt6" 2>/dev/null; then
    exec python3 main.py
fi

echo "PyQt6 is not installed in '.venv' or for 'python3'."
echo "Install dependencies first:  python3 -m pip install -r requirements.txt"
read -r -p "Press Return to close…" _

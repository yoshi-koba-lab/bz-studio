#!/bin/bash
# Double-click this file in Finder to launch the Keyence KTF Viewer.
# (First time: right-click → Open, to get past macOS Gatekeeper.)
cd "$(dirname "$0")" || exit 1

# Use a python3 that has PyQt6 available.
if python3 -c "import PyQt6" 2>/dev/null; then
    exec python3 main.py
fi

echo "PyQt6 is not installed for 'python3'."
echo "Install dependencies first:  python3 -m pip install -r requirements.txt"
read -r -p "Press Return to close…" _

#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

rm -r "${SCRIPT_DIR}/dist"
python3 -m build
python3 -m twine upload dist/*
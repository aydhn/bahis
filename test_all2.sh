#!/bin/bash
export PYTHONPATH=.
find tests -name "*.py" -type f | while read file; do
    echo "Running $file"
    ~/.pyenv/versions/3.12.12/bin/python -m pytest $file > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "$file PASS"
    else
        echo "$file FAIL"
    fi
done

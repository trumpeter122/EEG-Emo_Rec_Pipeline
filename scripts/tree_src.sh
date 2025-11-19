#!/bin/bash

tree src -I "__pycache__" -C --dirsfirst --sort=name

echo Total lines:
echo $(find src -type d -name __pycache__ -prune -o -type f -name '*.py' -print | xargs cat | wc -l)

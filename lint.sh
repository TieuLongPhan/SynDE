#!/bin/bash

# Check syntax-level failures and unused imports across library and tests.
# Formatting cleanup is intentionally separate from this correctness gate.
flake8 synde test --count \
    --select=E9,F401,F63,F7,F82 \
    --per-file-ignores="__init__.py:F401" \
    --statistics

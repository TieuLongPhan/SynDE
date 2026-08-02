#!/bin/bash

set -e

echo "Building Sphinx documentation for SynDE..."

SPHINX_BIN=$(which sphinx-build 2>/dev/null || echo "python -m sphinx")

echo "Using Sphinx: $SPHINX_BIN"

$SPHINX_BIN -b html ./doc ./docs

echo "✅ Documentation successfully built in ./docs/index.html"
#!/bin/bash

ROOT_DIR=$(cd $(dirname $0)/..; pwd)

DOCS_DIR="$ROOT_DIR/docs/sphinx/manual"
DOCS_OUT="$ROOT_DIR/docs/built"

# Clean up old documentation
rm -rf "$DOCS_OUT"
mkdir -p "$DOCS_OUT"

for lang in ja en; do
  # Generate API documentation
  rm -rf "$DOCS_DIR/$lang/source/api"
  uv run sphinx-apidoc -f -T -d 3 -e -o "$DOCS_DIR/$lang/source/api" "$ROOT_DIR/src/physbo"

  # Build documentation
  uv run sphinx-build -d "$DOCS_OUT/doctree/$lang" "$DOCS_DIR/$lang/source" "$DOCS_OUT/manual/$lang" --color -bhtml
done

#!/bin/bash

if ! which sphinx-build >/dev/null 2>&1; then
  echo "ERROR: Sphinx is not installed"
  exit 1
fi

LANGS="ja en"
if [[ $# -gt 0 ]]; then
  LANGS="$*"
fi

ROOT_DIR=$(
  cd $(dirname $0)/..
  pwd
)

DOCS_DIR="$ROOT_DIR/docs/sphinx/manual"
DOCS_OUT="$ROOT_DIR/docs/built"

# Clean up old documentation
rm -rf "$DOCS_OUT"
mkdir -p "$DOCS_OUT"

for lang in ${LANGS}; do
  if [[ $lang != "ja" && $lang != "en" ]]; then
    echo "ERROR: Unsupported language: $lang"
    exit 1
  fi

  # Generate API documentation
  rm -rf "$DOCS_DIR/$lang/source/api"
  sphinx-apidoc -f -T -d 3 -e -o "$DOCS_DIR/$lang/source/api" "$ROOT_DIR/src/physbo"

  # The policies share methods through base classes of a private module
  # (physbo/search/_policy_common.py), so list the inherited methods on the
  # page of each policy; elsewhere inherited members stay out.
  for policy in discrete discrete_multi discrete_unified range range_multi range_unified; do
    rst="$DOCS_DIR/$lang/source/api/physbo.search.$policy.rst"
    awk '{print} /^   :members:$/ {print "   :inherited-members:"}' "$rst" >"$rst.tmp" && mv "$rst.tmp" "$rst"
    # stop if the page does not have the form expected above, rather than
    # build a reference without the inherited methods
    if ! grep -q '^   :inherited-members:$' "$rst"; then
      echo "ERROR: could not add :inherited-members: to $rst"
      exit 1
    fi
  done

  # Build documentation
  sphinx-build -d "$DOCS_OUT/doctree/$lang" "$DOCS_DIR/$lang/source" "$DOCS_OUT/manual/$lang" --color -bhtml
done

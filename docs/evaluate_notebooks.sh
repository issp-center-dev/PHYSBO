#!/bin/bash

if ! python3 -m jupyter --version > /dev/null 2>&1; then
  echo "ERROR: Jupyter is not found"
  exit 1
fi

INPLACE=""
PARALLEL_JOBS=4

while (($# > 0)); do
  case "$1" in
    -h | --help)
      echo "Usage: $0 [--inplace] [-j <parallel-jobs>]"
      echo "  --inplace: Execute notebooks in place"
      echo "  -j <parallel-jobs>: Number of parallel jobs"
      echo "  -h | --help: Show this help message"
      exit 0
      ;;
    --inplace)
      INPLACE="--inplace"
      shift
      ;;
    -j | --parallel-jobs)
      if [[ -z "$2" ]] || [[ "$2" =~ ^[0-9]+$ ]]; then
        echo "Error: -j/--parallel-jobs requires an integer argument"
        exit 1
      fi
      PARALLEL_JOBS="$2"
      shift
      ;;
    *)
      break
      ;;
  esac
  shift
done

ROOT_DIR=$(cd $(dirname $0)/..; pwd)

DOCS_DIR="$ROOT_DIR/docs/sphinx/manual"

for lang in ja en; do
    NOTEBOOKS_DIR="$DOCS_DIR/$lang/source/notebook"
    cd "$NOTEBOOKS_DIR" || exit 1
    if which parallel > /dev/null 2>&1; then
        parallel -j "$PARALLEL_JOBS" python3 -m jupyter execute $INPLACE ::: *.ipynb
    else
        for notebook in *.ipynb; do
            python3 -m jupyter execute $INPLACE "$notebook"
        done
    fi
done

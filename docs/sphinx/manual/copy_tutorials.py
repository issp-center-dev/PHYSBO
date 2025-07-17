#!/usr/bin/env python3

# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import sys
import os
import re
import shutil
import jupytext


def extract_toctree_files(rst_file):
    """
    Extract file names from toctree directive in an RST file.

    Args:
        rst_file (str): Path to the RST file

    Returns:
        list: List of file names found in toctree directive
    """
    try:
        with open(rst_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File {rst_file} not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file {rst_file}: {e}")
        sys.exit(1)

    files = []
    in_toctree = False
    toctree_started = False

    for line in lines:
        # Detect start of toctree directive
        if re.match(r"^\s*\.\.\s+toctree::", line):
            in_toctree = True
            toctree_started = False
            continue

        if in_toctree:
            # Skip option lines (lines starting with :)
            if re.match(r"^\s*:", line):
                continue

            # Skip empty lines before content starts
            if re.match(r"^\s*$", line) and not toctree_started:
                continue

            # If we encounter an empty line after content started, end toctree
            if re.match(r"^\s*$", line) and toctree_started:
                in_toctree = False
                continue

            # If we encounter another directive (line starting with ..), end toctree
            if re.match(r"^\s*\.\.", line):
                in_toctree = False
                continue

            # If we get here, this should be a file name
            if re.match(r"^\s*[^\s]", line):
                toctree_started = True
                # Remove leading/trailing whitespace and add to list
                filename = line.strip()
                if filename:
                    files.append(filename)

    return files


def copy_file(source_basename, source_dir, target_dir, index: int):
    # example:
    # INPUT:
    #   source_basename: tutorial_basic
    #   index: 1
    # OUTPUT:
    #   target_dir: 01.basic
    #   target_path_base: 01.basic/basic

    source_file = source_basename + ".ipynb"
    source_path = os.path.join(source_dir, source_file)

    if source_basename.startswith("tutorial_"):
        target_basename = source_basename[len("tutorial_"):]
    else:
        target_basename = source_basename
    target_dir_name = os.path.join(target_dir, f"{index:02d}.{target_basename}")
    os.makedirs(target_dir_name, exist_ok=True)
    target_path_base = os.path.join(target_dir_name, target_basename)

    notebook = jupytext.read(source_path)
    jupytext.write(notebook, target_path_base + ".ipynb", fmt="notebook")
    jupytext.write(notebook, target_path_base + ".py", fmt="py:percent")

    use_s5_210 = False
    with open(target_path_base + ".py", "r") as f:
        for line in f:
            if "s5-210.csv" in line:
                use_s5_210 = True
                break
    if use_s5_210:
        shutil.copy(os.path.join(source_dir, "s5-210.csv"), target_dir_name)



def main():
    """Main function to handle command line arguments and execute the extraction."""
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <notebook_dir> <target_dir>")
        sys.exit(1)

    notebook_dir = sys.argv[1]
    target_dir = sys.argv[2]

    index_rst = os.path.join(notebook_dir, "index.rst")

    if not os.path.isfile(index_rst):
        print(f"Error: File {index_rst} not found")
        sys.exit(1)

    files = extract_toctree_files(index_rst)

    for index, file in enumerate(files):
        copy_file(file, notebook_dir, target_dir, index+1)


if __name__ == "__main__":
    main()

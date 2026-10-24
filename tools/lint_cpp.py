# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Check first-party C++/CUDA sources; --fix applies clang-format only."""

import argparse
from importlib.metadata import version
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = {"include", "src", "app", "tests"}
EXTENSIONS = {".cc", ".h", ".cu", ".cuh"}
FORMAT_VERSION = "19.1.5"
CPPLINT_VERSION = "2.0.2"


def source_files():
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT
    ).decode("utf-8").split("\0")
    return sorted(
        name for name in tracked
        if name and Path(name).parts[0] in SOURCE_DIRS
        and Path(name).suffix in EXTENSIONS
    )


def check_guards(files):
    failed = False
    for name in files:
        if Path(name).suffix not in {".h", ".cuh"}:
            continue
        guard = "GRTMESH_" + re.sub(r"[^A-Za-z0-9]", "_", name).upper() + "_"
        text = (ROOT / name).read_text(encoding="utf-8")
        lines = text.splitlines()
        directives = [line for line in lines if line.startswith("#")]
        expected = [f"#ifndef {guard}", f"#define {guard}"]
        if (directives[:2] != expected or "#pragma once" in text
                or not lines or lines[-1] != f"#endif  // {guard}"):
            print(f"{name}: expected include guard {guard}", file=sys.stderr)
            failed = True
    return int(failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fix", action="store_true")
    parser.add_argument("--format-only", action="store_true")
    parser.add_argument("--clang-format", default="clang-format")
    parser.add_argument("--tidy", action="store_true",
                        help="also check host .cc files using a compile database")
    parser.add_argument("--clang-tidy", default="clang-tidy")
    parser.add_argument("--build-dir", type=Path)
    args = parser.parse_args()
    format_tool = shutil.which(args.clang_format)
    if not format_tool:
        parser.error("clang-format not found; install tools/requirements-lint.txt")
    actual_version = subprocess.check_output(
        [format_tool, "--version"], text=True
    )
    if not re.search(r"\b" + re.escape(FORMAT_VERSION) + r"\b", actual_version):
        parser.error(f"clang-format {FORMAT_VERSION} required: {actual_version.strip()}")
    if args.tidy and (not args.build_dir or not
                     (args.build_dir / "compile_commands.json").is_file()):
        parser.error("--tidy requires --build-dir containing compile_commands.json")
    tidy_tool = shutil.which(args.clang_tidy) if args.tidy else None
    if args.tidy and not tidy_tool:
        parser.error("clang-tidy not found")
    files = source_files()
    if not files:
        parser.error("no tracked C++/CUDA sources found")
    print(f"Checking {len(files)} tracked C++/CUDA files", flush=True)
    format_args = ["-i"] if args.fix else ["--dry-run", "--Werror"]
    failed = subprocess.call(
        [format_tool, "--style=file", *format_args, *files], cwd=ROOT
    )
    if args.format_only:
        return int(failed != 0)
    if version("cpplint") != CPPLINT_VERSION:
        parser.error(f"cpplint {CPPLINT_VERSION} required")
    failed |= check_guards(files)
    failed |= subprocess.call(
        [sys.executable, "-m", "cpplint", "--extensions=cc,h,cu,cuh",
         "--headers=h,cuh", "--quiet", *files], cwd=ROOT
    )
    if args.tidy:
        for name in files:
            if Path(name).suffix == ".cc":
                failed |= subprocess.call(
                    [tidy_tool, "-p", str(args.build_dir.resolve()), name], cwd=ROOT
                )
    return int(failed != 0)


if __name__ == "__main__":
    sys.exit(main())

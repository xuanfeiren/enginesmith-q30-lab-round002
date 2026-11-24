#!/bin/bash
# Development only: build a variant of this engine tree in /tmp/q30x/NAME (sources copied, NEFF cache shared with
# this tree's target/neff/cache), with the environment the caller sets (ENGINE_DEC_BUCKETS, ENGINE_DEC_KW, ...).
# usage: tools/xbuild.sh NAME   -> /tmp/q30x/NAME/target/engine
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
D=/tmp/q30x/$1
rm -rf "$D"; mkdir -p "$D/target/neff" "$SRC/target/neff/cache"
cp -r "$SRC/build.sh" "$SRC/host" "$SRC/kernels" "$SRC/tools" "$D/"
ln -s "$SRC/target/neff/cache" "$D/target/neff/cache"
cd "$D" && bash build.sh

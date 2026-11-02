#!/bin/bash
# Build the NPU decoder for every bucket: surgery (A and B graphs), then compile each.
#
#   build/build_buckets.sh MODELS PYTHON TOOLKIT_PYTHON [BUCKETS...]
#
# MODELS          directory with dec.onnx (build/extract.py)
# PYTHON          a Python with onnx and numpy
# TOOLKIT_PYTHON  a Python with rknn-toolkit2 2.3.2
# BUCKETS         default: 64 96 128 160 200 256 320 400
#
# Writes MODELS/buckets/dec_T<T>_{A,B}.{onnx,rknn} and a log per graph in MODELS/buckets/logs.
# Each graph takes ~0.5-1.5 minutes to compile. A graph with REGTASK errors must not be run.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
MODELS=$1; PY=$2; TK=$3; shift 3
BUCKETS=${*:-64 96 128 160 200 256 320 400}
mkdir -p "$MODELS/buckets/logs"
bad=0
for T in $BUCKETS; do
  if ! "$PY" "$HERE/surgery.py" --models "$MODELS" --T "$T" --out "$MODELS/buckets/dec_T$T.onnx" > "$MODELS/buckets/logs/T${T}_surgery.log" 2>&1; then
    echo "T=$T: surgery FAILED (see $MODELS/buckets/logs/T${T}_surgery.log)"; bad=1; continue
  fi
  for k in A B; do
    log="$MODELS/buckets/logs/T${T}_$k.log"
    "$TK" "$HERE/compile_rknn.py" "$MODELS/buckets/dec_T${T}_$k.onnx" "$MODELS/buckets/dec_T${T}_$k.rknn" > "$log" 2>&1
    n=$(grep -a -c "exceeds the limit" "$log")
    echo "T=$T $k: $n REGTASK errors; $(grep -a COMPILED "$log" || echo 'compile FAILED')"
    if [ "$n" != 0 ]; then rm -f "$MODELS/buckets/dec_T${T}_$k.rknn"; echo "  removed dec_T${T}_$k.rknn: it would hang the NPU"; bad=1; fi
  done
done
exit $bad

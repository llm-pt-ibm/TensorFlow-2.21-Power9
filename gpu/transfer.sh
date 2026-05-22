#!/bin/bash
# Inspeciona os 3 BUILD files untracked em third_party/gpus/ antes de gerar
# o tarball de export pra branch power9-v2.21.0-gpu.
# Roda DENTRO do container gaby_workspace.

cd /root/tensorflow/tf221_workspace/tensorflow

for f in third_party/gpus/BUILD third_party/gpus/cuda/BUILD third_party/gpus/cudnn/BUILD; do
    if [ -f "$f" ]; then
        echo "=== $f ($(wc -l < "$f") lines) ==="
        head -40 "$f"
        echo
    else
        echo "=== $f (MISSING) ==="
        echo
    fi
done

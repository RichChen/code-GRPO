#!/usr/bin/env bash
# Fallback for the two base weight files when the HF mirror redirects them to a CDN that is too slow.
# Fetches them from ModelScope in parallel byte ranges, then accepts a file only if its sha256 equals the LFS
# sha256 that the HF mirror reports for snapshot 70d244cc86cc (X-Linked-ETag). Accepted files go into the HF cache
# as blobs/<sha256> with the snapshot symlink, so `hf download --revision 70d244cc86cc` then finds them.
set -euo pipefail
REV=70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
REPO=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B
STAGE=$HOME/.cache/arl-modelscope
PARTS=${PARTS:-16}
mkdir -p "$STAGE" "$REPO/blobs" "$REPO/snapshots/$REV"

hf_sha() {
  curl -sI -m 30 "https://hf-mirror.com/Qwen/Qwen3-1.7B/resolve/$REV/$1" |
    sed -n 's/^x-linked-etag: "\([0-9a-f]\{64\}\)".*/\1/ip' | head -1
}

fetch() {
  local name=$1 size=$2 url="https://modelscope.cn/models/Qwen/Qwen3-1.7B/resolve/master/$1"
  local chunk=$(( (size + PARTS - 1) / PARTS ))
  local pids=()
  for i in $(seq 0 $((PARTS - 1))); do
    (
      start=$((i * chunk)); end=$(( (i + 1) * chunk - 1 )); [ $end -ge $size ] && end=$((size - 1))
      part="$STAGE/$name.part$i"; want=$((end - start + 1))
      touch "$part"
      for _ in $(seq 1 200); do
        have=$(stat -c %s "$part")
        [ "$have" -ge "$want" ] && exit 0
        curl -s -L --speed-limit 20000 --speed-time 30 -r $((start + have))-$end "$url" >> "$part" || true
      done
      exit 1
    ) &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  : > "$STAGE/$name"
  for i in $(seq 0 $((PARTS - 1))); do cat "$STAGE/$name.part$i" >> "$STAGE/$name"; done
  rm -f "$STAGE/$name".part*
}

for entry in "model-00001-of-00002.safetensors 3441185608" "model-00002-of-00002.safetensors 622329984"; do
  set -- $entry
  name=$1 size=$2
  want=$(hf_sha "$name")
  [ -n "$want" ] || { echo "no HF sha256 for $name; stop" >&2; exit 1; }
  if [ -f "$REPO/blobs/$want" ]; then echo "$name already cached"; continue; fi
  t0=$(date +%s)
  fetch "$name" "$size"
  got=$(sha256sum "$STAGE/$name" | cut -d' ' -f1)
  echo "$name bytes=$(stat -c %s "$STAGE/$name") sha256=$got hf=$want seconds=$(( $(date +%s) - t0 ))"
  if [ "$got" != "$want" ]; then echo "sha256 mismatch for $name; stop" >&2; exit 1; fi
  mv "$STAGE/$name" "$REPO/blobs/$want"
  ln -sfn "../../blobs/$want" "$REPO/snapshots/$REV/$name"
  rm -f "$REPO/blobs/$want".*.incomplete
done
echo "fetch ok"

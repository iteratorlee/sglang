#!/usr/bin/env bash
# Run ONLY on authorized 910b-27. Creates this task's utility container only.
set -euo pipefail
case "$(hostname)" in *910b*27) ;; *) echo 'Expected authorized 910b-27 host' >&2; exit 1;; esac
root=/public-flash/liyan/glm53-flash-910b-opt/experiments/cold-score-opt-1012
container=glm53-910b-cold-score-micro
test -f "$root/source/python/sglang/srt/hardware_backend/npu/attention/glm53/pooled_score_prefill_npu.py"
mkdir -p "$root/results" "$root/triton-cache" "$root/ascend-cache"
docker run -d --name "$container" --privileged --network host --ipc host \
  --runtime ascend --stop-timeout 30 -w /eval \
  -e SOC_VERSION=ascend910b1 -e ASCEND_RT_VISIBLE_DEVICES=7 \
  -e USE_FLAGGEMS=0 -e SGLANG_FLAGGEMS_RECORD=0 \
  -e TRITON_CACHE_DIR=/eval/triton-cache -e ASCEND_CACHE_PATH=/eval/ascend-cache \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$root/source:/sgl-workspace/sglang:ro" -v "$root:/eval" \
  -v /usr/local/sbin:/usr/local/sbin:ro -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /var/log/npu:/usr/slog -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
  -v /etc/hccn.conf:/etc/hccn.conf:ro -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  quay.io/ascend/sglang:main-cann9.0.0-910b sleep infinity

#!/usr/bin/env bash
# 等待当前 Qwen Thinking-On 冒烟结束后，顺序跑完剩余的冒烟测试。
# 仅用于开发验证，不会影响全量实验目录。
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
LOG_DIR="$ROOT/experiments/spar_bench/logs"
mkdir -p "$LOG_DIR"
ts() { date '+%Y-%m-%d %H:%M:%S'; }

WAIT_PID="${1:-2451345}"
QWEN_ON_DIR="$ROOT/experiments/spar_bench/tiny_rgb/qwen3_5_4b/think_on_8192/smoke20"

echo "[$(ts)] waiting for pid $WAIT_PID" | tee -a "$LOG_DIR/smoke_remaining.log"
while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 30
done

count=$(wc -l < "$QWEN_ON_DIR/predictions.jsonl" 2>/dev/null || echo 0)
if [ "$count" -lt 20 ]; then
    echo "[$(ts)] ABORT: qwen think_on smoke only $count/20" | tee -a "$LOG_DIR/smoke_remaining.log"
    exit 1
fi
echo "[$(ts)] qwen think_on smoke done: $count/20" | tee -a "$LOG_DIR/smoke_remaining.log"

run_one() {
    local model="$1" env_name="$2" protocol="$3"
    local log="$LOG_DIR/smoke_${model}_${protocol}.log"
    echo "[$(ts)] SMOKE START ${model} ${protocol}" | tee -a "$log"
    micromamba run -n "$env_name" python -u evaluation/spar_bench/run_evaluation.py \
        --model "$model" --protocol "$protocol" --limit 20 --stratified \
        --run-name smoke20 --resume >>"$log" 2>&1
    local rc=$?
    echo "[$(ts)] SMOKE END ${model} ${protocol} rc=${rc}" | tee -a "$log"
    return "$rc"
}

run_one qwen35 vlm-test direct_off_64 || exit 1
run_one xiaomi_lens xiaomi-lens think_off_8192 || exit 1
run_one xiaomi_lens xiaomi-lens think_on_8192 || exit 1
run_one xiaomi_lens xiaomi-lens direct_off_64 || exit 1

echo "[$(ts)] ALL SMOKE DONE" | tee -a "$LOG_DIR/smoke_remaining.log"

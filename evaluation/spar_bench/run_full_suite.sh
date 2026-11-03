#!/usr/bin/env bash
# 顺序跑完 6 组 SPAR-Bench-Tiny 全量实验（每个模型三种协议）。
#
# 用法：
#   bash evaluation/spar_bench/run_full_suite.sh
#   nohup bash evaluation/spar_bench/run_full_suite.sh > /dev/null 2>&1 &
#
# 特性：
#   * 同一时间只加载一个模型，跑完一个模型再跑下一个；
#   * 每组都带 --resume，进程中断后可直接重跑本脚本；
#   * 日志写到 experiments/spar_bench/logs/<model>_<protocol>.log；
#   * 任意一组返回非 0 就停止，避免在坏状态下继续烧 GPU。
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
LOG_DIR="$ROOT/experiments/spar_bench/logs"
mkdir -p "$LOG_DIR"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

run_one() {
    local model="$1" env_name="$2" protocol="$3"
    local log="$LOG_DIR/${model}_${protocol}.log"
    echo "[$(ts)] START ${model} ${protocol}" | tee -a "$log"
    micromamba run -n "$env_name" python -u evaluation/spar_bench/run_evaluation.py \
        --model "$model" --protocol "$protocol" --resume \
        >>"$log" 2>&1
    local rc=$?
    echo "[$(ts)] END ${model} ${protocol} rc=${rc}" | tee -a "$log"
    return "$rc"
}

echo "[$(ts)] SUITE START pid=$$"

for spec in \
    "xiaomi_lens xiaomi-lens think_off_8192" \
    "xiaomi_lens xiaomi-lens think_on_8192" \
    "xiaomi_lens xiaomi-lens direct_off_64" \
    "qwen35 vlm-test think_off_8192" \
    "qwen35 vlm-test think_on_8192" \
    "qwen35 vlm-test direct_off_64"
do
    # shellcheck disable=SC2086
    set -- $spec
    if ! run_one "$1" "$2" "$3"; then
        echo "[$(ts)] SUITE STOP: $spec 失败" | tee -a "$LOG_DIR/suite.log"
        exit 1
    fi
done

echo "[$(ts)] SUITE COMPLETE"
micromamba run -n vlm-test python evaluation/spar_bench/make_reports.py \
    >>"$LOG_DIR/make_reports.log" 2>&1
micromamba run -n vlm-test python evaluation/spar_bench/verify_results.py \
    >>"$LOG_DIR/verify_results.log" 2>&1
echo "[$(ts)] REPORTS + VERIFICATION DONE"

#!/usr/bin/env bash
# Reproduce the red operational-decisions research (numbers and figures) from local
# wp6_red. The research note itself is kept outside the repository.
#
#   scripts/red_operations/run_all.sh <work_dir> [fig_dir]   (from the repo root)
#
# Needs LetsGrow history in the local red DB back to 2023-02 (see extract.py),
# and network access once for the public DE-LU day-ahead prices (red is at
# Neurath, Germany).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${1:?work dir}"
FIG="${2:-$WORK/fig}"
mkdir -p "$WORK"
export RED_OPS_DATA="$WORK/prices"
UV=(uv run --with pyarrow --with statsmodels --with matplotlib python -I -W ignore)

[ -f "$RED_OPS_DATA/epex_delu_2023.json" ] || "${UV[@]}" "$HERE/prices.py" 2023 2024 2025 2026
"${UV[@]}" "$HERE/extract.py" "$WORK/lg5.parquet"
"${UV[@]}" "$HERE/analyze.py" "$WORK/lg5.parquet" "$WORK/out"
"${UV[@]}" "$HERE/run_planner.py" "$WORK/lg5.parquet" "$WORK/out/planner.csv"
"${UV[@]}" "$HERE/activity_report.py" "$WORK/lg5.parquet" "$WORK/out/daily.csv" "$WORK/out/activity.json"
"${UV[@]}" "$HERE/figures.py" "$WORK/out" "$WORK/lg5.parquet" "$FIG" "2023/24,2024/25,2025/26"

#!/usr/bin/env bash
# Watchdog for the oellm_build baseline compilation. Polls every 60 seconds
# and writes structured status to $STATUS. Exits when the build process is
# gone (either successfully or after a crash).

set -u

LOG="${LOG:-$HOME/oe_locateanything/main/logs/baseline_qwen2_5-vl-3b.log}"
STATUS="${STATUS:-$HOME/oe_locateanything/main/logs/watchdog_status.txt}"
OUTDIR="${OUTDIR:-$HOME/oe_locateanything/main/language/baseline_outputs/qwen2_5-vl-3b_nash-p_w4}"
PROC_PATTERN="${PROC_PATTERN:-oellm_build.*qwen2_5-vl-3b}"
INTERVAL="${INTERVAL:-60}"

while true; do
  now=$(date -Iseconds)
  pid=$(pgrep -f "$PROC_PATTERN" | head -1 || true)

  if [[ -z "$pid" ]]; then
    {
      echo "ts=$now"
      echo "state=STOPPED"
      if ls "$OUTDIR"/*.hbm >/dev/null 2>&1; then
        echo "result=SUCCESS"
        echo "hbm_files:"
        ls -lh "$OUTDIR"/*.hbm 2>&1 | sed 's/^/  /'
      else
        echo "result=FAILED"
        echo "last_log_lines:"
        tail -40 "$LOG" 2>&1 | sed 's/^/  /'
      fi
    } > "$STATUS"
    exit 0
  fi

  etime=$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ' || echo 'unknown')
  log_mtime=$(stat -c %Y "$LOG" 2>/dev/null || echo 0)
  log_age=$(( $(date +%s) - log_mtime ))
  last_line=$(tail -1 "$LOG" 2>/dev/null | head -c 200)
  err_hit=$(tail -200 "$LOG" 2>/dev/null | grep -iE 'error|traceback|out of memory|cuda.*oom|killed' | tail -5 || true)
  hbm_count=$(ls "$OUTDIR"/*.hbm 2>/dev/null | wc -l)
  bc_count=$(ls "$OUTDIR"/*.bc 2>/dev/null | wc -l)
  hbo_count=$(ls "$OUTDIR"/*.hbo 2>/dev/null | wc -l)

  {
    echo "ts=$now"
    echo "state=RUNNING"
    echo "pid=$pid etime=$etime log_age=${log_age}s"
    echo "artifacts: bc=$bc_count hbo=$hbo_count hbm=$hbm_count"
    echo "last_line: $last_line"
    if [[ -n "$err_hit" ]]; then
      echo "ERROR_HITS:"
      echo "$err_hit" | sed 's/^/  /'
    fi
    if [[ "$log_age" -gt 1800 ]]; then
      echo "WARN: log stalled ${log_age}s (>15min)"
    fi
  } > "$STATUS"

  sleep "$INTERVAL"
done

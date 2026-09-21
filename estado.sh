#!/usr/bin/env bash
# Estado de los entrenamientos.
cd "$(dirname "$0")" || exit 1
printf "%-10s %-12s %-10s %-8s %-9s %s\n" RUN PASOS META FPS FALTA ETA
for f in artifacts/logs/*.out; do
  [ -f "$f" ] || continue
  run=$(basename "$f" .out)
  pid=$(pgrep -f -- "--run-id $run" | head -1)
  ts=$(grep -E "^\|    total_timesteps" "$f" | tail -1 | tr -dc '0-9')
  fps=$(grep -E "^\|    fps" "$f" | tail -1 | tr -dc '0-9')
  meta=$(grep -oE "total_timesteps: [0-9_]+" "artifacts/checkpoints/$run/config.yaml" 2>/dev/null | tr -dc '0-9')
  [ -z "$pid" ] && { printf "%-10s %-12s %-10s %-8s %-9s %s\n" "$run" "${ts:-0}" "${meta:-?}" "-" "-" "TERMINADO/MUERTO"; continue; }
  # Con --resume, SB3 suma total_timesteps al contador que ya traia.
  if [ -n "$ts" ] && [ -n "$meta" ] && [ "$ts" -gt "$meta" ]; then
    base=$(grep -oE "^\| *total_timesteps *\| *[0-9]+" "$f" | head -1 | tr -dc '0-9')
    meta=$(( meta + ${base:-0} ))
  fi
  if [ -n "$ts" ] && [ -n "$fps" ] && [ -n "$meta" ] && [ "$fps" -gt 0 ] && [ "$meta" -gt "$ts" ]; then
    faltan=$(( (meta - ts) / fps ))
    printf "%-10s %-12s %-10s %-8s %-9s %s\n" "$run" "$ts" "$meta" "$fps" \
      "$(printf '%dh%02dm' $((faltan/3600)) $(((faltan%3600)/60)))" \
      "$(date -d "+$faltan seconds" '+%a %H:%M')"
  else
    printf "%-10s %-12s %-10s %-8s %-9s %s\n" "$run" "${ts:-0}" "${meta:-?}" "${fps:-?}" "?" "arrancando"
  fi
done
echo
echo "--- ultima evaluacion (recompensa REAL, 3 vidas, sin recorte) ---"
for d in artifacts/logs/*/; do
  [ -f "$d/evaluations.npz" ] && .venv/bin/python -c "
import numpy as np,sys
z=np.load('$d/evaluations.npz'); r=z['results']
print(f\"  $(basename $d): paso {z['timesteps'][-1]:,} -> media {r[-1].mean():.0f}  max {r[-1].max():.0f}\")" 2>/dev/null
done
echo
echo "--- recursos ---"; free -h | sed -n 2p
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | sed 's/^/  GPU: /'

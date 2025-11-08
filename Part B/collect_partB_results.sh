#!/usr/bin/env bash
set -euo pipefail

# Run from script dir (handles "Part B" and relative paths)
cd "$(dirname "$0")"

Ks=(1 5 10 50 100)
step2_csv="partB_results_step2.csv"
step3_csv="partB_results_step3.csv"

# Ensure binaries exist
for bin in vecaddcpu vecaddgpu00 vecaddgpu01; do
  [[ -x "./$bin" ]] || { echo "Missing binary: $bin  (run: make)"; exit 1; }
done

echo "K_millions,CPU_ms,Scenario1_ms,Scenario2_ms,Scenario3_ms" > "$step2_csv"
echo "K_millions,CPU_ms,Scenario1_ms,Scenario2_ms,Scenario3_ms" > "$step3_csv"

# Extract milliseconds from various output styles:
#  - CSV: "checksum,12.34"
#  - Text: "time = 12.34 ms"
#  - Fallback: last number on the line
get_ms() {
  local line="$1" ms=""
  ms=$(awk -F, 'NF>1{print $2}' <<<"$line" | tr -d '[:space:]')
  if [[ -z "$ms" ]]; then
    ms=$(grep -Eo '[0-9]+([.][0-9]+)?' <<<"$line" | tail -1)
  fi
  printf "%s" "$ms"
}

for K in "${Ks[@]}"; do
  cpu_ms=$(get_ms "$(./vecaddcpu "$K")")

  s1=$(get_ms "$(./vecaddgpu00 "$K" 1)")
  s2=$(get_ms "$(./vecaddgpu00 "$K" 2)")
  s3=$(get_ms "$(./vecaddgpu00 "$K" 3)")
  echo "$K,$cpu_ms,$s1,$s2,$s3" >> "$step2_csv"

  u1=$(get_ms "$(./vecaddgpu01 "$K" 1)")
  u2=$(get_ms "$(./vecaddgpu01 "$K" 2)")
  u3=$(get_ms "$(./vecaddgpu01 "$K" 3)")
  echo "$K,$cpu_ms,$u1,$u2,$u3" >> "$step3_csv"
done

# Plot if possible; never fail the run on plotting
if command -v python3 >/dev/null 2>&1 && [[ -f plot_partB.py ]]; then
  python3 plot_partB.py "$step2_csv" "$step3_csv" || true
  python3 plot_partB.py "$step2_csv" "$step3_csv" --logy || true
else
  echo "Skipping plotting (python3 or plot_partB.py missing)."
fi

echo "Wrote $step2_csv and $step3_csv"

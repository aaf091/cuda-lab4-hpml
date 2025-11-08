!/usr/bin/env bash
set -euo pipefail

# Runs Part B experiments and writes two CSVs for plotting.
# Make sure binaries are built and you're on a GPU node.

Ks=(1 5 10 50 100)

step2_csv="partB_results_step2.csv"
step3_csv="partB_results_step3.csv"

echo "K_millions,CPU_ms,Scenario1_ms,Scenario2_ms,Scenario3_ms" > "${step2_csv}"
echo "K_millions,CPU_ms,Scenario1_ms,Scenario2_ms,Scenario3_ms" > "${step3_csv}"

for K in "${Ks[@]}"; do
  # CPU
  cpu_line=$(./vecaddcpu "$K")
  cpu_ms=$(echo "$cpu_line" | awk -F, '{print $2}')

  # Step 2 (discrete memory)
  s1=$(./vecaddgpu00 "$K" 1 | awk -F, '{print $2}')
  s2=$(./vecaddgpu00 "$K" 2 | awk -F, '{print $2}')
  s3=$(./vecaddgpu00 "$K" 3 | awk -F, '{print $2}')
  echo "${K},${cpu_ms},${s1},${s2},${s3}" >> "${step2_csv}"

  # Step 3 (Unified Memory)
  u1=$(./vecaddgpu01 "$K" 1 | awk -F, '{print $2}')
  u2=$(./vecaddgpu01 "$K" 2 | awk -F, '{print $2}')
  u3=$(./vecaddgpu01 "$K" 3 | awk -F, '{print $2}')
  echo "${K},${cpu_ms},${u1},${u2},${u3}" >> "${step3_csv}"
done

echo "Wrote ${step2_csv} and ${step3_csv}"

# Default: linear axes; saves PNGs in the current dir
python3 plot_partB.py partB_results_step2.csv partB_results_step3.csv

# Log–log axes (recommended when scales differ a lot)
python3 plot_partB.py partB_results_step2.csv partB_results_step3.csv --logy
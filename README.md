# HPML Lab 4 — CUDA Kernels, Memory, and Convolution

This assignment has three parts:

- **Part A:** Vector add & matrix multiply kernels (`00` vs `01`) and speedup analysis  
- **Part B:** CPU vs GPU performance with and without **Unified Memory**; generate two charts  
- **Part C:** 2D convolution (naive CUDA, tiled CUDA, cuDNN) that prints **exactly three lines**

---

## Environment

- **GPU node required**
- CUDA toolchain available either on the host or via **Singularity**
- **cuDNN** available/linked for Part C

NYU HPC (Burst Node) module+container usage:

```bash
cd /scratch/[NetID]
git clone https://github.com/aaf091/cuda-lab4-hpml.git
cd cuda-lab4-hpml
/scratch/work/public/singularity/run-cuda-12.2.2.bash
```



## Repository layout

```
Part A/
  vecadd00, vecadd01, matmult00, matmult01 sources (starter + your edits)
  Makefile

Part B/
  vecaddcpu, vecaddgpu00 (no UM), vecaddgpu01 (UM)
  collect_partB_results.sh               # optional helper
  plot_partB.py / plot_from_cli_log.py   # optional plotting helpers
  partB_results_step2.csv / partB_results_step3.csv  # produced during runs

Part C/
  convolution.cu → builds ./convolution  (or ./conv)
  Makefile
```

---

## Quick start (all parts)

```bash
# Part A
cd "Part A"
make clean
make vecadd00 vecadd01 matmult00 matmult01

# Part B
cd "../Part B"
make clean
make

# Part C
cd "../Part C"
make clean
make
```

---

## Part A — Vector Add & MatMul

### Build
```bash
cd "Part A"
make clean
make 
```

### Run (use identical problem sizes for fair comparison)

```bash
# vecadd (same "values per thread")
./vecadd00 500 
./vecadd00 1000
./vecadd00 2000

./vecadd01 500
./vecadd01 1000
./vecadd01 2000

# matmul00 (FOOTPRINT_SIZE=16) → pass 16/32/64 for 256/512/1024
./matmult00 16
./matmult00 32
./matmult00 64

# matmul01 (FOOTPRINT_SIZE=32) → pass 8/16/32 for 256/512/1024
./matmult01 8
./matmult01 16
./matmult01 32
```

### What to report
- A small table of **runtimes** and **speedups** (`speedup = T00 / T01`) for both vecadd and matmul.
- 1–2 sentences explaining any trends (e.g., why `01` does/doesn’t outperform `00` on your GPU).

> **Optional modernization (silence warnings):**
> ```bash
> sed -i 's/cudaThreadSynchronize()/cudaDeviceSynchronize()/g' vecadd.cu matmult.cu
> sed -i 's/cudaThreadExit()/cudaDeviceReset()/g' vecadd.cu
> ```

---

## Part B — CPU vs GPU (No UM vs UM)

Executables:
- `vecaddcpu` (CPU baseline)
- `vecaddgpu00` = **no Unified Memory** (device malloc/memcpy)
- `vecaddgpu01` = **Unified Memory**
- **Scenarios** (second CLI arg for GPU progs):  
  `1` = 1 block × 1 thread, `2` = 1 block × 256 threads, `3` = many blocks × 256 threads

### Build
```bash
cd "Part B"
make clean
make
```

### Run (concise runbook)
```bash
# CPU baseline (K in millions)
./vecaddcpu 1
./vecaddcpu 5
./vecaddcpu 10
./vecaddcpu 50
./vecaddcpu 100

# Step 2 — WITHOUT Unified Memory
./vecaddgpu00 1 1
./vecaddgpu00 5 1
./vecaddgpu00 10 1
./vecaddgpu00 50 1
./vecaddgpu00 100 1

./vecaddgpu00 1 2
./vecaddgpu00 5 2
./vecaddgpu00 10 2
./vecaddgpu00 50 2
./vecaddgpu00 100 2

./vecaddgpu00 1 3
./vecaddgpu00 5 3
./vecaddgpu00 10 3
./vecaddgpu00 50 3
./vecaddgpu00 100 3

# Step 3 — WITH Unified Memory
./vecaddgpu01 1 1
./vecaddgpu01 5 1
./vecaddgpu01 10 1
./vecaddgpu01 50 1
./vecaddgpu01 100 1

./vecaddgpu01 1 2
./vecaddgpu01 5 2
./vecaddgpu01 10 2
./vecaddgpu01 50 2
./vecaddgpu01 100 2

./vecaddgpu01 1 3
./vecaddgpu01 5 3
./vecaddgpu01 10 3
./vecaddgpu01 50 3
./vecaddgpu01 100 3
```

### Collect results into CSVs (recommended)

Create `partB_results_step2.csv` and `partB_results_step3.csv` with header:
```
K_millions,CPU_ms,Scenario1_ms,Scenario2_ms,Scenario3_ms
```

You can automate using the helper script (if present):
```bash
bash collect_partB_results.sh
```

Or run the concise loop that writes both CSVs:
```bash
# initialize CSVs
echo "K_millions,CPU_ms,Scenario1_ms,Scenario2_ms,Scenario3_ms" > partB_results_step2.csv
echo "K_millions,CPU_ms,Scenario1_ms,Scenario2_ms,Scenario3_ms" > partB_results_step3.csv

get_ms() { awk -F, 'NF>1{print $2; exit} END{print ""}' <<<"$1" | tr -d ' ' ; }
for K in 1 5 10 50 100; do
  cpu_ms=$(get_ms "$(./vecaddcpu $K)")
  s1=$(get_ms "$(./vecaddgpu00 $K 1)"); s2=$(get_ms "$(./vecaddgpu00 $K 2)"); s3=$(get_ms "$(./vecaddgpu00 $K 3)")
  echo "$K,$cpu_ms,$s1,$s2,$s3" >> partB_results_step2.csv
  u1=$(get_ms "$(./vecaddgpu01 $K 1)"); u2=$(get_ms "$(./vecaddgpu01 $K 2)"); u3=$(get_ms "$(./vecaddgpu01 $K 3)")
  echo "$K,$cpu_ms,$u1,$u2,$u3" >> partB_results_step3.csv
done
```

### Plot (two charts)

You may use the provided plotting utility if it exists:

```bash
# linear axes
python3 plot_partB.py partB_results_step2.csv partB_results_step3.csv

# log–log y-axis (recommended)
python3 plot_partB.py partB_results_step2.csv partB_results_step3.csv --logy
# -> partB_step2_chart.png, partB_step3_chart.png
```

If you only have a terminal transcript, use `plot_from_cli_log.py` (if provided) to parse it into CSVs/PNGs:
```bash
python3 plot_from_cli_log.py transcript.log
```

### What to submit
- **Two plots** (PNG):  
  1) **Step 2 (no UM)** — CPU + scenarios 1–3 vs **K (millions)**, y = ms  
  2) **Step 3 (UM)** — CPU + scenarios 1–3 vs **K (millions)**, y = ms  
  Use a log y-axis if scales differ widely.
- The **CSVs** used to produce the plots (nice for reproducibility).
- A short paragraph explaining the trend (allocation overhead, scenario scaling, UM comparison).

---

## Part C — Convolution (CUDA & cuDNN)

### Build & run
```bash
cd "Part C"
make clean
make
./convolution
```

**Expected output (exactly three lines):**
```
C1_<checksum>,<time_ms>
C2_<checksum>,<time_ms>
C3_<checksum>,<time_ms>
```


**Save a CSV (optional but nice):**
```bash
{ echo "Case,Checksum,Time_ms"; \
  ./convolution | sed -E 's/^C([123])_([0-9]+),([0-9.]+)/\1,\2,\3/'; } > partC_results.csv
```

---

## Submission checklist

- **Environment block** at top of your report (GPU model, CUDA version, container name).
- **Part A**
  - Runtimes for `vecadd00` vs `vecadd01` and `matmult00` vs `matmult01`
  - **Speedup** tables and 1–2 sentences of analysis
- **Part B**
  - Two charts (**no UM** and **UM**) including CPU
  - CSVs used to generate the charts
  - Brief discussion of results (allocation overhead, scenario scaling, UM comparison)
- **Part C**
  - The three-line output (or `partC_results.csv`)
  - Note that C1/C2 checksums match C3 (within FP tolerance)

---

## Troubleshooting

- **Deprecated CUDA warnings**  
  Replace in sources:  
  `cudaThreadSynchronize() → cudaDeviceSynchronize()`  
  `cudaThreadExit() → cudaDeviceReset()`

  > If you see warnings like `cudaThreadSynchronize()` is deprecated, they are harmless.

  > To modernize: `cudaThreadSynchronize() → cudaDeviceSynchronize()` and `cudaThreadExit() → cudaDeviceReset()`.

  ```bash
  sed -i 's/cudaThreadSynchronize()/cudaDeviceSynchronize()/g' <list of files separated by spaces>
  sed -i 's/cudaThreadExit()/cudaDeviceReset()/g' <list of files separated by spaces>
  ```

- **cuDNN linking errors (Part C)**  
  Ensure `-L/usr/local/cuda/lib64 -lcudnn` in the link line and `LD_LIBRARY_PATH` contains CUDA libs:
  ```bash
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
  ```

- **Singularity doesn’t see your home files**  
  Start shell with `--home $HOME` or ensure the home is bind-mounted by default.

---

## Notes

- Keep commands with **spaces in folder names** quoted, e.g., `cd "Part B"`.
- If your cluster forbids `git` on compute nodes, pull/push from login nodes only.

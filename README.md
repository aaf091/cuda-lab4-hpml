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



## Analysis

- **Part A**
  - Runtimes for `vecadd00` vs `vecadd01` and `matmult00` vs `matmult01`
    <img width="1170" height="234" alt="image" src="https://github.com/user-attachments/assets/41ded450-8dbc-46f8-baad-32f2525313e1" />

    <img width="1204" height="258" alt="image" src="https://github.com/user-attachments/assets/e339ee78-a712-4a12-a193-846b3e628e74" />

  - **Analysis**
    
    •	vecadd: The “01” variant is consistently slower , indicating the kernel is bandwidth-bound and the change reduced effective memory throughput(coalescing/ILP/occupancy         didn’t improve).
    
	  •	matmul: The “01” kernel wins at larger sizes because assigning a 2×2 tile per thread increases arithmetic intensity and data reuse, which pays off as N grows.
- **Part B**
  - Two charts (**no UM** and **UM**) including CPU
    <img width="1152" height="864" alt="image" src="https://github.com/user-attachments/assets/842af211-4401-45de-bc76-70eed28aa16d" />

    <img width="1152" height="864" alt="image" src="https://github.com/user-attachments/assets/121d82d3-6036-41d5-86e7-04da22f607e0" />

    <img width="1152" height="864" alt="image" src="https://github.com/user-attachments/assets/3cfe43a5-b9dc-4501-bc3a-35d29f27115f" />

    <img width="1152" height="864" alt="image" src="https://github.com/user-attachments/assets/f60c1598-3100-4b7d-ae6f-8711e2fee51c" />
  - End-to-end time is dominated by allocation (~380–400 ms per run) and is almost independent of K; it accounts for ~ 85–90% of total, while kernel execution is only ~55–68ms. As a result, the kernel-only curves are nearly flat and the total-time curves barely change with K. Changing the launch configuration (scenario scaling) has modest impact: scenario 3 (many blocks×256) is not consistently faster because the kernel is memory-bandwidth–bound and the extra parallelism doesn’t reduce the fixed costs; measured differences are within a few milliseconds. Unified Memory vs explicit device memory shows negligible difference on this workload—UM allocation is similar to cudaMalloc, and with linear, single-touch access there’s little page migration overhead. Takeaway: to see GPU speedups, reuse allocations (time only the compute region), consider pinned host memory + async copies, and amortize setup across multiple iterations.
- **Part C**
  - The three-line output
    
    <img width="626" height="138" alt="Screenshot 2025-11-08 at 17 04 49" src="https://github.com/user-attachments/assets/174851c0-70cc-430e-bd11-4a422087495c" />

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

# Onboard Satellite Image Triage and Downlink Optimization System

An autonomous, edge-compute payload system for Low Earth Orbit (LEO) Earth-observation satellites. The pipeline performs real-time cloud segmentation on multi-spectral imagery, penalizes geographic overlap via submodular bounding box evaluation, models entropy-driven 4-band Deflate compression, executes dynamic buffer eviction under physical memory constraints, and schedules downlinks during ground station contacts using an $\mathcal{O}(N \log N)$ greedy value-density selector or exact Dynamic Programming (DP).

---

## Architecture Overview

1. **Computer Vision Inference Subsystem:**
* Model: `TinyCloudUNet` (~482k parameters).
* **Primary Motivation (SRAM/Flash Footprint):** Flight computers and radiation-tolerant onboard systems operate with strictly bounded volatile memory shared across flight dynamics, ADCS, and core payload software. Static INT8 PTQ reduces the binary footprint by **72.68%** ($1.85\text{ MB} \to 0.50\text{ MB}$), fitting within restricted embedded SRAM/flash budgets.
* **Latency Profiling (Zero-Regression Verification):** Under a 30s optical acquisition cadence, FP32 inference ($93.99\text{ ms}$, single-thread) already operated with a **319.2x real-time margin**. Latency benchmarking was conducted strictly as a verification gate: INT8 execution achieved **67.40 ms** (1-thread) / **30.34 ms** (4-thread), preserving a **>445x real-time margin** with negligible segmentation drift ($\Delta\text{mIoU} = -0.0013$). Power reduction is expected from lower memory access traffic and integer ALU operations, but was not empirically measured on hardware.
* Inputs: 4-band multi-spectral tiles (Red, Green, Blue, NIR) at $384 \times 384$ resolution.


2. **Entropy-Driven Multi-Spectral Compression Proxy:**
* Preserves all 4 spectral bands in interleaved 8-bit DN format, compressed via Deflate (`zlib` level 6).
* Eliminates 3-channel lossy WebP distortion and couples byte footprint ($s_i$) directly to surface information entropy:
* Homogeneous / cloudy scenes: $\sim 30\text{--}60\text{ KB}$.
* Textured / high-frequency ground features: $\sim 150\text{--}250\text{ KB}$.




3. **Submodular Spatial Diversity & Priority Engine:**
* Priority equation balancing surface clarity, compressed footprint, temporal staleness, and geographic redundancy:


$$P_i(t \mid \mathcal{S}) = \frac{U_{\text{eff}}(i \mid \mathcal{S})^\alpha}{s_i^\beta} \cdot \exp(-\lambda (t - t_i))$$


* **Submodular Overlap Penalty:** Formally verified under diminishing marginal returns ($A \subseteq B \implies \Delta U(x \mid A) \ge \Delta U(x \mid B)$). For candidate tile $\tau_i$ and buffer $\mathcal{S}$:


$$U_{\text{eff}}(i \mid \mathcal{S}) = (1 - c_i) \cdot \max\left(0, 1 - \gamma \cdot \max_{j \in \mathcal{S}} \text{IoU}(B_i, B_j)\right)$$


* $\gamma = 0.8$: Discourages buffer saturation over repeated passes of identical landmarks.


4. **Onboard Memory Buffer Manager:**
* Enforces physical capacity limits ($C_{\max}$) with online marginal eviction.
* Evaluates marginal value gains: incoming tiles evict the lowest marginal-priority buffered frames only if the new candidate provides greater net utility.


5. **Downlink Contact Schedulers (Greedy vs. Exact DP):**
   * Configurable via runtime configuration (`greedy` vs `dp` / `knapsack`).
   * **Greedy Value-Density ($\rho_i = \frac{U_i}{s_i}$):** Solves in $\mathcal{O}(N \log N)$ time and $\mathcal{O}(1)$ auxiliary memory (**$1.72\text{ ms}$** at $N=2500$).
   * **Theoretical Bounds vs. Operational Domain:** Value-density greedy is unbounded in the general case ($\lim_{W \to \infty} \text{Gap} = 100\%$). A domain-constrained adversarial test ($s_0 \approx 0.51 W$) confirms an empirical collapse of **48.78%**. However, under operational LEO conditions where individual tiles are small relative to pass capacity ($\max_i(s_i) \ll W$), Greedy operates within **$<0.55\%$ of exact DP**.
   * **Exact 0/1 Knapsack (DP):** Solves discretized capacity ($1\text{ KB}$ granularity) in $\mathcal{O}(N \cdot W)$ time and space, retaining an explicit $N \times (W+1)$ boolean table to recover discrete tile IDs for radio transmission. Verified against exhaustive $2^N$ brute force at $N=10$.
   * **Isolating Algorithmic vs. Runtime Speedup:**
     * Against a standard un-jitted CPython DP implementation, Greedy appears **$\sim 900\text{x}$ faster** at $N=1600$, while CPython DP times out past $N=2000$ ($>6\text{ s}$, $>100\text{ MB}$ pointer graph).
     * Profiling against a contiguous, vectorized NumPy DP table—which was formally verified to yield identical selected sets and utility ($25.4143$ at $N=2500$)—reveals the true compute cost: **$25.89\text{ ms}$** and **$12.33\text{ MB}$**.
     * The true algorithmic speedup of Greedy over exact DP is therefore **$\sim 15\text{x}$**, not $900\text{x}$. The $900\text{x}$ figure measures CPython bytecode dispatch and pointer boxing overhead.
     * The operational rationale for choosing Greedy on flight hardware is not raw execution time (both $1.72\text{ ms}$ and $25.89\text{ ms}$ satisfy real-time pass margins), but **memory architecture**: Greedy requires zero dynamic heap allocation, avoiding the $12.33\text{ MB}$ contiguous SRAM lock required by the DP backtracking matrix.



---

## Directory Structure

```text
onboard-triage/
├── checkpoints/
│   ├── tiny_cloud_unet.onnx
│   └── tiny_cloud_unet_int8.onnx
├── data/
│   └── raw/
│       └── 38-Cloud/
├── outputs/
│   ├── scheduler_benchmark.csv
│   ├── scheduler_scaling.png
│   ├── triage_error_audit.png
│   └── sim_runs/
├── src/
│   ├── cv/
│   │   ├── dataset.py
│   │   ├── model.py
│   │   └── quantize.py
│   ├── sim/
│   │   ├── audit_errors.py
│   │   ├── benchmark_quant.py
│   │   ├── benchmark_scheduler.py
│   │   ├── pass_engine.py
│   │   ├── recorder.py
│   │   ├── scheduler.py
│   │   └── simulate.py
│   └── utils/
│       └── metrics.py
└── tests/
    ├── base_config.yaml
    └── test_submodular_penalty.py

```

---

## Requirements and Installation

* Python 3.10+
* PyTorch (CUDA 12.1 recommended)
* ONNX Runtime
* NumPy, Pandas, Matplotlib, PyYAML, Pillow, Pytest

Install dependencies:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install onnxruntime opencv-python numpy pandas matplotlib pyyaml pillow pytest

```

---

## Configuration

System settings reside in `tests/base_config.yaml`:

```yaml
simulation:
  random_seed: 42
  total_duration_s: 10800       # 3 hours (approx 2 full LEO orbits)
  time_step_s: 10               # Main simulation loop tick

orbit:
  altitude_km: 500.0
  orbital_period_s: 5700        # ~95 min LEO orbit
  passes:
    - contact_id: "PASS_01_SVALBARD"
      start_time_s: 2400
      duration_s: 300           # 5-minute LOS window
      data_rate_kbps: 4000.0    # 4 Mbps X-band downlink
    - contact_id: "PASS_02_TROLL"
      start_time_s: 7800
      duration_s: 360           # 6-minute LOS window
      data_rate_kbps: 4000.0

payload:
  capture_interval_s: 30        # Optical acquisition cadence
  daylight_duty_cycle_s: 3600   # 60 min daylight, 35 min eclipse
  image_bands: 4                # Red, Green, Blue, NIR
  tile_dim: 384                 # Spatial resolution (384x384)

storage:
  buffer_capacity_kb: 4096.0    # 4 MB physical memory ceiling (~20-30 real tiles)
  max_backlog_tiles: 100

triage:
  alpha: 1.5                    # Cloud clarity exponent
  beta: 0.25                    # Size penalty exponent
  decay_lambda: 0.0001          # Temporal staleness decay
  cloud_threshold: 0.5          # Binary segmentation cutoff

downlink:
  algorithm: "greedy"           # "greedy" or "dp"

paths:
  onnx_model: "checkpoints/tiny_cloud_unet.onnx"
  data_root: "data/raw/38-Cloud"
  manifest_csv: "data/raw/38-Cloud/training_patches_38-cloud_nonempty.csv"
  output_dir: "outputs/sim_runs"

```

---

## Verification & Execution

### 1. Mathematical Verification: Submodularity & DP Exactness

Formally verifies diminishing marginal returns and checks DP against $2^N$ combinatorial brute force:

```powershell
pytest tests/test_submodular_penalty.py

```

### 2. Quantization Zero-Regression Profiling

Profiles memory footprint reduction, segmentation mIoU, and single/multi-thread CPU runtime:

```powershell
python src/sim/benchmark_quant.py

```

### 3. Scheduler Benchmarks (Adversarial Stress Test & Scaling Sweep)

Evaluates heuristic failure on an adversarial poison-pill instance and profiles $\mathcal{O}(N \log N)$ vs. $\mathcal{O}(NW)$ scaling across $N \in [25, 2500]$:

```powershell
python src/sim/benchmark_scheduler.py

```

### 4. Qualitative Error & Albedo Failure Analysis

Isolates VNIR false-positive snow confusion and cirrus leakage edge cases:

```powershell
python src/sim/audit_errors.py

```

### 5. Multi-Orbit Flight Simulation

Runs the full 3-hour LEO simulation comparing active triage against unmanaged FIFO:

```powershell
python src/sim/simulate.py

```

---

## Empirical Benchmark Results

### 1. Model Quantization Audit (150 Tiles)

| Metric | FP32 Model | Static INT8 PTQ | Net Change |
| --- | --- | --- | --- |
| **Model Binary Size** | 1.85 MB | 0.50 MB | **-72.68% (Target Metric)** |
| **Validation mIoU** | 0.7025 | 0.7013 | -0.0013 ($\Delta\text{mIoU}$) |
| **Cloud Coverage Estimate** | 52.65% | 52.25% | 0.0039 (MAE) |
| **CPU Latency (1 Thread)** | 93.99 ms | 67.40 ms | -28.29% |
| **CPU Latency (4 Threads)** | - | 30.34 ms | -67.72% |
| **Real-Time Margin (30s cadence)** | 319.2x | 445.1x | Zero-regression verified |

### 2. Downlink Scheduler Scalability (Pass Budget = 5,120 KB, Discretization = 1 KB)

| Candidate Pool ($N$) | Greedy Latency | Vectorized DP Latency | CPython DP Latency | Peak DP Memory | Sub-Optimality Gap |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$N = 25$** | 0.026 ms | 0.42 ms | 20.60 ms | 0.12 MB | <0.01% |
| **$N = 100$** | 0.057 ms | 1.15 ms | 62.33 ms | 0.49 MB | <0.35% |
| **$N = 400$** | 0.206 ms | 4.31 ms | 217.77 ms | 1.95 MB | 0.53% |
| **$N = 1,600$** | 0.986 ms | 16.80 ms | 765.78 ms | 7.82 MB | 0.29% |
| **$N = 2,500$** | 1.717 ms | 25.89 ms | Aborted (>6s / OOM) | 12.33 MB | Safe on-orbit execution |
| **Adversarial Instance** | 0.012 ms | 0.35 ms | 0.84 ms | 0.01 MB | **48.78% (Heuristic limit)** |

> **Analysis of the 900x vs. 15x Disparity:**  
> A naive CPython DP implementation fails past $N \approx 1,600$ due to interpreter loop overhead and boxed pointer allocation across $12.8\text{M}$ cells ($>100\text{ MB}$), creating an apparent $\sim 900\text{x}$ speedup for Greedy. Re-implementing the DP solver with a contiguous, vectorized boolean array reduces execution time to **$25.89\text{ ms}$** and allocation to **$12.33\text{ MB}$**, establishing the true algorithmic speedup at **$\approx 15\text{x}$**. Both implementations were cross-verified to confirm identical 0/1 selection invariants.  
> 
> The decision to deploy Greedy over DP in flight software is driven by **deterministic memory constraints**: Greedy operates in $\mathcal{O}(1)$ auxiliary memory, eliminating the need to allocate and partition a $12.33\text{ MB}$ contiguous block in flight SRAM during ground passes.

### 3. Orbital Simulation: Ground-Truth Performance (3 Hours LEO, 4 MB Storage)

Telemetry comparison of delivered payload quality across unconstrained and bandwidth-bottlenecked passes:

#### Unconstrained Downlink Regimes (150–180 MB Pass Capacity)

Buffer-limited regime where each contact drains the buffer:

| Observable Flight Metric | Baseline (FIFO) | Triage Engine | Net Change |
| --- | --- | --- | --- |
| **Total Downlinked Tiles** | 48 | 43 | -10.4% (Higher entropy per frame) |
| **Delivered Clear Tiles (Cloud $\le$ 30%)** | 20 | 29 | **+45.0%** |
| **Mean Ground-Truth Cloud Cover** | 50.1% | 23.5% | **-26.6% absolute (-53.1% relative)** |
| **Transmitted Downlink Volume (MB)** | 7.96 MB | 7.97 MB | Downlink parity (100.1%) |
| **Internal Objective Utility** | 23.95 | 32.90 | *+37.4% (Optimizer diagnostic score)* |

#### Bottlenecked Downlink Regimes (960 KB Pass Capacity)

Bandwidth-limited regime ($\approx 25\%$ of buffer capacity deliverable per pass):

| Observable Flight Metric | Baseline (FIFO) | Triage Engine | Net Change |
| --- | --- | --- | --- |
| **Total Downlinked Tiles** | 9 | 23 | **+155.6%** |
| **Delivered Clear Tiles (Cloud $\le$ 30%)** | 3 | 19 | **+533.3% (6.3x)** |
| **Mean Ground-Truth Cloud Cover** | 57.0% | 18.6% | **-38.4% absolute (-67.4% relative)** |
| **Transmitted Downlink Volume (MB)** | 1.57 MB | 1.83 MB | Link saturated |
| **Internal Objective Utility** | 3.87 | 18.72 | *+383.7% (Optimizer diagnostic score)* |

**Note: Cumulative Utility is the internal objective function optimized by the pipeline. Operational validation is demonstrated through physical cloud cover reduction and clear tile delivery under identical downlink volumes.*

---

## Sensor Suite Failure Mode Analysis

Diagnostic error sweeps (`outputs/triage_error_audit.png`) isolated two systematic failure modes inherent to 4-band VNIR sensors:

1. **False-Positive Snow Eviction:** Snowpack and glaciated terrain exhibit high visible reflectance ($\rho \approx 0.8\text{--}0.9$) and near-zero Green-NIR NDSI, mimicking stratiform clouds and causing accidental eviction. Resolving this on the hardware level requires a Short-Wave Infrared (SWIR, $1.6\,\mu\text{m}$) channel, where snow crystals absorb heavily while water droplets scatter.
2. **Cirrus Leakage:** Thin, high-altitude cirrus clouds transmit ground radiance in VNIR, eluding detection thresholds and consuming downlink bandwidth. Adding a dedicated $1.38\,\mu\text{m}$ water vapor absorption band suppresses surface radiance, exposing ice clouds.
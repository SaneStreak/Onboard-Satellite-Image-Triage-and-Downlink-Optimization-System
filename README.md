# Onboard Satellite Image Triage and Downlink Optimization System

An autonomous, edge-compute payload system for Low Earth Orbit (LEO) Earth-observation satellites. The pipeline performs real-time cloud segmentation on multi-spectral imagery, evaluates dynamic time-decayed priority scores, manages onboard storage under physical capacity constraints, and schedules downlinks during ground-station passes using 0/1 knapsack optimization.

---

## Architecture Overview

1. **Computer Vision Inference Subsystem:**
* Model: `TinyCloudUNet` (~482k parameters).
* Format: ONNX Runtime binary (`1.85 MB`).
* Inputs: 4-band multi-spectral image tiles (Red, Green, Blue, Near-Infrared) at $384 \times 384$ resolution.
* Outputs: Spatial cloud segmentation mask and scalar cloud coverage fraction.
* Validation Accuracy: 0.8006 validation mIoU on the 38-Cloud dataset.


2. **Dynamic Utility and Priority Engine:**
* Priority equation balancing clarity, compressed storage cost, and age latency:

$$P_i(t) = \frac{U_i^\alpha}{s_i^\beta} \cdot \exp(-\lambda (t - t_i))$$


* $U_i = (1 - \text{cloud\_fraction})$: Base observation utility.
* $s_i$: Compressed byte size in memory (KB).
* $\alpha$: Exponential penalty factor for cloud contamination.
* $\beta$: Storage size penalty normalizing priority per byte.
* $\lambda$: Exponential temporal decay rate modeling observation staleness.


3. **Onboard Memory Buffer Manager:**
* Enforces strict physical SRAM/flash capacity caps ($C_{\max}$).
* Executes real-time online admission and eviction: incoming high-value captures displace older, cloud-covered, or low-utility tiles when storage saturates.


4. **Downlink Contact Scheduler:**
* Models orbital ground passes with limited duration and transmission data rates.
* Solves the 0/1 Knapsack problem via dynamic programming to extract the optimal subset of stored tiles within the contact bandwidth budget.
* Includes a greedy density-based approximation ($O(n \log n)$) for compute-constrained microcontrollers.



---

## Directory Structure

```text
onboard-triage/
├── checkpoints/
│   └── tiny_cloud_unet.onnx
├── data/
│   └── raw/
│       └── 38-Cloud/
├── outputs/
│   └── sim_runs/
│       ├── simulation_telemetry.png
│       └── parametric_sweep_results.csv
├── src/
│   ├── cv/
│   │   ├── dataset.py
│   │   └── model.py
│   ├── sim/
│   │   ├── benchmark_analysis.py
│   │   ├── pass_engine.py
│   │   ├── recorder.py
│   │   ├── scheduler.py
│   │   └── simulate.py
│   └── utils/
│       └── metrics.py
└── tests/
    └── base_config.yaml

```

---

## Requirements and Installation

* Python 3.10+
* PyTorch (CUDA 12.1 recommended)
* ONNX Runtime
* OpenCV (`opencv-python`)
* NumPy, Pandas, Matplotlib, PyYAML

Install dependencies:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install onnxruntime opencv-python numpy pandas matplotlib pyyaml rasterio pillow

```

---

## Configuration

System settings reside in `tests/base_config.yaml`:

```yaml
simulation:
  random_seed: 42
  total_duration_s: 10800       # 3 hours (approx 2 full LEO orbits)
  time_step_s: 10

orbit:
  altitude_km: 500.0
  orbital_period_s: 5700
  passes:
    - contact_id: "PASS_01_SVALBARD"
      start_time_s: 2400
      duration_s: 60
      data_rate_kbps: 128.0
    - contact_id: "PASS_02_TROLL"
      start_time_s: 7800
      duration_s: 60
      data_rate_kbps: 128.0

payload:
  capture_interval_s: 15
  daylight_duty_cycle_s: 3600
  image_bands: 4
  tile_dim: 384
  webp_quality: 90

storage:
  buffer_capacity_kb: 250.0     # Strict onboard storage capacity ceiling

triage:
  alpha: 1.5                    # Cloud penalty exponent
  beta: 0.5                     # Size penalty exponent
  decay_lambda: 0.0001          # Temporal decay constant
  cloud_threshold: 0.5

```

---

## Usage

### 1. Test the Priority Engine

Verify model loading and priority calculation:

```powershell
python src/sim/pass_engine.py

```

### 2. Test Onboard Buffer Eviction

Verify dynamic admission and priority-based replacement under storage limits:

```powershell
python src/sim/recorder.py

```

### 3. Test Pass Downlink Scheduling

Verify 0/1 knapsack tile selection against a bandwidth constraint:

```powershell
python src/sim/scheduler.py

```

### 4. Run the Multi-Pass Orbital Benchmark

Run a 3-hour LEO simulation comparing active triage against unmanaged FIFO:

```powershell
python src/sim/simulate.py

```

### 5. Run the Parameter Sweep and Telemetry Visualization

Execute an automated parameter sweep across $\alpha$ and $\lambda$, and generate telemetry plots:

```powershell
python src/sim/benchmark_analysis.py

```

---

## Benchmark Results

Under constrained orbital conditions (250 KB storage ceiling, 128 kbps pass bandwidth over two 60-second contacts):

| Metric | Baseline FIFO Buffer | Active Triage Engine | Net Change |
| --- | --- | --- | --- |
| **Delivered Clear Tiles** | 61 tiles | 86 tiles | +41.0% |
| **Mean Downlinked Cloud Cover** | 51.1% | 14.4% | -71.8% |
| **Cumulative Utility Yield** | 29.81 | 73.61 | +146.9% |
| **Downlinked Volume** | 0.49 MB | 0.48 MB | Equivalent bandwidth usage |

The triage system avoids tail-drop starvation caused by cloudy captures, purging obsolete and hazy observations to prioritize clean surface imagery prior to ground pass windows.
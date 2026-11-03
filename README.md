# FlowPilot

Agentic AI assistant for real-time lane-change decisions on NVIDIA Jetson Orin Nano.

FlowPilot is designed to ingest multi-modal sensor data (camera, LiDAR, radar, IMU), run a multi-step reasoning chain over the fused scene state, and emit structured lane-change decisions with confidence scores. Safety interlocks and a human-override controller sit outside the agent loop so they catch failures caused by the agent itself, not just environmental edge cases.

## Status

**Work in progress.** Phases 1-2 of 7 are done (see [Implementation Phases](#implementation-phases)).

- **Built and tested:** sensor ingestion adapters for camera, LiDAR, radar and IMU, temporal synchronization into a single `SceneState`, hardware smoke tests, and a latency benchmarking harness with regression comparison. 36 unit tests run without a GPU (`pytest -m "not cuda"`).
- **Designed, not yet implemented:** perception, the agent reasoning loop and the safety layer. The `perception/`, `agent/` and `safety/` packages are placeholders. The architecture, safety and latency sections below describe the target design.

---

## Architecture

```
Sensors (camera · LiDAR · radar · IMU)
        │  async streams, hardware timestamps
        ▼
  Sensor Ingestion Layer          flowpilot/sensors/
        │  synchronized SceneState @ 30 Hz
        ▼
  Perception Stack                flowpilot/perception/
  (lane detection · object tracking · EKF fusion)
        │  SceneState dataclass
        ▼
  Agentic Reasoning Engine        flowpilot/agent/
  gap analysis → risk → intent → decision synthesis
        │  LaneChangeDecision + confidence + rationale
        ▼
  Safety Orchestrator             flowpilot/safety/
  (TTC gate · confidence gate · sensor health · override)
        │  cleared decision or HOLD/ABORT
        ▼
  Actuation / Human Operator
```

The agent loop is designed to use deterministic Python tool calls as the primary decision path (gap analysis, TTC, intent prediction). A small quantized LLM running locally on the Jetson will be reserved for generating human-readable rationale on override events — keeping the hot path well within the 80 ms per-cycle latency budget.

---

## Hardware

| Component | Spec |
|-----------|------|
| SoC | NVIDIA Jetson Orin Nano |
| GPU | 1024-core Ampere (sm87), 40 TOPS |
| DLA | 2x cores, 26.2 TOPS int8 each |
| Memory | 8 GB LPDDR5 shared CPU/GPU |
| CUDA | 12.x (via JetPack 6) |

Planned deployment: models compiled to TensorRT engines (int8 for the detection backbone, fp16 for heads), with the DLA handling conv-heavy backbone passes and the GPU handling detection heads and the EKF tracker.

---

## Project Structure

```
flowpilot/
  sensors/       Ingestion adapters (camera, LiDAR, radar, IMU) + synchronizer
  perception/    (planned) Lane detection, object tracking, sensor fusion
  agent/         (planned) LangGraph state machine, reasoning tools, decision synthesis
  safety/        (planned) Safety gates, confidence thresholds, override controller
  benchmarks/    Latency and accuracy measurement harness
tests/
  sensors/       Unit + integration tests for ingestion layer
scripts/
  setup_jetson.sh   Full environment bootstrap for Jetson Orin Nano
```

---

## Getting Started

### On Jetson Orin Nano (target hardware)

```bash
git clone https://github.com/kza40/FlowPilot.git && cd FlowPilot
chmod +x scripts/setup_jetson.sh
./scripts/setup_jetson.sh
```

The script installs JetPack system packages, GStreamer, SocketCAN utilities, the NVIDIA Jetson PyTorch wheel, and all Python dependencies, then runs the hardware smoke tests.

### On a desktop CUDA machine (development)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
# Standard PyTorch (x86, CUDA 12.1):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### On a CPU-only machine (CI / no GPU)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -m "not cuda"   # skips all GPU tests
```

---

## Common Commands

```bash
make setup          # create venv and install all deps
make lint           # ruff + black check
make fmt            # auto-fix formatting
make test           # run non-CUDA tests (fast, no GPU needed)
make test-cuda      # run CUDA smoke tests (requires GPU)
make bench          # run hardware benchmark, save JSON to benchmarks/results/
make bench-compare  # compare latest two benchmark runs, flag regressions
```

---

## Safety Design (planned)

Safety interlocks will run as a separate async task, independent of the agent, so they catch agent bugs and not just environmental edge cases.

| Gate | Trigger | Action |
|------|---------|--------|
| Confidence | Decision confidence < 0.75 | HOLD, request override |
| TTC | Time-to-collision < 2.5 s in target lane | ABORT |
| Sensor health | Any sensor > 2 frames stale | HOLD |
| Velocity envelope | Speed < 20 km/h or > 120 km/h | HOLD |
| Human override | Operator signal (keyboard / CAN / REST) | Immediate HOLD + audit log |

Every decision, including gate results, confidence scores and agent rationale, will be written to an append-only audit log.

---

## Latency Targets

These are design targets; end-to-end latency will be measured once the perception and agent layers exist.

| Stage | Target |
|-------|--------|
| Sensor to SceneState | <= 30 ms |
| SceneState to LaneChangeDecision | <= 50 ms |
| Full pipeline end-to-end | <= 80 ms |

Latency is measured and tracked at the **99th percentile** (not the mean) because thermal throttling on embedded hardware produces tail spikes that the mean hides.

---

## Implementation Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1 - Foundation | Complete | Project skeleton, pyproject.toml, hardware smoke tests, benchmark harness |
| 2 - Sensor Ingestion | Complete | Camera (NVMM GStreamer), LiDAR (Ouster UDP), radar (SocketCAN), IMU (serial) + temporal sync |
| 3 - Perception | Planned | TensorRT lane detection, YOLOv8 + ByteTrack, LiDAR clustering, EKF fusion |
| 4 - Agent | Planned | LangGraph state machine, reasoning tool chain, decision synthesis |
| 5 - Safety | Planned | Interlock layer, override controller, append-only audit log |
| 6 - Benchmarking | Planned | CARLA simulation, accuracy vs. latency Pareto sweep |
| 7 - Hardening | Planned | TensorRT + DLA offload, thermal profiling, graceful degradation |

# AGENTS.md — Ultra-Detailed Project Context & Agent Guide (v2025-07-25)

## 🚨 PROJECT: Apple Metal Orchard (FlashAttention on Mac, Custom PyTorch Build) 🚨

**Objective:**
Implement, integrate, and benchmark a custom Metal (Apple Silicon) FlashAttention kernel in PyTorch, with reproducible, debuggable, agent-friendly code + CI. Critical context for any advanced agent, LLM, or developer working at the top 0.01% of systems/ML performance.

## 1. 🔥 Repo & Directory Layout

* **orchard/** (Main repo, Python/Swift/C++)

  * `orchard_ops/`          — Custom kernels, C++/Metal/Swift interface code

    * `enable_flash.py`     — Monkeypatch for HuggingFace GPT-2 to enable custom Metal FlashAttention
    * `mps/flash_attn.mm`   — Metal FlashAttention (MPS) kernel dispatch, C++/ObjC++
    * `CMakeLists.txt`      — Kernel ops CMake config
  * `kernels_inventory.md`  — Kernel inventory, doc
  * `tests/`                — Pytest suite
  * `benchmarks/`           — Run scripts, benchmarking harness
  * `data/wikitext-2/`      — Training data
  * `run_epoch.sh`          — Main training script
  * `.gitmodules`           — Submodule config
  * `build_custom_torch.sh` — Script to build/patch custom PyTorch wheel (critical for agent workflows)
  * `AGENTS.md`             — YOU ARE HERE: full agent/project context
  * `.gitignore`            — Ignore build artefacts, venv, caches, etc.

* **pytorch/** *(submodule: [git@github.com](mailto:git@github.com)\:Ian-Reitsma/pytorch.git)*

  * **Custom fork** (not vanilla upstream)
  * Main codebase for all C++/ATen/Python core PyTorch
  * Local modifications:

    * `aten/src/ATen/native/mps/Attention.mm` — New: FlashAttention backward/interop kernel
    * `tools/shared/_utils_internal.py` — Patch for c10/std::optional type interop
    * Possibly changes to CMakeLists.txt, native\_functions.yaml (see below)
  * All submodules recursively initialized (flash-attn, cutlass, etc.)
  * Branch typically: `main` (may be pinned at specific commit)

* **third\_party/metal-flash-attention/** (submodule, from Philip Turner)

  * Reference Metal FlashAttention kernel (used for baseline and integration)

## 2. 🏗️ Build, Patch, and Environment Flow

### Core Build/Setup (see `build_custom_torch.sh`):

**Main workflow:**

1. Clone **orchard** (Apple-Metal-Orchard) and all submodules recursively:

   * `git clone --recurse-submodules git@github.com:Ian-Reitsma/Apple-Metal-Orchard.git`
2. (If not already):

   * `cd orchard`
   * `git submodule update --init --recursive`
   * Confirms submodules pytorch/ and third\_party/metal-flash-attention/ present
3. Setup Python environment:

   * `python3 -m venv orchard-env`
   * `source orchard-env/bin/activate`
   * `pip install -r requirements_lock.txt` *(or use build script)*
4. Build custom PyTorch wheel:

   * `bash build_custom_torch.sh`

     * Upgrades pip/wheel/ninja/cmake
     * Cleans prior builds
     * Enters `pytorch/`, builds wheel via `pip wheel --no-deps -w ../dist .`
     * Installs built torch wheel
     * Installs compatible torchvision/torchaudio against this build
     * Ensures patched PyTorch is visible to all Python scripts

**Note:** You MUST use the custom built torch wheel from the submodule. DO NOT pip install torch from PyPI — agent flow depends on custom backend (build is \~20–30 min on Mac).

### Kernel/Extension Build (Apple ARM only):

* `swift build -c release -Xswiftc -Ounchecked` — in repo root (if using Swift/Metal)
* Copies built .dylib to `kernel_lib/flashattn/`
* Not required for vanilla CPU-only agent eval, but MANDATORY for all MPS/Metal workflows

## 3. 🧬 What’s Custom/Pinned/Broken (Known Issues)

* **PyTorch fork:** All changes must be made against the Ian-Reitsma/pytorch.git fork, not upstream. See notes in AGENTS.md, git log, and submodule commit hash.
* **Monkeypatching:** `orchard_ops/enable_flash.py` must be called to patch in Metal FlashAttention kernels.
* **C++/ObjC++ glue:** Native kernel interface in `flash_attn.mm` — not just Python, must diagnose at C++ level
* **Optional Interop:** `_utils_internal.py` patch exists to hack through c10::optional/std::optional type mismatch
* **CMake:** If adding or moving kernels, update ALL relevant CMakeLists.txt files, or build may not see source changes
* **Submodules:** Make sure **all submodules are initialized** — builds break if flash-attn or cutlass is missing!
* **Testing:** Use pytest on test suite; tests will fail if MPS/Metal unavailable, or if kernel signature changes (see test\_flash\_patch.py)
* **Build time:** Full PyTorch build is huge (5–10GB disk, 20–40 min wall clock)
* **Agent/LLM:** Must be able to index both the repo and ALL submodules for any chance of code navigation/diagnosis.

## 4. 🛠️ AGENT ACTION PLAN / DIAG FLOW

### **DO THIS IMMEDIATELY ON LOAD:**

* Verify full repo tree including submodules is present (at least: `orchard/`, `orchard/pytorch/`, `orchard/third_party/metal-flash-attention/`)
* Check that the following files exist in their stated paths:

  * `orchard_ops/mps/flash_attn.mm`
  * `orchard_ops/enable_flash.py`
  * `pytorch/aten/src/ATen/native/mps/Attention.mm`
  * `pytorch/tools/shared/_utils_internal.py`
  * `orchard/build_custom_torch.sh`
  * `.gitmodules` is correct, with pytorch + flash-attn entries
* **Check submodule HEADs:** Run `git submodule status` — confirm that pytorch points to the correct fork/commit. Any drift = build break.
* **Confirm virtualenv is using custom built torch:** `python -c 'import torch; print(torch.__file__)'` — must be orchard-env site-packages, and show the correct commit/hash.
* Run: `pip list | grep torch` — verify torch, torchvision, torchaudio are installed from correct wheels/branches (not PyPI default!)
* Confirm MPS/Metal availability: `python -c 'import torch; print(torch.backends.mps.is_available())'`
* If using Swift/Metal: ensure `libflash_attn.dylib` is present in `kernel_lib/flashattn/`.
* All pytest tests should import torch and patch in custom kernel (see `test_flash_patch.py`)

### **IF DIAGNOSING BUILD FAILURES:**

1. Re-run build with `bash build_custom_torch.sh` (preferred for deterministic env)
2. Capture ALL output logs (`tee build_custom_torch.log`)
3. If C++ build errors, verify CMakeLists.txt (ATen/native, orchard\_ops) include new/renamed files (see current CMakeLists for example)
4. For symbol/linker errors, verify all API signatures in:

   * `native_functions.yaml`
   * ATen/native/mps/Attention.mm
   * orchard\_ops/mps/flash\_attn.mm
   * header interfaces (flash\_attn.h, etc)
5. For Python/C++ API interop bugs, check for c10::optional/std::optional mismatches (see `_utils_internal.py` hack)
6. If torch fails to import or kernel is missing at runtime, check that monkeypatch (enable\_flash.py) is executed BEFORE model/optimizer is instantiated.

### **AGENT-LEVEL DEBUGGING (0.01% rigor):**

* You MUST treat the pytorch/ submodule as code you can modify, not as a black box. You have full write/commit access in this fork.
* All patches/changes must be reproducible via git diff, and documented in AGENTS.md, commit logs, and possibly as a patch file in the repo root.
* Test for both CPU and MPS/Metal paths — ensure fallback to PyTorch baseline if Metal unavailable.
* Keep ALL edits isolated, small, and cross-documented (every C++/Python/CMake change should be paired with a comment in AGENTS.md and/or commit message).
* If you add a kernel, update the kernel inventory markdown and AGENTS.md — all interface, signature, and API shape changes MUST be documented here.
* If you break the build, bisect or revert — all breakage must be described in AGENTS.md for historical traceability.

## 5. 🗒️ FILE/CHANGE INVENTORY (Last Known Good)

**orchard/.gitmodules**

```ini
[submodule "third_party/metal-flash-attention"]
    path = third_party/metal-flash-attention
    url = https://github.com/philipturner/metal-flash-attention
[submodule "pytorch"]
    path = pytorch
    url = https://github.com/Ian-Reitsma/pytorch.git
```

**Custom files/changes (as of 2025-07-25):**

* orchard\_ops/enable\_flash.py         # HuggingFace monkeypatch for custom kernel
* orchard\_ops/mps/flash\_attn.mm      # Metal kernel C++/ObjC++ glue (fused forward/backward, etc)
* orchard\_ops/CMakeLists.txt         # Must include all .mm/.h files for build
* orchard/build\_custom\_torch.sh      # Custom build script for deterministic PyTorch wheel
* tests/test\_flash\_patch.py          # Pytest runner for enable\_flash.py integration
* pytorch/aten/src/ATen/native/mps/Attention.mm      # New native kernel
* pytorch/tools/shared/\_utils\_internal.py            # c10::optional hack (interop fix)
* pytorch/aten/src/ATen/native/native\_functions.yaml # (if changed — confirm kernel signature matches all callsites)
* (If changed) CMakeLists.txt (ATen/native, orchard\_ops, etc)

## 6. 🧑‍💻 AGENT WORKFLOW CHECKLIST (FOR TOP 0.01% AGENTS)

1. **Repo/Env Boot:**

   * `git clone --recurse-submodules ...`
   * `python3 -m venv orchard-env; source orchard-env/bin/activate`
   * `bash build_custom_torch.sh`
2. **Edit/Debug:**

   * Edit kernel, patch Python/Cpp, edit CMake as needed
   * Re-run build script for every change
   * Always run pytest before/after to catch regressions
3. **Push/Pull/PR:**

   * All submodule updates must be committed & pushed (see submodule docs)
   * Document all changes, pin hashes, update AGENTS.md
4. **CI/CD/Agent Infra:**

   * If agent infra/CI can’t index the pytorch submodule (due to size), copy-paste critical changes into orchard repo root as .patch or .diff files
   * Keep AGENTS.md and kernels\_inventory.md updated with every interface/tuning/API change — this is your single source of truth for what the agent needs to know.

## 7. 🚨 KNOWN BLOCKERS / ACTIVE ISSUES (AS OF 2025-07-25)

* Build intermittently fails due to C++/CMake file path or optional type mismatch (std::optional vs c10::optional)
* Agent must be prepared to fully diagnose, edit, and rebuild both Python and C++/ObjC++ source trees
* Pinned submodule commit hashes are the only reliable way to sync versions — always check/update after pulling or merging
* If using agents/LLMs with project context, you **must** enable submodule scanning or provide all custom kernel source and patch files in the root project (as a backup)

## 8. 🧭 FINAL NOTE FOR AGENTS

If you are an advanced AI agent, LLM, or systems engineer, treat this document as the law: **all source, kernel, patch, build, and config knowledge must be extracted from AGENTS.md, commit logs, and the current tree.**

If you are unable to resolve build or integration issues, cross-reference all C++/Python/CMake/API boundaries, and document any observed mismatches or failure points. **Do not proceed until all repo/submodule/env states are verified.**

For any major changes, update this file immediately, increment the date/version, and notify the next human/agent of the delta.

---

Further Context:

## 1. Executive Summary

This repo is a **first-principles, Metal-optimized implementation of FlashAttention for Apple Silicon** (M1/M2/M3), aimed at order-of-magnitude speedups for LLM training/fine-tuning on consumer Macs.  
**The core challenge:** Patch and validate all FlashAttention logic (forward/backward) for the MPS backend, integrated with custom Metal/Swift kernels, while retaining drop-in compatibility with PyTorch’s modern dispatcher and HuggingFace APIs.  
**Current status:** Build *runs but is nonfunctional*, backward kernel not yet hooked/working; custom PyTorch submodule and all C++/Metal/Swift/CMake infra are custom-forked and heavily patched.

## 2. Repo Topology
repo-root/
.gitmodules # Submodule: pytorch (custom fork!), metal-flash-attention
AGENTS.md # (this file)
build_custom_torch.sh # Canonical script to build/install custom torch/vision/audio
CMakeLists.txt # Project CMake, ties kernels to Python modules
orchard_ops/ # Main Python bindings + custom ops loader
enable_flash.py # Entry for monkeypatching/dispatch hook
mps/
flash_attn.mm # Custom MPS kernel: FlashAttention FWD/BWD C++/ObjC++ logic
flash_attn.h # Headers for kernel API (all params defined here)
pytorch/ # Submodule: custom fork, https://github.com/Ian-Reitsma/pytorch.git
... # Fully local, includes custom C++/CMake edits
third_party/metal-flash-attention/ # Reference Metal kernels, not directly used but as source
data/wikitext-2/ # Example corpus
tests/ # pytest suite for all ops, shape/dtype/fwd-bwd correctness
dist/ # Wheels land here (if built)
kernel_lib/ # Output for Swift/Metal dylibs
requirements_lock.txt # Exact env lock
orchard-env/ # venv for project isolation
native_functions.yaml # Edits for custom op registration (PyTorch ATen YAML)
## 3. Submodules and Forks

- `pytorch/` is a **fully custom fork**—do not assume it’s stock PyTorch!
  - Remote: `git@github.com:Ian-Reitsma/pytorch.git`
  - All FlashAttention and optional/c10 interface logic are patched here.
  - **Agents must clone the submodule!**  
    **Size:** ~5–7GB on disk; agents must be able to handle/build this scale.
- `third_party/metal-flash-attention` — Reference kernels from Philip Turner (source only).

## 4. Build & Setup

**Primary build scripts/workflow:**
- `build_custom_torch.sh`: Canonical, stepwise script to
  1. Install/upgrade pip, wheel, ninja, cmake
  2. Initialize all submodules recursively (this is *required*, else pytorch/* is empty)
  3. Clean PyTorch build tree (`python setup.py clean`)
  4. Build a local PyTorch wheel (places in `dist/`)
  5. Installs the wheel, then builds vision/audio against *this* torch version (no-deps)
  6. Fails if C++/kernel build has errors (exit on fail)
- **Dependencies**: Python 3.11+ (venv enforced), full clang/llvm toolchain (Xcode on Mac), `ninja`, `cmake`, up-to-date pip.
- **ENV quirks:**  
  - `orchard-env/` venv required; *do not install globally*.
  - On M1/M2: set `ON_MAC_ARM=1` (auto-detected in scripts).
  - Fails on Linux/CI for Metal/Swift builds; CPU fallback only.

## 5. Key Files and Their Roles

- `orchard_ops/mps/flash_attn.mm` — C++/ObjC++: FWD/BWD logic for FlashAttention. Calls native Metal when available, else falls back to stock PyTorch dispatcher.
  - **FWD:** Verified to call native when enabled.
  - **BWD:** *Stub/partial*: not yet fully hooked; likely falling back to default C++/CPU.
- `orchard_ops/enable_flash.py` — Entrypoint for monkeypatching all GPT-2 (HuggingFace) attention modules to use our custom Fused FlashAttention (by patching the forward/backward calls at runtime).
- `pytorch/aten/src/ATen/native/mps/Attention.mm` — Custom kernel for MPS attention (copied/modified, see patch history).
- `native_functions.yaml` — PyTorch ATen YAML; must register all new ops (esp. backward) here for dispatcher to recognize and bind at runtime.
- `tests/test_flash_patch.py`, `test_flash_attn_shape.py` — Pytest: shape/dtype correctness, monkeypatch coverage. (Skip if torch or MPS unavailable.)

## 6. Workflow: Agent/Dev Onboarding

**A. Full Project Checkout:**
1. `git clone --recursive git@github.com:Ian-Reitsma/Apple-Metal-Orchard.git`
2. `cd Apple-Metal-Orchard`
3. `git submodule update --init --recursive` — _Do NOT skip this!_ Clones both `pytorch/` and `metal-flash-attention/`
   - If `pytorch/` is missing: Check `.gitmodules` points to `Ian-Reitsma/pytorch.git`

**B. Build PyTorch**
1. Run `bash build_custom_torch.sh`
   - If it fails, check for missing toolchain, bad Python version, or lack of disk space (PyTorch build is large).
   - Wheel should appear in `dist/`, installed in venv.

**C. Install and Build Vision/Audio**
- Both must match torch wheel build (see script).
- If audio fails to build: check for `cmake` errors (see below).

**D. Test & Debug**
1. Activate env: `source orchard-env/bin/activate`
2. Run tests: `pytest tests/`
   - Use `pytest -v` for verbose; ensure `orchard_ops.enable_flash` patch applied.
   - Most tests are skipped if torch/MPS/Metal is missing.

**E. Kernel Build (Apple Silicon only)**
- `swift build -c release -Xswiftc -Ounchecked`
- Copies Metal kernel dylib to `kernel_lib/flashattn/`

**F. PyTorch Registration**
- New ops in `native_functions.yaml` MUST be re-registered, else dispatcher fails at runtime.

## 7. Current Issues / Blockers

- **Backward kernel (`scaled_dot_product_attention_backward_mps`) not registered or found in dispatcher.**
  - `nm build/lib/libtorch_cpu.dylib | grep scaled_dot_product_attention_backward_mps` shows missing symbol.
  - `test_flash_patch.py` skips/fails all BWD calls.
- **C10/optional interop hacks in `tools/shared/_utils_internal.py`** — dirty hack to bridge c10/std::optional interface at runtime. Needs refactor once kernel is working.
- **Vision/audio install fails if cmake/c++ toolchain or matching torch version not found.**
- **Agent/Codex will need to FULLY clone all submodules** (~7GB) and have disk/ram to build PyTorch from source.
- **Build time:** 10–30min on M1/M2. Agents must not time out mid-build.
- **Tests will skip on non-Mac/non-MPS systems; code analysis can proceed but not hardware validation.**

## 8. File/Function Mapping and Responsibilities

| File                                      | Purpose / Functions                                      | Agent Task / Notes                                   |
|--------------------------------------------|----------------------------------------------------------|------------------------------------------------------|
| `orchard_ops/mps/flash_attn.mm`           | FlashAttention FWD/BWD C++/Metal kernel                  | Ensure correct entry/exit for both paths             |
| `orchard_ops/mps/flash_attn.h`            | API for Metal kernels; function params/types              | Any new tuning/params MUST be reflected here         |
| `enable_flash.py`                         | Python monkeypatch/dispatcher hook                       | Should patch all relevant HF/torch attention modules |
| `pytorch/aten/src/ATen/native/mps/Attention.mm` | PyTorch MPS kernel, reference for custom ops        | Must match/override FWD/BWD signatures               |
| `native_functions.yaml`                    | Op registration; add/remove here for PyTorch dispatcher  | Missing/incorrect entries will break everything      |
| `build_custom_torch.sh`                   | Canonical build/install/test script                      | Run top to bottom, log any errors                    |
| `tests/test_flash_patch.py`                | Pytest; validates FlashAttention at high level           | Monkeypatch check + shape/dtype correctness          |

## 9. Developer/Agent “Do Not Assume” List

- *Never* assume any file in `pytorch/` or `orchard_ops/` is vanilla or unmodified—**always diff against upstream**.
- Always check that **.gitmodules** and **submodule paths** are valid and initialized; `pytorch/` may be missing if not cloned recursively.
- **CMake and Python/C++ toolchain must be system-matched:** Python version, clang/llvm (esp. on Mac), pip, etc.
- If agent cannot build Metal/Swift kernels (not on Mac ARM), **skip kernel build** and focus on code/logic review only.
- **Agent must log all errors with full output**—do not truncate, as most build failures are in >20 lines of output.

## 10. Gotchas / Troubleshooting

- **If a new op is not found:** Double-check `native_functions.yaml` and confirm correct signature, registration, and dispatcher call in both C++/Python.
- **If wheel build fails:** Check CMake output, especially for missing deps, linker errors, or missing submodules.
- **If build is too slow:** PyTorch source builds are huge (multi-GB); agents may want to cache build artefacts, but MUST recompile for C++ header/source changes.
- **If agent times out:** Try partial build/test (e.g., only orchard_ops/ first), then scale up to full build.
- **Python/C++ version mismatches are fatal:** venv isolates this, but all system deps must be consistent.

## 11. Summary Checklist for Agents

1. Clone with `--recursive`, verify submodules present (esp. `pytorch/`).
2. Run `build_custom_torch.sh`, do NOT skip step.
3. On build fail: check Python, CMake, clang, RAM, disk, missing submodules.
4. Run all pytest (`pytest -v tests/`), check shape, dtype, monkeypatching.
5. On MPS errors: inspect FWD/BWD dispatch, `flash_attn.mm`, and dispatcher logs.
6. If new kernel not showing up, re-register in `native_functions.yaml`, re-run full build.
7. Log ALL output; upload as agent artifacts for review.

---

**If you are an LLM agent or elite human dev, follow these steps in order, provide logs, and do NOT skip context from this file.**  
If any step or submodule is missing, STOP and ask for clarification with logs.  
**This file must be updated with every major change or fix.**

---


# Orchard × Metal FlashAttention — **Agents Guide**

> **Read me first.** This document is the *single source of truth* for autonomous code‑agents (Codex, GPT‑4o, et al.) that operate on this repository.  It tells you *where everything lives*, *how to build it*, *what is already done*, and *what is broken*.  Nothing here is optional.  Follow it verbatim unless you have a provably better plan.

---

## 1  📂 Repository topology

```text
.
├── orchard_ops/                 # Python + C++/ObjC++ runtime glue
│   ├── enable_flash.py          # Loads the Metal dylib + registers autograd op
│   ├── flash_attn_function.py   # Autograd Function wrapper
│   └── mps/
│       ├── flash_attn.h         # C++ API header
│       └── flash_attn.mm        # ObjC++ kernel shim → dispatcher
├── kernel_lib/flashattn/        # build target for libflash_attn.dylib (Swift/Metal)
├── benchmarks/                  # end‑to‑end perf harness + monkey‑patch
│   └── orchard_patch_flash.py   # GPT‑2 monkey‑patch for debug logging
├── tests/                       # pytest smoke tests
│   └── test_flash_attn_shape.py
├── build_custom_torch.sh        # one‑shot script → wheel w/ custom ATen changes
├── setup_orchard.sh (WIP)       # sets up venv, downloads dataset, builds dylib
├── .gitmodules                  # submodule declarations (see below)
└── pytorch/ (submodule)         # **forked** @Ian‑Reitsma/pytorch.git
    ├── aten/src/ATen/native/mps/Attention.mm     # NEW (fwd stub + linker symbol)
    └── tools/shared/_utils_internal.py           # Hack: std::optional ↔︎ c10::optional
```

### 1.1  Submodules

| Path                                | Purpose                                         |
| ----------------------------------- | ----------------------------------------------- |
| `third_party/metal-flash-attention` | Philip Turner’s reference Swift + Metal kernels |
| `pytorch`                           | Fork with Orchard patches (see ↑)               |

Initialise with:

```bash
git submodule update --init --recursive
```

---

## 2  🛠️ Build matrix

| Host               | Supported? | Notes                                               |
| ------------------ | ---------- | --------------------------------------------------- |
| **macOS 14 arm64** | ✅          | Primary target.  MPS + Metal available.             |
| macOS (Intel)      | ⚠️         | No Metal; CPU/MPS fallback only.                    |
| Linux (x86\_64)    | ⚠️         | Build PyTorch wheel for CI, but Metal features off. |

Environment probes are inside `setup_orchard.sh` → sets `ON_MAC_ARM`.

---

## 3  🔑 Environment variables

| Variable                 | Default | Description                                      |
| ------------------------ | ------- | ------------------------------------------------ |
| `USE_FLASH_ATTN`         | `1`     | Enable monkey‑patch + Metal kernels.             |
| `USE_BF16`               | `1`     | Model dtype in benchmark.                        |
| `ORCHARD_DEBUG_FLASHATN` | `0`     | If `1`, fall back to PyTorch backward for CI.    |
| `FLASHATTN_LOG_LEVEL`    | `1`‑`2` | Kernel call logger verbosity (see patch script). |
| `PROGRESS_SEC`           | `10`    | Seconds between training tick prints.            |

Set in shell or export inside `setup_orchard.sh`.

---

## 4  ♻️ Full build flow (local dev)

> **Make sure Xcode command‑line tools + Swift 5.10 are installed.**

```bash
./setup_orchard.sh          # create venv, install deps, build Swift Metal dylib
source orchard-env/bin/activate

# optional: rebuild Torch with custom ATen patches
chmod +x build_custom_torch.sh
./build_custom_torch.sh     # produces dist/torch-*.whl and installs it

# run smoke tests
pytest -q tests/test_flash_attn_shape.py
```

`build_custom_torch.sh` performs:

1. `pip wheel` of the **submodule** at revision `heads/main` (custom files already staged).
2. Installs wheel in editable mode.
3. Builds torchvision / torchaudio from source *against that wheel*.

Wheel artefact ends in `dist/` — CI caches this to avoid 20‑min rebuilds.

---

## 5  🩹 Current patches inside submodule `pytorch`

| File                                    | Delta                                                                                       |
| --------------------------------------- | ------------------------------------------------------------------------------------------- |
| `aten/src/ATen/native/mps/Attention.mm` | Adds stub for `scaled_dot_product_attention_backward_mps`, exports symbol the linker wants. |
| `tools/shared/_utils_internal.py`       | Compile‑time shim: exposes `std::optional` as `c10::optional` for newer ATen macros.        |

*No other source files are modified.  All heavy `CMakeLists.txt` diffs are *outside* submodule (Orchard side).*
Check with:

```bash
cd pytorch && git diff origin/main --stat
```

---

## 6  ❗ Open issues for agents

1. **Link‑time undefined symbol** `scaled_dot_product_attention_backward_mps` when building Torch on macOS arm64.

   * The stub is present, but object may not be compiled into `libtorch_cpu.dylib`.
   * Hypothesis: the `file(GLOB mps_mm ...)` in ATen’s root `CMakeLists.txt` does **not** catch `native/mps/Attention.mm` because of path or exclusion macros.
   * *Task*: patch CMake so new `.mm` file is part of `ATen_MPS_SRCS`.

2. **torchaudio fails** because `cmake` Python wheel absent inside venv when pip builds it.

   * *Task*: ensure `pip install cmake ninja` before building torchaudio.

3. **Runtime**: verify that forward path routes through `flash_attn_mps._flash_attn_fwd` and that fallback logs are clean.

4. **CI optimisation**: cache `dist/torch-*.whl` and dylib SHA256 to avoid 20‑minute compile.

---

## 7  🧪 Testing recipes

```bash
# 1. Pure unit test (shape only)
pytest -q tests/test_flash_function.py

# 2. Autograd + monkey‑patch
pytest -q tests/test_flash_attn_shape.py

# 3. Full GPT‑2 benchmark (~15 min quick‑check)
QUICK_CHECK=1 USE_FLASH_ATTN=1 ./run_epoch.sh --tag ref_no_flash
```

Log files live in `runs/<timestamp>_*.log`.  Kernel call counter logs to `/tmp/flashattn_kernel_calls.log`.

---

## 8  🚀 Agent workflow checklist

1. `git clone https://github.com/Ian-Reitsma/Apple-Metal-Orchard.git orchard && cd orchard`
2. `git submodule update --init --recursive`
3. **Mac ARM only:** install Xcode & Metal SDK.
4. Run `./setup_orchard.sh`; confirm *✓ FlashAttention dylib SHA256* line.
5. If linker error persists, open `pytorch/aten/.../CMakeLists.txt` and ensure `Attention.mm` is in `ATen_MPS_SRCS`.
6. Re‑run `build_custom_torch.sh`; validate `nm build/lib/libtorch_cpu.dylib | grep scaled_dot_product_attention_backward_mps` shows a `T` symbol.
7. Run `pytest`; expect green.

---

## 9  📌 Conventions

* **Do not** commit large build artefacts (>5 MB) — use Git LFS or CI cache.
* All new C++/ObjC++ **must compile with `/std:c++20`**.
* Keep public Python API stable; breaking changes require bumping `ORCHARD_FLASH_ATTN_API_LEVEL`.
* Update this file *every time* you add new environment flags, scripts, or submodules.

---

## 10  🔄 Submodule commit workflow

The root repository records exact commits for its submodules. When changing files
inside `submodules/metal-tensor`, push that submodule before committing the root
repository:

```bash
cd submodules/metal-tensor
# edit files, then commit
git commit -am "Describe change"
git push origin agent/codex

cd ../..
# record the new submodule commit in the root repo
git add submodules/metal-tensor
git commit -m "Update metal-tensor submodule pointer"
git push origin agent/codex
```

If the submodule commit isn't on the remote, `git submodule update` will fail for
others cloning the repository. After pushing, use `git pull --recurse-submodules`
followed by `git submodule update --init --recursive` before making further edits
to keep everything in sync.

---

End of AGENTS.md

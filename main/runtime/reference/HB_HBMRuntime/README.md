# HB_HBMRuntime reference

Read-only snapshot of the D-Robotics `hbm_runtime` Python wrapper source,
copied from the S600 deploy machine `/usr/hobot/lib/hbm_runtime/`.

## Contents

| File | Purpose |
|---|---|
| `HB_HBMRuntime.cc` | Canonical hbDNN/hbUCP C++ wrapper (1755 lines) |
| `HB_RuntimeUtils.cc` | numpy/array helpers |
| `HBMRuntimeBinding.cc` | pybind11 binding |
| `HB_HBMRuntime.hpp` | C++ class declaration |
| `HB_RuntimeUtils.hpp` | Helper declarations |
| `CMakeLists.txt` | D-Robotics's build config for the Python `.so` |
| `README.md` | D-Robotics's own README |

## Usage

Reference for the LA C++ host runtime. The canonical BPU inference flow
is `HB_HBMRuntime.cc::InferSingleModel` (~line 625):

```c
hbDNNInferV2(&task, output_tensors, input_tensors, dnn_handle);
hbUCPSchedParam sched{};
HB_UCP_INITIALIZE_SCHED_PARAM(&sched);
sched.backend = GetBPUCoreMaskForModel(name, bpu_cores);
hbUCPSubmitTask(task, &sched);
hbUCPWaitTaskDone(task, 0);
```

`main/runtime/src/hbm_session.cpp::Graph::Execute` mirrors this flow
(see `docs/KNOWN_ISSUES.md` #016).

## Notes

- These files are not compiled into LA. Borrowed code lives under
  `main/runtime/src/` with renamed identifiers (`HbmSession`, `Graph`,
  `Execute`), per the project's code-isolation rule.
- Source remains at `/usr/hobot/lib/hbm_runtime/` on the S600 host.

// Copyright (c) 2026 LiuAnclouds / Kangjie Xu / D-Robotics
//
// Implementation of HbmSession / Graph — thin C++ wrapper over the hbDNN /
// hbUCP C API shipped by the D-Robotics hobot-dnn deb on S600 (and the
// arm64 hbdk4-runtime package on the build host).
//
// The C flow for one inference is:
//   1) hbDNNInitializeFromFiles  -> packed_handle
//   2) hbDNNGetModelNameList     -> graph_names
//   3) hbDNNGetModelHandle       -> graph_handle (per name)
//   4) hbDNNGetInputTensorProperties / hbDNNGetOutputTensorProperties
//   5) for each input:  hbUCPMallocCached + memcpy + hbUCPMemFlush(CLEAN)
//   6) for each output: hbUCPMallocCached
//   7) hbDNNInferV2               -> task_handle
//   8) hbUCPSubmitTask            -> kicks off the BPU
//   9) hbUCPWaitTaskDone          -> block until done
//  10) hbUCPMemFlush(INVALIDATE) on each output  -> pull results from BPU
//  11) hbUCPReleaseTask           -> free the task handle
//  12) hbUCPFree                  -> free input / output buffers

#include "locateanything_runtime/hbm_session.hpp"

#include <cstring>
#include <iostream>

extern "C" {
#include "hobot/dnn/hb_dnn.h"
#include "hobot/hb_ucp.h"
#include "hobot/hb_ucp_sys.h"
}

namespace locateanything_runtime {

namespace {

// Table mirror of hb_dnn.h HB_DNN_TENSOR_TYPE_* enums. Keeping them as plain
// ints here lets hbm_session.hpp forward-declare without including hb_dnn.h.
constexpr int32_t kTypeS4 = 0;
constexpr int32_t kTypeU4 = 1;
constexpr int32_t kTypeS8 = 2;
constexpr int32_t kTypeU8 = 3;
constexpr int32_t kTypeF16 = 4;
constexpr int32_t kTypeS16 = 5;
constexpr int32_t kTypeU16 = 6;
constexpr int32_t kTypeF32 = 7;
constexpr int32_t kTypeS32 = 8;
constexpr int32_t kTypeU32 = 9;
constexpr int32_t kTypeF64 = 10;
constexpr int32_t kTypeS64 = 11;
constexpr int32_t kTypeU64 = 12;
constexpr int32_t kTypeBool8 = 13;

int32_t ElementBytesForType(int32_t dtype) {
  switch (dtype) {
    case kTypeS4:
    case kTypeU4:
    case kTypeBool8:  // actually 1 byte
    case kTypeS8:
    case kTypeU8: return 1;
    case kTypeF16:
    case kTypeS16:
    case kTypeU16: return 2;
    case kTypeF32:
    case kTypeS32:
    case kTypeU32: return 4;
    case kTypeF64:
    case kTypeS64:
    case kTypeU64: return 8;
    default: return 0;
  }
}

// Total element count from a hbDNNTensorShape. Skips the numDimensions
// tail of the dimensionSize[] array.
int64_t ElementCount(const hbDNNTensorShape &shape) {
  int64_t total = 1;
  for (int32_t i = 0; i < shape.numDimensions; ++i) {
    total *= shape.dimensionSize[i];
  }
  return total;
}

// Deep-copy a hbDNNTensorProperties into our plain C++ vectors so we can drop
// the C handle and not worry about lifetime.
void CopyPropsToVectors(const hbDNNTensorProperties &props,
                        std::vector<int32_t> *shape_out,
                        int32_t *dtype_out) {
  shape_out->assign(props.validShape.dimensionSize,
                    props.validShape.dimensionSize + props.validShape.numDimensions);
  *dtype_out = props.tensorType;
}

const char *DtypeNameImpl(int32_t dtype) {
  switch (dtype) {
    case kTypeS4: return "S4";
    case kTypeU4: return "U4";
    case kTypeS8: return "S8";
    case kTypeU8: return "U8";
    case kTypeF16: return "F16";
    case kTypeS16: return "S16";
    case kTypeU16: return "U16";
    case kTypeF32: return "F32";
    case kTypeS32: return "S32";
    case kTypeU32: return "U32";
    case kTypeF64: return "F64";
    case kTypeS64: return "S64";
    case kTypeU64: return "U64";
    case kTypeBool8: return "BOOL8";
    default: return "?";
  }
}

}  // namespace

int32_t DtypeElementBytes(int32_t dtype) {
  return ElementBytesForType(dtype);
}

const char *DtypeName(int32_t dtype) {
  return DtypeNameImpl(dtype);
}

// ---------------------------------------------------------------------------
// HbmSession
// ---------------------------------------------------------------------------

HbmSession::~HbmSession() {
  if (packed_handle_ != nullptr) {
    hbDNNRelease(packed_handle_);
    packed_handle_ = nullptr;
  }
}

Result HbmSession::Load(const std::string &hbm_path) {
  const char *files[1] = {hbm_path.c_str()};
  hbDNNPackedHandle_t packed = nullptr;
  int32_t err = hbDNNInitializeFromFiles(&packed, files, 1);
  if (err != 0) {
    return Result::Err(err,
        "hbDNNInitializeFromFiles failed for " + hbm_path);
  }
  packed_handle_ = packed;

  // Pull the graph name list. hbDNNGetModelNameList hands us a pointer to
  // an array of char* whose lifetime is tied to the packed handle.
  char const **name_list = nullptr;
  int32_t name_count = 0;
  err = hbDNNGetModelNameList(&name_list, &name_count, packed);
  if (err != 0) {
    return Result::Err(err, "hbDNNGetModelNameList failed");
  }
  graph_names_.clear();
  for (int32_t i = 0; i < name_count; ++i) {
    if (name_list[i] != nullptr) {
      graph_names_.emplace_back(name_list[i]);
    }
  }
  return Result::Ok();
}

Graph *HbmSession::GetGraph(const std::string &name) {
  auto it = graphs_.find(name);
  if (it != graphs_.end()) {
    return it->second.get();
  }
  if (packed_handle_ == nullptr) {
    return nullptr;
  }
  hbDNNHandle_t graph_handle = nullptr;
  int32_t err = hbDNNGetModelHandle(&graph_handle, packed_handle_, name.c_str());
  if (err != 0 || graph_handle == nullptr) {
    return nullptr;
  }
  auto g = std::make_unique<Graph>();
  g->SetHandle(graph_handle);
  Result r = g->RefreshIO(graph_handle);
  if (!r.ok()) {
    std::cerr << "[HbmSession] graph " << name
              << " RefreshIO failed: " << r.message << std::endl;
    return nullptr;
  }
  Graph *raw = g.get();
  graphs_[name] = std::move(g);
  return raw;
}

Result HbmSession::ExecuteGraphByName(const std::string &graph_name,
                                       const std::vector<Tensor> &inputs,
                                       std::vector<Tensor> *outputs) {
  Graph *g = GetGraph(graph_name);
  if (g == nullptr) {
    return Result::Err(-1, "graph not found: " + graph_name);
  }
  return g->Execute(inputs, outputs);
}

// ---------------------------------------------------------------------------
// Graph
// ---------------------------------------------------------------------------

Result Graph::RefreshIO(hbDNNHandle_t handle) {
  if (io_ready_) {
    return Result::Ok();
  }

  // Inputs
  int32_t input_count = 0;
  int32_t err = hbDNNGetInputCount(&input_count, handle);
  if (err != 0) {
    return Result::Err(err, "hbDNNGetInputCount failed");
  }
  input_names_.clear();
  input_shapes_.clear();
  input_dtypes_.clear();
  for (int32_t i = 0; i < input_count; ++i) {
    char const *name = nullptr;
    err = hbDNNGetInputName(&name, handle, i);
    if (err != 0 || name == nullptr) {
      return Result::Err(err, "hbDNNGetInputName idx=" + std::to_string(i));
    }
    input_names_.emplace_back(name);

    hbDNNTensorProperties props;
    err = hbDNNGetInputTensorProperties(&props, handle, i);
    if (err != 0) {
      return Result::Err(err, "hbDNNGetInputTensorProperties idx=" + std::to_string(i));
    }
    std::vector<int32_t> shape;
    int32_t dtype = 0;
    CopyPropsToVectors(props, &shape, &dtype);
    input_shapes_.push_back(std::move(shape));
    input_dtypes_.push_back(dtype);
  }

  // Outputs
  int32_t output_count = 0;
  err = hbDNNGetOutputCount(&output_count, handle);
  if (err != 0) {
    return Result::Err(err, "hbDNNGetOutputCount failed");
  }
  output_names_.clear();
  output_shapes_.clear();
  output_dtypes_.clear();
  for (int32_t i = 0; i < output_count; ++i) {
    char const *name = nullptr;
    err = hbDNNGetOutputName(&name, handle, i);
    if (err != 0 || name == nullptr) {
      return Result::Err(err, "hbDNNGetOutputName idx=" + std::to_string(i));
    }
    output_names_.emplace_back(name);

    hbDNNTensorProperties props;
    err = hbDNNGetOutputTensorProperties(&props, handle, i);
    if (err != 0) {
      return Result::Err(err, "hbDNNGetOutputTensorProperties idx=" + std::to_string(i));
    }
    std::vector<int32_t> shape;
    int32_t dtype = 0;
    CopyPropsToVectors(props, &shape, &dtype);
    output_shapes_.push_back(std::move(shape));
    output_dtypes_.push_back(dtype);
  }

  io_ready_ = true;
  return Result::Ok();
}

Result Graph::Execute(const std::vector<Tensor> &inputs,
                      std::vector<Tensor> *outputs) {
  if (c_handle_ == nullptr) {
    return Result::Err(-1, "Graph::Execute: no C handle set");
  }
  return Execute(c_handle_, inputs, outputs);
}

Result Graph::Execute(hbDNNHandle_t handle,
                      const std::vector<Tensor> &inputs,
                      std::vector<Tensor> *outputs) {
  if (!io_ready_) {
    Result r = RefreshIO(handle);
    if (!r.ok()) return r;
  }
  if (inputs.size() != input_names_.size()) {
    return Result::Err(-1,
        "input count mismatch: got " + std::to_string(inputs.size()) +
        ", expected " + std::to_string(input_names_.size()));
  }

  // Allocate input device buffers, copy host data in, flush clean.
  // hbDNNTensor is {hbUCPSysMem sysMem; hbDNNTensorProperties properties;}
  // We must populate properties to match the declared valid shape + dtype,
  // otherwise hbDNNInferV2 will reject the tensor.
  std::vector<hbDNNTensor> in_tensors(input_names_.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    int32_t err = hbDNNGetInputTensorProperties(&in_tensors[i].properties,
                                               handle, static_cast<int32_t>(i));
    if (err != 0) {
      return Result::Err(err, "Execute: hbDNNGetInputTensorProperties idx=" + std::to_string(i));
    }
    // Allocate aligned-bytes cacheable memory and memcpy the user data.
    uint64_t need_bytes = in_tensors[i].properties.alignedByteSize;
    // Sanity: user-provided data size should be >= element_count * elem_bytes.
    int64_t want_elems = ElementCount(in_tensors[i].properties.validShape);
    int64_t want_bytes = want_elems * ElementBytesForType(in_tensors[i].properties.tensorType);
    if (static_cast<int64_t>(inputs[i].data.size()) < want_bytes) {
      return Result::Err(-1,
          "input " + std::to_string(i) + " data too small: got " +
          std::to_string(inputs[i].data.size()) + " bytes, need " +
          std::to_string(want_bytes));
    }
    err = hbUCPMallocCached(&in_tensors[i].sysMem, need_bytes, 0);
    if (err != 0) {
      return Result::Err(err, "hbUCPMallocCached input idx=" + std::to_string(i));
    }
    std::memcpy(in_tensors[i].sysMem.virAddr, inputs[i].data.data(),
                static_cast<size_t>(std::min<int64_t>(want_bytes, need_bytes)));
    err = hbUCPMemFlush(&in_tensors[i].sysMem, HB_SYS_MEM_CACHE_CLEAN);
    if (err != 0) {
      hbUCPFree(&in_tensors[i].sysMem);
      return Result::Err(err, "hbUCPMemFlush CLEAN input idx=" + std::to_string(i));
    }
  }

  // Allocate output device buffers (BPU will fill them).
  std::vector<hbDNNTensor> out_tensors(output_names_.size());
  for (size_t i = 0; i < output_names_.size(); ++i) {
    int32_t err = hbDNNGetOutputTensorProperties(&out_tensors[i].properties,
                                                handle, static_cast<int32_t>(i));
    if (err != 0) {
      // Free any inputs already allocated.
      for (size_t j = 0; j <= i; ++j) {
        if (in_tensors[j].sysMem.virAddr != nullptr) {
          hbUCPFree(&in_tensors[j].sysMem);
        }
      }
      return Result::Err(err, "Execute: hbDNNGetOutputTensorProperties idx=" + std::to_string(i));
    }
    err = hbUCPMallocCached(&out_tensors[i].sysMem,
                            out_tensors[i].properties.alignedByteSize, 0);
    if (err != 0) {
      for (auto &t : in_tensors) {
        if (t.sysMem.virAddr != nullptr) hbUCPFree(&t.sysMem);
      }
      for (size_t j = 0; j < i; ++j) {
        if (out_tensors[j].sysMem.virAddr != nullptr) hbUCPFree(&out_tensors[j].sysMem);
      }
      return Result::Err(err, "hbUCPMallocCached output idx=" + std::to_string(i));
    }
  }

  // Submit inference + wait.
  // The hobot-dnn C API splits submission and wait:
  //   hbDNNInferV2  -> create the task + bind tensors (NOT auto-submitted)
  //   hbUCPSubmitTask -> actually kick off the BPU
  //   hbUCPWaitTaskDone -> block until done
  // SchedParam can be NULL on S600 — the runtime fills defaults (priority 0,
  // custom_id 0, all-cores). If non-default scheduling is needed later, we
  // plumb a hbUCPSchedParam through the API; for now NULL is fine.
  hbUCPTaskHandle_t task = nullptr;
  int32_t err = hbDNNInferV2(&task, out_tensors.data(),
                              in_tensors.data(), handle);
  if (err != 0) {
    for (auto &t : in_tensors) if (t.sysMem.virAddr) hbUCPFree(&t.sysMem);
    for (auto &t : out_tensors) if (t.sysMem.virAddr) hbUCPFree(&t.sysMem);
    return Result::Err(err, "hbDNNInferV2 failed");
  }
  err = hbUCPSubmitTask(task, nullptr);
  if (err != 0) {
    hbUCPReleaseTask(task);
    for (auto &t : in_tensors) if (t.sysMem.virAddr) hbUCPFree(&t.sysMem);
    for (auto &t : out_tensors) if (t.sysMem.virAddr) hbUCPFree(&t.sysMem);
    return Result::Err(err, "hbUCPSubmitTask failed");
  }
  // Timeout -1 = wait forever.
  err = hbUCPWaitTaskDone(task, -1);
  if (err != 0) {
    hbUCPReleaseTask(task);
    for (auto &t : in_tensors) if (t.sysMem.virAddr) hbUCPFree(&t.sysMem);
    for (auto &t : out_tensors) if (t.sysMem.virAddr) hbUCPFree(&t.sysMem);
    return Result::Err(err, "hbUCPWaitTaskDone failed");
  }

  // Pull output data from BPU cache into host vectors.
  outputs->clear();
  outputs->reserve(out_tensors.size());
  for (size_t i = 0; i < out_tensors.size(); ++i) {
    hbUCPMemFlush(&out_tensors[i].sysMem, HB_SYS_MEM_CACHE_INVALIDATE);
    int64_t want_elems = ElementCount(out_tensors[i].properties.validShape);
    int32_t elem_bytes = ElementBytesForType(out_tensors[i].properties.tensorType);
    int64_t want_bytes = want_elems * elem_bytes;

    Tensor t;
    t.shape.assign(out_tensors[i].properties.validShape.dimensionSize,
                   out_tensors[i].properties.validShape.dimensionSize +
                       out_tensors[i].properties.validShape.numDimensions);
    t.dtype = out_tensors[i].properties.tensorType;
    t.data.resize(static_cast<size_t>(want_bytes));
    std::memcpy(t.data.data(), out_tensors[i].sysMem.virAddr,
                static_cast<size_t>(want_bytes));
    outputs->push_back(std::move(t));
  }

  // Release task + free buffers.
  hbUCPReleaseTask(task);
  for (auto &t : in_tensors) if (t.sysMem.virAddr) hbUCPFree(&t.sysMem);
  for (auto &t : out_tensors) if (t.sysMem.virAddr) hbUCPFree(&t.sysMem);

  return Result::Ok();
}

}  // namespace locateanything_runtime

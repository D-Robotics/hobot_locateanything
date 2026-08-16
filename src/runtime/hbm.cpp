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
//   5) allocate graph IO buffers once; memcpy + hbUCPMemFlush(CLEAN) per input
//   6) reuse the graph output buffers
//   7) hbDNNInferV2               -> task_handle
//   8) hbUCPSubmitTask            -> kicks off the BPU
//   9) hbUCPWaitTaskDone          -> block until done
//  10) hbUCPMemFlush(INVALIDATE) on each output  -> pull results from BPU
//  11) hbUCPReleaseTask           -> free the task handle
//  12) retain IO buffers while the graph remains active

#include "runtime/hbm.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
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

/**
 * @brief Return the vendor tensor element width used for packing.
 * @param dtype Vendor tensor-type integer.
 * @return Element width in bytes, or zero when unsupported.
 */
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
/**
 * @brief Multiply all declared tensor dimensions into an element count.
 * @param shape Vendor tensor shape.
 * @return Logical element count.
 */
int64_t ElementCount(const hbDNNTensorShape &shape) {
  int64_t total = 1;
  for (int32_t i = 0; i < shape.numDimensions; ++i) {
    total *= shape.dimensionSize[i];
  }
  return total;
}

/**
 * @brief Build contiguous row-major byte strides for a tensor shape.
 * @param shape Vendor tensor shape.
 * @param element_bytes Byte width of one element.
 * @return Byte stride for every dimension.
 */
std::vector<int64_t> CompactStrides(const hbDNNTensorShape &shape,
                                    int32_t element_bytes) {
  std::vector<int64_t> strides(shape.numDimensions);
  int64_t stride = element_bytes;
  for (int32_t index = shape.numDimensions - 1; index >= 0; --index) {
    strides[index] = stride;
    stride *= shape.dimensionSize[index];
  }
  return strides;
}

/**
 * @brief Read vendor strides, falling back to compact strides when absent.
 * @param properties Vendor tensor properties.
 * @param element_bytes Byte width of one element.
 * @return Effective byte stride for every dimension.
 */
std::vector<int64_t> DeclaredStrides(const hbDNNTensorProperties &properties,
                                     int32_t element_bytes) {
  auto strides = CompactStrides(properties.validShape, element_bytes);
  for (int32_t index = 0; index < properties.validShape.numDimensions; ++index) {
    if (properties.stride[index] > 0) strides[index] = properties.stride[index];
  }
  return strides;
}

/**
 * @brief Compute the byte span required by a shape and its strides.
 * @param shape Vendor tensor shape.
 * @param strides Effective byte strides.
 * @param element_bytes Byte width of one element.
 * @return Required byte span.
 */
int64_t LayoutBytes(const hbDNNTensorShape &shape,
                    const std::vector<int64_t> &strides,
                    int32_t element_bytes) {
  int64_t bytes = element_bytes;
  for (int32_t index = 0; index < shape.numDimensions; ++index) {
    bytes += (shape.dimensionSize[index] - 1) * strides[index];
  }
  return bytes;
}

/**
 * @brief Copy a tensor between compact or strided host/vendor layouts.
 * @param source Source tensor bytes.
 * @param source_strides Source byte strides.
 * @param destination Destination tensor bytes.
 * @param destination_strides Destination byte strides.
 * @param shape Shared logical tensor shape.
 * @param element_bytes Byte width of one element.
 * @return True when source and destination layouts are valid.
 */
bool CopyLayout(const uint8_t *source, const std::vector<int64_t> &source_strides,
                uint8_t *destination, const std::vector<int64_t> &destination_strides,
                const hbDNNTensorShape &shape, int32_t element_bytes) {
  if (shape.numDimensions <= 0 || source == nullptr || destination == nullptr) {
    return false;
  }
  const auto compact = CompactStrides(shape, element_bytes);
  if (source_strides == compact && destination_strides == compact) {
    std::memcpy(destination, source,
                static_cast<size_t>(ElementCount(shape)) * element_bytes);
    return true;
  }
  const int32_t last = shape.numDimensions - 1;
  int64_t rows = 1;
  for (int32_t index = 0; index < last; ++index) rows *= shape.dimensionSize[index];
  for (int64_t row = 0; row < rows; ++row) {
    int64_t remainder = row;
    int64_t source_offset = 0;
    int64_t destination_offset = 0;
    for (int32_t index = last - 1; index >= 0; --index) {
      const int32_t coordinate = remainder % shape.dimensionSize[index];
      remainder /= shape.dimensionSize[index];
      source_offset += coordinate * source_strides[index];
      destination_offset += coordinate * destination_strides[index];
    }
    if (source_strides[last] == element_bytes &&
        destination_strides[last] == element_bytes) {
      std::memcpy(destination + destination_offset, source + source_offset,
                  static_cast<size_t>(shape.dimensionSize[last]) * element_bytes);
      continue;
    }
    for (int32_t column = 0; column < shape.dimensionSize[last]; ++column) {
      std::memcpy(destination + destination_offset + column * destination_strides[last],
                  source + source_offset + column * source_strides[last], element_bytes);
    }
  }
  return true;
}

// Deep-copy a hbDNNTensorProperties into our plain C++ vectors so we can drop
// the C handle and not worry about lifetime.
/**
 * @brief Copy vendor IO metadata into lifetime-independent vectors.
 * @param props Vendor tensor properties.
 * @param shape_out Destination logical dimensions.
 * @param dtype_out Destination vendor tensor-type integer.
 */
void CopyPropsToVectors(const hbDNNTensorProperties &props,
                        std::vector<int32_t> *shape_out,
                        int32_t *dtype_out) {
  shape_out->assign(props.validShape.dimensionSize,
                    props.validShape.dimensionSize + props.validShape.numDimensions);
  *dtype_out = props.tensorType;
}

/**
 * @brief Map a vendor dtype integer to a stable diagnostic name.
 * @param dtype Vendor tensor-type integer.
 * @return Static short name.
 */
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

struct DeviceBuffer::Impl {
  hbUCPSysMem memory{};
};

/** Allocate the private vendor-memory state without allocating device bytes. */
DeviceBuffer::DeviceBuffer() : impl_(std::make_unique<Impl>()) {}

/** Release the UCP allocation owned by this buffer. */
DeviceBuffer::~DeviceBuffer() {
  if (impl_ != nullptr && impl_->memory.virAddr != nullptr) {
    hbUCPFree(&impl_->memory);
    impl_->memory = {};
  }
}

/** Return the allocated UCP byte count, or zero for an empty buffer. */
size_t DeviceBuffer::size() const {
  return impl_ == nullptr ? 0 : static_cast<size_t>(impl_->memory.memSize);
}

/** Allocate cacheable UCP memory and optionally clean an all-zero buffer. */
Result AllocateDeviceBuffer(size_t bytes, bool zero_initialize,
                            std::shared_ptr<DeviceBuffer> *buffer) {
  if (buffer == nullptr || bytes == 0) {
    return Result::Err(-1, "AllocateDeviceBuffer: invalid argument");
  }
  auto allocated = std::shared_ptr<DeviceBuffer>(new DeviceBuffer());
  const int32_t err = hbUCPMallocCached(&allocated->impl_->memory, bytes, 0);
  if (err != 0) {
    return Result::Err(err, "hbUCPMallocCached device buffer failed");
  }
  if (zero_initialize) {
    std::memset(allocated->impl_->memory.virAddr, 0, bytes);
    const int32_t flush_err = hbUCPMemFlush(
        &allocated->impl_->memory, HB_SYS_MEM_CACHE_CLEAN);
    if (flush_err != 0) {
      return Result::Err(flush_err,
                         "hbUCPMemFlush CLEAN device buffer failed");
    }
  }
  *buffer = std::move(allocated);
  return Result::Ok();
}

/** Copy and cache-clean one changed range in a device-backed tensor. */
Result WriteDeviceBuffer(const std::shared_ptr<DeviceBuffer> &buffer,
                         size_t byte_offset, const void *source, size_t bytes) {
  if (buffer == nullptr || buffer->impl_ == nullptr || source == nullptr ||
      bytes == 0 || byte_offset > buffer->size() ||
      bytes > buffer->size() - byte_offset) {
    return Result::Err(-1, "WriteDeviceBuffer: invalid range");
  }
  auto *destination = static_cast<uint8_t *>(buffer->impl_->memory.virAddr) +
                      byte_offset;
  std::memcpy(destination, source, bytes);
  hbUCPSysMem view{};
  view.phyAddr = buffer->impl_->memory.phyAddr + byte_offset;
  view.virAddr = destination;
  view.memSize = bytes;
  const int32_t err = hbUCPMemFlush(&view, HB_SYS_MEM_CACHE_CLEAN);
  if (err != 0) {
    return Result::Err(err, "hbUCPMemFlush CLEAN device range failed");
  }
  return Result::Ok();
}

struct Graph::PersistentBuffers {
  std::vector<hbDNNTensor> inputs;
  std::vector<hbDNNTensor> outputs;

  /** Release every graph-private input and output UCP allocation. */
  ~PersistentBuffers() {
    for (auto &tensor : inputs) {
      if (tensor.sysMem.virAddr != nullptr) hbUCPFree(&tensor.sysMem);
    }
    for (auto &tensor : outputs) {
      if (tensor.sysMem.virAddr != nullptr) hbUCPFree(&tensor.sysMem);
    }
  }
};

/** Create an empty graph metadata/cache wrapper. */
Graph::Graph() = default;
/** Release graph-owned persistent IO buffers. */
Graph::~Graph() = default;

/** Expose the vendor-independent element width to the Language runtime. */
int32_t DtypeElementBytes(int32_t dtype) {
  return ElementBytesForType(dtype);
}

/** Expose a short vendor-independent dtype name for diagnostics. */
const char *DtypeName(int32_t dtype) {
  return DtypeNameImpl(dtype);
}

// ---------------------------------------------------------------------------
// HbmSession
// ---------------------------------------------------------------------------

/** Release all graph wrappers before releasing the packed HBM handle. */
HbmSession::~HbmSession() {
  graphs_.clear();
  if (packed_handle_ != nullptr) {
    hbDNNRelease(packed_handle_);
    packed_handle_ = nullptr;
  }
}

/** Load one packed HBM file and cache its graph-name contract. */
Result HbmSession::Load(const std::string &hbm_path) {
  constexpr char kL2MemoryVariable[] = "HB_DNN_USER_DEFINED_L2M_SIZES";
  if (std::getenv(kL2MemoryVariable) == nullptr &&
      setenv(kL2MemoryVariable, "6:6:6:6", 0) != 0) {
    return Result::Err(errno,
                       "failed to configure default BPU L2 memory allocation");
  }

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

/** Lazily resolve a named graph and refresh its IO metadata once. */
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
  g->SetBackendMask(backend_mask_);
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

/** Resolve and execute a named graph using owned input tensor values. */
Result HbmSession::ExecuteGraphByName(const std::string &graph_name,
                                       const std::vector<Tensor> &inputs,
                                       std::vector<Tensor> *outputs,
                                       ExecutionMetrics *metrics,
                                       const std::vector<OutputSlice> *output_slices) {
  Graph *g = GetGraph(graph_name);
  if (g == nullptr) {
    return Result::Err(-1, "graph not found: " + graph_name);
  }
  return g->Execute(inputs, outputs, metrics, output_slices);
}

/** Resolve and execute a named graph using caller-owned tensor views. */
Result HbmSession::ExecuteGraphByName(
    const std::string &graph_name,
    const std::vector<const Tensor *> &inputs,
    std::vector<Tensor> *outputs,
    ExecutionMetrics *metrics,
    const std::vector<OutputSlice> *output_slices) {
  Graph *g = GetGraph(graph_name);
  if (g == nullptr) {
    return Result::Err(-1, "graph not found: " + graph_name);
  }
  return g->Execute(inputs, outputs, metrics, output_slices);
}

// ---------------------------------------------------------------------------
// Graph
// ---------------------------------------------------------------------------

/** Read and cache the graph's input/output names, shapes, and dtypes. */
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

/** Adapt owned input values to views and execute through the shared path. */
Result Graph::Execute(const std::vector<Tensor> &inputs,
                      std::vector<Tensor> *outputs,
                      ExecutionMetrics *metrics,
                      const std::vector<OutputSlice> *output_slices) {
  std::vector<const Tensor *> views;
  views.reserve(inputs.size());
  for (const auto &input : inputs) views.push_back(&input);
  return Execute(views, outputs, metrics, output_slices);
}

/** Execute using the handle remembered by SetHandle. */
Result Graph::Execute(const std::vector<const Tensor *> &inputs,
                      std::vector<Tensor> *outputs,
                      ExecutionMetrics *metrics,
                      const std::vector<OutputSlice> *output_slices) {
  if (c_handle_ == nullptr) {
    return Result::Err(-1, "Graph::Execute: no C handle set");
  }
  return Execute(c_handle_, inputs, outputs, metrics, output_slices);
}

/** Adapt owned input values for an explicitly supplied vendor handle. */
Result Graph::Execute(hbDNNHandle_t handle,
                      const std::vector<Tensor> &inputs,
                      std::vector<Tensor> *outputs,
                      ExecutionMetrics *metrics,
                      const std::vector<OutputSlice> *output_slices) {
  std::vector<const Tensor *> views;
  views.reserve(inputs.size());
  for (const auto &input : inputs) views.push_back(&input);
  return Execute(handle, views, outputs, metrics, output_slices);
}

/** Allocate reusable vendor IO tensors using the graph's declared layout. */
Result Graph::EnsurePersistentBuffers(hbDNNHandle_t handle) {
  if (buffers_ != nullptr) return Result::Ok();

  auto buffers = std::make_unique<PersistentBuffers>();
  buffers->inputs.resize(input_names_.size());
  buffers->outputs.resize(output_names_.size());

  for (size_t index = 0; index < buffers->inputs.size(); ++index) {
    hbDNNTensor &tensor = buffers->inputs[index];
    int32_t err = hbDNNGetInputTensorProperties(
        &tensor.properties, handle, static_cast<int32_t>(index));
    if (err != 0) {
      return Result::Err(err, "hbDNNGetInputTensorProperties idx=" +
                                 std::to_string(index));
    }
    const int32_t element_bytes = ElementBytesForType(tensor.properties.tensorType);
    if (element_bytes <= 0) {
      return Result::Err(-1, "unsupported input dtype idx=" +
                                 std::to_string(index));
    }
    const auto strides = DeclaredStrides(tensor.properties, element_bytes);
    const int64_t layout_bytes = LayoutBytes(
        tensor.properties.validShape, strides, element_bytes);
    int64_t allocation_bytes = tensor.properties.alignedByteSize;
    if (allocation_bytes <= 0) allocation_bytes = layout_bytes;
    if (layout_bytes > allocation_bytes) {
      return Result::Err(-1, "input stride layout exceeds allocation idx=" +
                                 std::to_string(index));
    }
    // Input storage is allocated lazily in Execute. Device-resident KV inputs
    // replace this view directly and must not reserve another private copy for
    // every Language graph.
  }

  for (size_t index = 0; index < buffers->outputs.size(); ++index) {
    hbDNNTensor &tensor = buffers->outputs[index];
    int32_t err = hbDNNGetOutputTensorProperties(
        &tensor.properties, handle, static_cast<int32_t>(index));
    if (err != 0) {
      return Result::Err(err, "hbDNNGetOutputTensorProperties idx=" +
                                 std::to_string(index));
    }
    const int32_t element_bytes = ElementBytesForType(tensor.properties.tensorType);
    if (element_bytes <= 0) {
      return Result::Err(-1, "unsupported output dtype idx=" +
                                 std::to_string(index));
    }
    const auto strides = DeclaredStrides(tensor.properties, element_bytes);
    const int64_t layout_bytes = LayoutBytes(
        tensor.properties.validShape, strides, element_bytes);
    int64_t allocation_bytes = tensor.properties.alignedByteSize;
    if (allocation_bytes <= 0) allocation_bytes = layout_bytes;
    if (layout_bytes > allocation_bytes) {
      return Result::Err(-1, "output stride layout exceeds allocation idx=" +
                                 std::to_string(index));
    }
    err = hbUCPMallocCached(&tensor.sysMem, allocation_bytes, 0);
    if (err != 0) {
      return Result::Err(err, "hbUCPMallocCached output idx=" +
                                 std::to_string(index));
    }
  }

  buffers_ = std::move(buffers);
  return Result::Ok();
}

/** Pack inputs, submit one synchronous BPU task, and materialize outputs. */
Result Graph::Execute(hbDNNHandle_t handle,
                      const std::vector<const Tensor *> &inputs,
                      std::vector<Tensor> *outputs,
                      ExecutionMetrics *metrics,
                      const std::vector<OutputSlice> *output_slices) {
  using Clock = std::chrono::steady_clock;
  const auto total_started = Clock::now();
  if (metrics != nullptr) *metrics = {};
  if (!io_ready_) {
    Result r = RefreshIO(handle);
    if (!r.ok()) return r;
  }
  if (inputs.size() != input_names_.size()) {
    return Result::Err(-1,
        "input count mismatch: got " + std::to_string(inputs.size()) +
        ", expected " + std::to_string(input_names_.size()));
  }
  if (output_slices != nullptr && output_slices->size() != output_names_.size()) {
    return Result::Err(-1, "output slice count mismatch: got " +
                                std::to_string(output_slices->size()) +
                                ", expected " + std::to_string(output_names_.size()));
  }
  const auto prepare_started = Clock::now();
  Result ready = EnsurePersistentBuffers(handle);
  if (metrics != nullptr) {
    metrics->buffer_prepare_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - prepare_started).count();
  }
  if (!ready.ok()) return ready;

  auto &in_tensors = buffers_->inputs;
  auto &out_tensors = buffers_->outputs;
  // Keep Graph-owned input allocations intact. Device-resident inputs replace
  // only the per-submission hbUCPSysMem view and may be shared across graphs.
  std::vector<hbDNNTensor> bound_inputs = in_tensors;
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i] == nullptr || inputs[i]->shape != input_shapes_[i] ||
        inputs[i]->dtype != input_dtypes_[i]) {
      return Result::Err(-1, "input shape or dtype mismatch idx=" +
                                 std::to_string(i));
    }
    int64_t want_elems = ElementCount(in_tensors[i].properties.validShape);
    const int32_t element_bytes = ElementBytesForType(
        in_tensors[i].properties.tensorType);
    int64_t want_bytes = want_elems * element_bytes;
    const auto compact_strides = CompactStrides(
        in_tensors[i].properties.validShape, element_bytes);
    const auto declared_strides = DeclaredStrides(
        in_tensors[i].properties, element_bytes);
    const int64_t layout_bytes = LayoutBytes(
        in_tensors[i].properties.validShape, declared_strides, element_bytes);
    int64_t aligned_bytes = in_tensors[i].properties.alignedByteSize;
    if (aligned_bytes <= 0) aligned_bytes = layout_bytes;
    if (inputs[i]->device_buffer != nullptr) {
      const size_t offset = inputs[i]->byte_offset;
      if (offset > inputs[i]->device_buffer->size() ||
          static_cast<uint64_t>(aligned_bytes) >
              inputs[i]->device_buffer->size() - offset) {
        return Result::Err(-1, "device input range too small idx=" +
                                   std::to_string(i));
      }
      const hbUCPSysMem &base = inputs[i]->device_buffer->impl_->memory;
      bound_inputs[i].sysMem.phyAddr = base.phyAddr + offset;
      bound_inputs[i].sysMem.virAddr =
          static_cast<uint8_t *>(base.virAddr) + offset;
      bound_inputs[i].sysMem.memSize = aligned_bytes;
      if (metrics != nullptr) {
        metrics->resident_input_bytes += static_cast<uint64_t>(want_bytes);
      }
      continue;
    }
    if (in_tensors[i].sysMem.virAddr == nullptr) {
      const int32_t alloc_err = hbUCPMallocCached(
          &in_tensors[i].sysMem, aligned_bytes, 0);
      if (alloc_err != 0) {
        return Result::Err(alloc_err, "hbUCPMallocCached input idx=" +
                                         std::to_string(i));
      }
    }
    bound_inputs[i].sysMem = in_tensors[i].sysMem;
    if (inputs[i]->byte_offset > inputs[i]->data.size() ||
        static_cast<int64_t>(inputs[i]->data.size() - inputs[i]->byte_offset) <
            want_bytes) {
      return Result::Err(-1,
          "input " + std::to_string(i) + " data too small: got " +
          std::to_string(inputs[i]->data.size() -
                         std::min(inputs[i]->byte_offset, inputs[i]->data.size())) +
          " bytes after offset, need " +
          std::to_string(want_bytes));
    }
    const auto pack_started = Clock::now();
    if (aligned_bytes > layout_bytes || compact_strides != declared_strides) {
      std::memset(in_tensors[i].sysMem.virAddr, 0,
                  static_cast<size_t>(aligned_bytes));
    }
    if (!CopyLayout(inputs[i]->data.data() + inputs[i]->byte_offset,
                    compact_strides,
                    static_cast<uint8_t *>(in_tensors[i].sysMem.virAddr),
                    declared_strides, in_tensors[i].properties.validShape,
                    element_bytes)) {
      return Result::Err(-1, "cannot pack input tensor idx=" + std::to_string(i));
    }
    if (metrics != nullptr) {
      metrics->input_pack_ms += std::chrono::duration<double, std::milli>(
          Clock::now() - pack_started).count();
      metrics->input_bytes += static_cast<uint64_t>(want_bytes);
    }
    const auto flush_started = Clock::now();
    int32_t err = hbUCPMemFlush(&in_tensors[i].sysMem, HB_SYS_MEM_CACHE_CLEAN);
    if (err != 0) {
      return Result::Err(err, "hbUCPMemFlush CLEAN input idx=" + std::to_string(i));
    }
    if (metrics != nullptr) {
      metrics->input_flush_ms += std::chrono::duration<double, std::milli>(
          Clock::now() - flush_started).count();
    }
  }

  // Submit inference + wait.
  // The hobot-dnn C API splits submission and wait:
  //   hbDNNInferV2  -> create the task + bind tensors (NOT auto-submitted)
  //   hbUCPSubmitTask -> actually kick off the BPU
  //   hbUCPWaitTaskDone -> block until done
  // UCP 3.12.3 requires a valid scheduling parameter. The session default
  // selects all four S600 BPU cores and callers may override the mask.
  hbUCPTaskHandle_t task = nullptr;
  const auto submit_started = Clock::now();
  int32_t err = hbDNNInferV2(&task, out_tensors.data(),
                               bound_inputs.data(), handle);
  if (err != 0) {
    return Result::Err(err, "hbDNNInferV2 failed");
  }
  hbUCPSchedParam sched{};
  HB_UCP_INITIALIZE_SCHED_PARAM(&sched);
  sched.backend = backend_mask_;
  err = hbUCPSubmitTask(task, &sched);
  if (err != 0) {
    hbUCPReleaseTask(task);
    return Result::Err(err, "hbUCPSubmitTask failed");
  }
  if (metrics != nullptr) {
    metrics->submit_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - submit_started).count();
  }
  // timeout = 0 (synchronous wait — match HB_HBMRuntime.cc::InferSingleModel)
  const auto wait_started = Clock::now();
  err = hbUCPWaitTaskDone(task, 0);
  if (err != 0) {
    hbUCPReleaseTask(task);
    return Result::Err(err, "hbUCPWaitTaskDone failed");
  }
  if (metrics != nullptr) {
    metrics->bpu_wait_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - wait_started).count();
  }

  // Pull output data from BPU cache into host vectors.
  outputs->resize(out_tensors.size());
  for (size_t i = 0; i < out_tensors.size(); ++i) {
    Tensor &t = (*outputs)[i];
    t.byte_offset = 0;
    t.device_buffer.reset();
    if (output_slices != nullptr &&
        !(*output_slices)[i].materialize) {
      t.shape.assign(out_tensors[i].properties.validShape.dimensionSize,
                     out_tensors[i].properties.validShape.dimensionSize +
                         out_tensors[i].properties.validShape.numDimensions);
      t.dtype = out_tensors[i].properties.tensorType;
      t.data.clear();
      continue;
    }
    const auto flush_started = Clock::now();
    err = hbUCPMemFlush(&out_tensors[i].sysMem, HB_SYS_MEM_CACHE_INVALIDATE);
    if (err != 0) {
      hbUCPReleaseTask(task);
      return Result::Err(err, "hbUCPMemFlush INVALIDATE output idx=" +
                              std::to_string(i));
    }
    if (metrics != nullptr) {
      metrics->output_flush_ms += std::chrono::duration<double, std::milli>(
          Clock::now() - flush_started).count();
    }
    const auto unpack_started = Clock::now();
    int32_t elem_bytes = ElementBytesForType(out_tensors[i].properties.tensorType);
    hbDNNTensorShape selected_shape = out_tensors[i].properties.validShape;
    int32_t row_start = 0;
    if (output_slices != nullptr && (*output_slices)[i].row_count >= 0) {
      const OutputSlice &slice = (*output_slices)[i];
      if (selected_shape.numDimensions < 2 || slice.row_start < 0 ||
          slice.row_count <= 0 ||
          slice.row_start + slice.row_count > selected_shape.dimensionSize[1]) {
        hbUCPReleaseTask(task);
        return Result::Err(-1, "invalid output row slice idx=" +
                                   std::to_string(i));
      }
      row_start = slice.row_start;
      selected_shape.dimensionSize[1] = slice.row_count;
    }
    const int64_t want_elems = ElementCount(selected_shape);
    const int64_t want_bytes = want_elems * elem_bytes;

    t.shape.assign(selected_shape.dimensionSize,
                   selected_shape.dimensionSize + selected_shape.numDimensions);
    t.dtype = out_tensors[i].properties.tensorType;
    t.data.resize(static_cast<size_t>(want_bytes));
    const auto declared_strides = DeclaredStrides(
        out_tensors[i].properties, elem_bytes);
    const auto compact_strides = CompactStrides(selected_shape, elem_bytes);
    const int64_t aligned_bytes = out_tensors[i].properties.alignedByteSize > 0
        ? out_tensors[i].properties.alignedByteSize
        : LayoutBytes(out_tensors[i].properties.validShape, declared_strides,
                      elem_bytes);
    if (LayoutBytes(out_tensors[i].properties.validShape, declared_strides,
                    elem_bytes) > aligned_bytes ||
        !CopyLayout(static_cast<const uint8_t *>(out_tensors[i].sysMem.virAddr) +
                        static_cast<int64_t>(row_start) * declared_strides[1],
                    declared_strides, t.data.data(), compact_strides,
                    selected_shape, elem_bytes)) {
      hbUCPReleaseTask(task);
      return Result::Err(-1, "cannot unpack output tensor idx=" + std::to_string(i));
    }
    if (metrics != nullptr) {
      metrics->output_unpack_ms += std::chrono::duration<double, std::milli>(
          Clock::now() - unpack_started).count();
      metrics->output_bytes += static_cast<uint64_t>(want_bytes);
    }
  }

  // The task is per-inference; graph IO allocations stay cached for reuse.
  hbUCPReleaseTask(task);

  if (metrics != nullptr) {
    metrics->total_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - total_started).count();
  }

  return Result::Ok();
}

}  // namespace locateanything_runtime

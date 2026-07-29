// Copyright (c) 2026 LiuAnclouds / Kangjie Xu / D-Robotics
//
// C++ host runtime — HBm session wrapper over libhbrt4 / hb_dnn C API.
//
// Loads a .hbm file (one or multiple graphs packed inside) and exposes a
// minimal execute() that:
//   - allocates BPU-cached memory for each input tensor
//   - memcpy user data into the device buffers
//   - allocates output tensor memory
//   - submits an inference task and waits for it to complete
//   - copies the results back into host-side std::vector
//
// Phase 1 scope: just enough to load vision.hbm and run the "visual" graph
// once with a dummy input. Phase 2+ will add multi-graph orchestration,
// KV-cache reuse, PBD mask building etc. on top.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// Forward-declare the C handles so we don't leak hb_dnn.h into every TU.
extern "C" {
typedef void *hbDNNPackedHandle_t;
typedef void *hbDNNHandle_t;
struct hbDNNTensor;
struct hbDNNTensorProperties;
}

namespace locateanything_runtime {

class DeviceBuffer;
struct Result;

// A typed blob of host memory used as either an input to or output from a
// graph. `data` is contiguous row-major; `shape` is in NCHW-ish order
// (whatever the hbm declared); `dtype` follows hb_dnn's HB_DNN_TENSOR_TYPE_*.
struct Tensor {
  std::vector<int32_t> shape;   // e.g. [1, 1024, 588]
  int32_t dtype = 0;            // HB_DNN_TENSOR_TYPE_F16 = 4 etc.
  std::vector<uint8_t> data;    // raw bytes, size = element_count * element_bytes
  size_t byte_offset = 0;       // logical view start; used by ring-buffer KV caches
  // Optional UCP/DDR backing. When present, Graph binds this storage directly
  // instead of copying `data` into its private input buffer.
  std::shared_ptr<DeviceBuffer> device_buffer;
};

// Cacheable UCP memory visible to both Host and BPU. The implementation keeps
// hbUCPSysMem private so public callers do not depend on the vendor headers.
class DeviceBuffer {
 public:
  ~DeviceBuffer();
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  size_t size() const;

 private:
  DeviceBuffer();
  struct Impl;
  std::unique_ptr<Impl> impl_;

  friend class Graph;
  friend Result AllocateDeviceBuffer(size_t, bool,
                                     std::shared_ptr<DeviceBuffer> *);
  friend Result WriteDeviceBuffer(const std::shared_ptr<DeviceBuffer> &,
                                  size_t, const void *, size_t);
};

// Status returned by every wrapper method. 0 == success, non-zero carries
// the underlying hbDNN / hbUCP error code.
struct Result {
  int32_t code = 0;
  std::string message;
  bool ok() const { return code == 0; }
  static Result Ok() { return {0, ""}; }
  static Result Err(int32_t c, std::string m) { return {c, std::move(m)}; }
};

struct ExecutionMetrics {
  double buffer_prepare_ms = 0.0;
  double input_pack_ms = 0.0;
  double input_flush_ms = 0.0;
  double submit_ms = 0.0;
  double bpu_wait_ms = 0.0;
  double output_flush_ms = 0.0;
  double output_unpack_ms = 0.0;
  double total_ms = 0.0;
  uint64_t input_bytes = 0;
  uint64_t resident_input_bytes = 0;
  uint64_t output_bytes = 0;
};

// Allocate cacheable UCP/DDR memory. `zero_initialize` also cleans the cache so
// a graph can consume the buffer immediately without another full-buffer copy.
Result AllocateDeviceBuffer(size_t bytes, bool zero_initialize,
                            std::shared_ptr<DeviceBuffer> *buffer);

// Copy a changed range into UCP memory and clean only that range for the BPU.
Result WriteDeviceBuffer(const std::shared_ptr<DeviceBuffer> &buffer,
                         size_t byte_offset, const void *source, size_t bytes);

// Optional row selection for graph outputs whose sequence dimension is axis 1.
// A negative row_count keeps the complete tensor. The BPU still produces the
// declared output, but the Host only materializes the requested rows.
struct OutputSlice {
  int32_t row_start = 0;
  int32_t row_count = -1;
};

// One loaded graph inside a packed hbm file. Lazily populated with IO
// metadata (input/output names, properties, shapes) the first time it is
// queried.
class Graph {
 public:
  Graph();
  ~Graph();

  // Query the graph's name list and per-tensor properties. Cheap after the
  // first call (results cached).
  Result RefreshIO(hbDNNHandle_t handle);

  // Release graph-private input/output allocations while retaining the graph
  // handle and cached IO metadata. Device-resident KV buffers are owned by
  // callers and are unaffected.
  void ReleasePersistentBuffers();

  // Remember the C handle for later Execute calls. Kept separate from
  // RefreshIO so that the HbmSession can pass the handle in once when it
  // first looks the graph up.
  void SetHandle(void *handle) { c_handle_ = handle; }
  void *GetHandle() const { return c_handle_; }

  // Run the graph once with the given inputs, using the previously SetHandle
  // C handle. `inputs` must be in the same order as GetInputNames(); each
  // Tensor's shape/dtype must match the graph's declared IO. On success,
  // `outputs` is filled in the order returned by GetOutputNames().
  Result Execute(const std::vector<Tensor> &inputs,
                 std::vector<Tensor> *outputs,
                 ExecutionMetrics *metrics = nullptr,
                 const std::vector<OutputSlice> *output_slices = nullptr);
  Result Execute(const std::vector<const Tensor *> &inputs,
                 std::vector<Tensor> *outputs,
                 ExecutionMetrics *metrics = nullptr,
                 const std::vector<OutputSlice> *output_slices = nullptr);

  // Run the graph with an explicit C handle (used when the caller has one
  // but didn't go through SetHandle — e.g. from HbmSession).
  Result Execute(hbDNNHandle_t handle,
                 const std::vector<Tensor> &inputs,
                 std::vector<Tensor> *outputs,
                 ExecutionMetrics *metrics = nullptr,
                 const std::vector<OutputSlice> *output_slices = nullptr);
  Result Execute(hbDNNHandle_t handle,
                 const std::vector<const Tensor *> &inputs,
                 std::vector<Tensor> *outputs,
                 ExecutionMetrics *metrics = nullptr,
                 const std::vector<OutputSlice> *output_slices = nullptr);

  const std::vector<std::string> &GetInputNames() const { return input_names_; }
  const std::vector<std::string> &GetOutputNames() const { return output_names_; }
  const std::vector<std::vector<int32_t>> &GetInputShapes() const { return input_shapes_; }
  const std::vector<std::vector<int32_t>> &GetOutputShapes() const { return output_shapes_; }
  const std::vector<int32_t> &GetInputDtypes() const { return input_dtypes_; }
  const std::vector<int32_t> &GetOutputDtypes() const { return output_dtypes_; }

 private:
  struct PersistentBuffers;

  Result EnsurePersistentBuffers(hbDNNHandle_t handle);

  bool io_ready_ = false;
  void *c_handle_ = nullptr;  // hbDNNHandle_t, owned by the packed handle
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<std::vector<int32_t>> input_shapes_;
  std::vector<std::vector<int32_t>> output_shapes_;
  std::vector<int32_t> input_dtypes_;
  std::vector<int32_t> output_dtypes_;
  std::unique_ptr<PersistentBuffers> buffers_;
};

// A loaded hbm file (may contain multiple graphs). Owns the packed handle
// and the per-graph metadata map.
class HbmSession {
 public:
  HbmSession() = default;
  ~HbmSession();

  HbmSession(const HbmSession &) = delete;
  HbmSession &operator=(const HbmSession &) = delete;

  // Load `hbm_path` into BPU memory. After this, GetGraphNames() returns the
  // packed file's graph name list.
  Result Load(const std::string &hbm_path);

  // Fetch a graph by name. First call per name will lazily refresh its IO
  // metadata. The returned pointer is owned by the session (do not delete).
  Graph *GetGraph(const std::string &name);

  // Convenience: look up the graph by name and run it in one call. Avoids
  // having to plumb the raw C handle out to callers (the Graph object does
  // not expose its handle).
  Result ExecuteGraphByName(const std::string &graph_name,
                            const std::vector<Tensor> &inputs,
                            std::vector<Tensor> *outputs,
                            ExecutionMetrics *metrics = nullptr,
                            const std::vector<OutputSlice> *output_slices = nullptr);
  Result ExecuteGraphByName(const std::string &graph_name,
                            const std::vector<const Tensor *> &inputs,
                            std::vector<Tensor> *outputs,
                            ExecutionMetrics *metrics = nullptr,
                            const std::vector<OutputSlice> *output_slices = nullptr);

  // List of all graph names in this hbm (e.g. ["visual"], or
  // ["prefill", "decode"]).
  const std::vector<std::string> &GetGraphNames() const { return graph_names_; }

 private:
  void *packed_handle_ = nullptr;  // hbDNNPackedHandle_t
  std::vector<std::string> graph_names_;
  std::unordered_map<std::string, std::unique_ptr<Graph>> graphs_;
  Graph *active_graph_ = nullptr;  // owned by graphs_
};

// Helper: number of bytes per element for a given HB_DNN_TENSOR_TYPE_*.
// Mirrors HB_RuntimeUtils.hpp's dtype-size table.
int32_t DtypeElementBytes(int32_t dtype);

// Helper: convert HB_DNN_TENSOR_TYPE_* -> a short human-readable string.
const char *DtypeName(int32_t dtype);

}  // namespace locateanything_runtime

// Copyright (c) 2025 D-Robotics Co,.Ltd. All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of D-Robotics Co,.Ltd. This is proprietary information owned by
// D-Robotics Co,.Ltd. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of D-Robotics Co,.Ltd.

#include <filesystem>
#include <stdexcept>
#include <chrono>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "HB_HBMRuntime.hpp"

using namespace std;

/**
 * @brief Constructor for HB_HBMRuntime that loads a single HBM model from file.
 *
 * @param[in] model_file The file path of the HBM model to load.
 *
 * @throws std::runtime_error If the file does not exist or has an invalid extension.
 */
HB_HBMRuntime::HB_HBMRuntime(const string& model_file)
{
    // Check if the model file exists on the filesystem.
    if (!filesystem::exists(model_file)) {
        throw runtime_error("The input hbm model " + model_file + " does not exist.");
    }

    // Ensure the file has the correct '.hbm' extension.
    if (model_file.substr(model_file.find_last_of('.') + 1) != "hbm") {
        throw runtime_error("Invalid model file extension. Only .hbm supported.");
    }

    // Convert the model path to a C-style string pointer, as required by the SDK API.
    const char* model_cstr = model_file.c_str();
    const char* model_file_cstrs[] = { model_cstr };

    // Initialize the packed DNN model handle using Horizon SDK API.
    // This loads the model into memory and prepares it for inference.
    HBDNN_CHECK_SUCCESS(
        hbDNNInitializeFromFiles(&dnn_packed_handle, model_file_cstrs, 1),
        "hbDNN initialize from " + model_file + " file failed."
    );

    // Load and parse model-related metadata (e.g. input/output layout, quantization info).
    LoadParameters(std::vector<std::string>{model_file});

    // Initialize the scheduling parameters map with default values.
    InitSchedParamMapWithModels(model_sched_params, model_names);
}

/**
 * @brief Constructor for HB_HBMRuntime that loads multiple HBM model files.
 *
 * This constructor initializes the runtime by loading multiple HBM model files.
 * It performs checks on the input file list, verifies file existence and extensions,
 * converts the list to C-style strings, and initializes the DNN runtime handle.
 * Afterwards, it loads model parameters such as input/output metadata and quantization info.
 *
 * @param[in] model_files A vector of file paths to HBM model files to be loaded.
 *
 * @throws std::runtime_error If no model files are provided, any file does not exist,
 *                            or a file has an invalid extension.
 */
HB_HBMRuntime::HB_HBMRuntime(const std::vector<std::string>& model_files)
{
    // Check if the input file list is empty.
    if (model_files.empty()) {
        throw std::runtime_error("No input hbm model files provided.");
    }

    // Verify each file: check existence and ensure it has a '.hbm' extension.
    for (const auto& file : model_files) {
        if (!std::filesystem::exists(file)) {
            throw std::runtime_error("The input hbm model " + file + " does not exist.");
        }
        if (file.substr(file.find_last_of('.') + 1) != "hbm") {
            throw std::runtime_error("Invalid model file extension: " + file + ". Only .hbm supported.");
        }
    }

    // Convert the list of model file paths into an array of C-style strings,
    // as required by the Horizon DNN initialization API.
    std::vector<const char*> model_file_cstrs;
    for (const auto& file : model_files) {
        model_file_cstrs.push_back(file.c_str());
    }

    // Initialize the packed DNN model handle with multiple HBM files.
    // This step loads the model group into memory and prepares it for multi-model inference.
    HBDNN_CHECK_SUCCESS(
        hbDNNInitializeFromFiles(&dnn_packed_handle, model_file_cstrs.data(), model_file_cstrs.size()),
        "hbDNN initialize from multiple .hbm files failed."
    );

    // Load model parameters (e.g., input/output info, quantization, metadata).
    LoadParameters(model_files);

    // Initialize the scheduling parameters map with default values.
    InitSchedParamMapWithModels(model_sched_params, model_names);
}

/**
 * @brief Release the memory allocated for a list of hbDNNTensor objects.
 *
 * This function iterates over each tensor and checks if the virtual address
 * (virAddr) is valid (non-null). If so, it calls hbUCPFree to free the memory
 * allocated for that tensor, and then resets the corresponding pointer and
 * physical address to safe default values.
 *
 * @param tensors A vector of hbDNNTensor objects whose sysMem needs to be freed.
 */
void HB_HBMRuntime::FreeTensorMem(std::vector<hbDNNTensor>& tensors) {
    for (auto& tensor : tensors) {
        if (tensor.sysMem.virAddr != nullptr) {
            // Free the memory associated with the tensor's system memory.
            HBUCP_CHECK_SUCCESS(hbUCPFree(&(tensor.sysMem)), "");

            // Reset memory pointers to avoid dangling references.
            tensor.sysMem.virAddr = nullptr;
            tensor.sysMem.phyAddr = 0;
        }
    }
}

/**
 * @brief Destructor for the HB_HBMRuntime class.
 *
 * This destructor ensures that the DNN packed handle is properly released
 * when the HB_HBMRuntime object is destroyed. This prevents resource leaks
 * related to DNN runtime context allocated by hbDNNInitializeFromFiles.
 */
HB_HBMRuntime::~HB_HBMRuntime()
{
    // Check if the handle was initialized, then release it.
    if (dnn_packed_handle) {
        hbDNNRelease(dnn_packed_handle);
    }
}

/**
 * @brief Convert hbDNN tensor type to NumPy dtype format string.
 *
 * This function maps internal hbDNN data types (hbDNNDataType enums) to
 * Python's NumPy format descriptor characters used to construct py::array.
 *
 * @param[in] hb_type An integer representing the hbDNN tensor data type.
 *
 * @return A string containing a single character NumPy format descriptor.
 *
 * @throws std::invalid_argument if the provided hb_type is unsupported.
 */
std::string HB_HBMRuntime::HbDNNTypeToNumpyFormat(int hb_type) {
    switch (hb_type) {
        case HB_DNN_TENSOR_TYPE_S8:
            return "b";  // int8
        case HB_DNN_TENSOR_TYPE_U8:
            return "B";  // uint8
        case HB_DNN_TENSOR_TYPE_F16:
            return "e";  // float16
        case HB_DNN_TENSOR_TYPE_S16:
            return "h";  // int16
        case HB_DNN_TENSOR_TYPE_U16:
            return "H";  // uint16
        case HB_DNN_TENSOR_TYPE_F32:
            return "f";  // float32
        case HB_DNN_TENSOR_TYPE_S32:
            return "i";  // int32
        case HB_DNN_TENSOR_TYPE_U32:
            return "I";  // uint32
        case HB_DNN_TENSOR_TYPE_F64:
            return "d";  // float64
        case HB_DNN_TENSOR_TYPE_S64:
            return "l";  // int64
        case HB_DNN_TENSOR_TYPE_U64:
            return "L";  // uint64
        case HB_DNN_TENSOR_TYPE_BOOL8:
            return "?";  // boolean
        default:
            throw std::invalid_argument("Unsupported hbDNNDataType");
    }
}

/**
 * @brief Validate and populate dynamic dimensions for model input shape.
 *
 * This function checks whether the input tensor from Python matches
 * the expected model tensor shape in terms of number of dimensions
 * and per-dimension sizes. If the model shape contains -1, it will
 * be filled using the input tensor's shape (dynamic dimension).
 *
 * @param[in]      model_name         Name of the model being validated.
 * @param[in]      input_tensor_name  Name of the input tensor.
 * @param[in,out]  model_shape        The shape structure defined by the model. If it contains -1 in any
 *                                   dimension, it will be updated in-place using the input_array shape.
 * @param[in]      input_array        The actual input array from Python (py::array) to validate against.
 *
 * @return 0 if validation and update succeed.
 *
 * @throws std::runtime_error if the number of dimensions or any fixed-size dimension mismatches.
 */
int32_t HB_HBMRuntime::PrepareInputValidShape(std::string &model_name, std::string &input_tensor_name,
                                              hbDNNTensorShape &model_shape, py::array &input_array)
{
    int ndim = input_array.ndim();
    const ssize_t* input_shape = input_array.shape();

    // Check dimension count
    if (ndim != model_shape.numDimensions) {
        std::ostringstream oss;
        oss << "Mismatch in tensor dimensions for model: " << model_name
            << ", input: " << input_tensor_name
            << ", input data ndim = " << ndim
            << ", model numDimensions = " << model_shape.numDimensions;
        throw std::runtime_error(oss.str());
    }

    // Validate or populate each dimension
    for (int32_t idx{0U}; idx < model_shape.numDimensions; idx++) {
        if (model_shape.dimensionSize[idx] < 0) {
            // Allow dynamic dimension to be inferred from input
            model_shape.dimensionSize[idx] = input_shape[idx];
        } else if (model_shape.dimensionSize[idx] != input_shape[idx]) {
            std::ostringstream oss;
            oss << "Mismatch in tensor dimension at index " << idx
                << " for model: " << model_name
                << ", input: " << input_tensor_name
                << ". input data dim = " << input_shape[idx]
                << ", model expected dim = " << model_shape.dimensionSize[idx];
            throw std::runtime_error(oss.str());
        }
    }

    input_shape_ready = true;
    return 0;
}

/**
 * @brief Prepare and resolve dynamic strides for an input tensor.
 *
 * This function fills in dynamic stride values based on the shape,
 * assuming row-major layout with 64-byte alignment requirements.
 * The last dimension's stride must be static, as dynamic stride
 * calculation depends on it.
 *
 * If input shape validation has not been done previously, it will be invoked.
 *
 * @param[in]  model_name        The name of the model.
 * @param[in]  input_tensor_name The name of the input tensor.
 * @param[in,out] shape          The shape structure of the tensor (with dimension sizes).
 * @param[in,out] stride         Pointer to stride array. Elements with negative values
 *                              are considered dynamic and will be computed in-place.
 * @param[in]  input_array       The actual input array from Python, passed in for shape validation.
 *
 * @return 0 on success.
 *
 * @throws std::runtime_error if the last dimension has a dynamic (negative) stride,
 *         or if input shape validation fails.
 */
int32_t HB_HBMRuntime::PrepareInputStride(std::string &model_name, std::string &input_tensor_name,
                                          hbDNNTensorShape &shape, int64_t *stride, py::array &input_array)
{
    // Ensure shape is valid before computing stride
    if (!input_shape_ready) {
        PrepareInputValidShape(model_name, input_tensor_name, shape, input_array);
    }

    // Fill dynamic strides from innermost to outermost dimension
    for (int32_t idx = shape.numDimensions - 1; idx >= 0; idx--) {
        if (stride[idx] < 0) {
            if (idx == shape.numDimensions - 1) {
                // Last dimension must have a fixed stride
                throw std::runtime_error("Unsupported model: The last dimension of stride cannot be dynamic.");
            }
            // Compute stride based on next dimension and 64-byte alignment
            stride[idx] = ALIGN_64(stride[idx + 1] * shape.dimensionSize[idx + 1]);
        }
    }

    return 0;
}

/**
 * @brief Prepare input tensor memory and copy data from Python array.
 *
 * This function allocates cached memory for an input tensor using hbUCPMallocCached,
 * handles padding if necessary, and flushes the memory to ensure consistency
 * before inference. If the user input array's total size doesn't match the
 * model's aligned memory layout, appropriate padding will be applied.
 *
 * @param[in]  model_name         The name of the model.
 * @param[in]  input_tensor_name  The name of the input tensor.
 * @param[out] input_tensor             The destination hbDNNTensor object to fill data into.
 * @param[in]  input_array        The input tensor data passed from Python.
 *
 * @return 0 on success.
 *
 * @throws std::runtime_error if memory allocation or data copy fails.
 */
int32_t HB_HBMRuntime::PrepareInputTensorData(std::string &model_name, std::string &input_tensor_name,
                                              hbDNNTensor &input_tensor, py::array & input_array)
{
    auto& properties   = input_tensor.properties;
    auto type          = properties.tensorType;
    auto& valid_shape  = properties.validShape;
    auto& strides      = properties.stride;
    auto& mem          = input_tensor.sysMem;

    // Calculate raw data length (in bytes) from input array
    auto length = input_array.itemsize() * get_array_total_elements(input_array);

    // Get element size and compute aligned size
    int32_t elem_size = get_element_size(type);
    auto aligned_size = (properties.alignedByteSize < 0)
                        ? strides[0] * valid_shape.dimensionSize[0]
                        : properties.alignedByteSize;

    // Allocate cached memory for input_tensor
    HBUCP_CHECK_SUCCESS(hbUCPMallocCached(&mem, aligned_size, 0),
                        model_name + input_tensor_name + " failed to allocate cached memory via hbUCPMallocCached.");

    if (length == aligned_size) {
        // Fast path: directly copy input data into allocated memory
        memcpy(reinterpret_cast<char *>(mem.virAddr),
               reinterpret_cast<const char *>(input_array.data()), length);
    } else {
        // Padding required: shape mismatch between input and aligned memory layout
        std::vector<uint32_t> real_dim(valid_shape.numDimensions);
        for (int32_t idx = 0; idx < valid_shape.numDimensions; ++idx) {
            real_dim[idx] = static_cast<uint32_t>(valid_shape.dimensionSize[idx]);
        }

        // Use custom padding utility to copy and pad data into target buffer
        if (add_padding(mem.virAddr, reinterpret_cast<const char *>(input_array.data()),
                        static_cast<uint32_t>(valid_shape.numDimensions),
                        real_dim.data(), properties.stride,
                        static_cast<uint32_t>(elem_size)) != 0) {
            std::ostringstream oss;
            oss << "add_padding interface failed for model: " << model_name
                << ", input tensor: " << input_tensor_name;
            throw std::runtime_error(oss.str());
        }
    }

    // Flush memory to make sure it's ready for device access
    HBUCP_CHECK_SUCCESS(hbUCPMemFlush(&mem, HB_SYS_MEM_CACHE_CLEAN), " ");

    return 0;
}

/**
 * @brief Check if the input tensor's numpy dtype matches the expected HB DNN tensor type.
 *
 * This function compares the expected data type (from the model's definition)
 * against the actual numpy dtype of the input array provided by the user.
 * It converts both into numpy format codes and throws an exception if they differ.
 *
 * @param[in] model_name       The name of the model the tensor belongs to.
 * @param[in] tensor_name      The name of the input tensor to be validated.
 * @param[in] expected_hb_type Expected data type defined in hbDNN (e.g., HB_DNN_TENSOR_TYPE_F32).
 * @param[in] input_array      Numpy array from Python input.
 *
 * @throws std::runtime_error If the data type of the input array does not match the expected format.
 */
void HB_HBMRuntime::CheckInputDtypeCompatibility(const std::string& model_name,
                                                 const std::string& tensor_name,
                                                 int expected_hb_type,
                                                 const py::array& input_array)
{
    // Convert Python dtype (char code) to string, e.g., 'f' for float32
    std::string actual_format = std::string(1, input_array.dtype().char_());

    // Convert expected HB DNN type to corresponding numpy dtype char code
    std::string expected_format = HbDNNTypeToNumpyFormat(expected_hb_type);

    // If types do not match, throw error with detailed message
    if (actual_format != expected_format) {
        throw std::runtime_error("Data type mismatch for input tensor '" + tensor_name +
                                 "' in model '" + model_name +
                                 "': expected numpy dtype format '" + expected_format +
                                 "', but received '" + actual_format + "'");
    }
}

/**
 * @brief Prepare input tensors for a given model using provided input data.
 *
 * This function iterates over all input tensors of the specified model,
 * retrieves their properties from the underlying DNN handle,
 * validates the input data type and shape,
 * prepares the tensor strides,
 * and copies (or pads) the input data into device memory.
 *
 * @param[in]  model_name         Name of the model whose inputs are being prepared.
 * @param[out] input_tensors Vector to store prepared hbDNNTensor structures for the model inputs.
 * @param[in]  input_arrays   Map from input tensor name to numpy array (input data).
 *
 * @return int32_t Returns 0 on success.
 *
 * @throws std::runtime_error If any step fails, including tensor property query,
 *         input data validation, or memory preparation.
 */
int32_t HB_HBMRuntime::PrepareInputTensor(std::string model_name, std::vector<hbDNNTensor> &input_tensors,
                                          std::unordered_map<std::string, py::array>& input_arrays)
{
    // Loop over all input tensors for the model
    for (int32_t input_tensor_idx{0}; input_tensor_idx < input_counts[model_name]; input_tensor_idx++) {
        // Get the tensor name by index
        std::string input_name = input_names[model_name][input_tensor_idx];

        // Get the input data numpy array from user input
        py::array & input_array = input_arrays[input_name];

        // Create a new hbDNNTensor object and append to vector
        hbDNNTensor tensor;
        input_tensors.push_back(tensor);

        // Reference to tensor properties for convenience
        hbDNNTensorProperties &input_properties{input_tensors[input_tensor_idx].properties};

        // Query tensor properties from the DNN handle
        HBDNN_CHECK_SUCCESS(hbDNNGetInputTensorProperties(&input_properties, dnn_handle_list[model_name], input_tensor_idx),
                            "Failed to get tensor properties for model: " + model_name + ", input: " + input_name);

        // Verify input data type compatibility (numpy dtype vs model expected type)
        CheckInputDtypeCompatibility(model_name, input_name, input_properties.tensorType, input_array);

        // Validate and update valid input shape based on input data
        PrepareInputValidShape(model_name, input_name, input_properties.validShape, input_array);

        // Calculate and set tensor strides based on shape and input data
        PrepareInputStride(model_name, input_name, input_properties.validShape, input_properties.stride, input_array);

        // Allocate memory and copy (or pad) input data to device buffer
        PrepareInputTensorData(model_name, input_name, input_tensors[input_tensor_idx], input_array);
    }

    return 0;
}

/**
 * @brief Prepare output tensors for a given model by allocating device memory.
 *
 * This function iterates over all output tensors of the specified model,
 * queries their properties from the DNN handle,
 * allocates cached memory for each output tensor according to its aligned byte size,
 * and stores the prepared tensors in the provided vector.
 *
 * @param[in]  model_name          Name of the model whose outputs are being prepared.
 * @param[out] output_tensors Vector to store prepared hbDNNTensor structures for the model outputs.
 *
 * @return int32_t Returns 0 on success.
 *
 * @throws std::runtime_error If retrieving tensor properties or memory allocation fails.
 */
int32_t HB_HBMRuntime::PrepareOutputTensor(std::string model_name, std::vector<hbDNNTensor> & output_tensors)
{
    auto it = output_counts.find(model_name);
    if (it == output_counts.end()) {
        throw std::runtime_error("Unknown model name: " + model_name);
    }

    for (int32_t idx{0}; idx < output_counts[model_name]; idx++) {
        std::string tensor_name = output_names[model_name][idx];

        output_tensors.emplace_back();  // 添加新 tensor
        hbDNNTensor& tensor = output_tensors.back();
        hbDNNTensorProperties& tensor_props = tensor.properties;

        // Retrieve output tensor properties from the DNN handle
        HBDNN_CHECK_SUCCESS(hbDNNGetOutputTensorProperties(&tensor_props, dnn_handle_list[model_name], idx),
                            "Failed to get tensor properties for model: " + model_name + ", output: " + tensor_name);

        // Allocate cached memory for the output tensor
        HBUCP_CHECK_SUCCESS(hbUCPMallocCached(&tensor.sysMem, tensor_props.alignedByteSize, 0),
                            model_name + tensor_name + " failed to allocate cached memory via hbUCPMallocCached.");
    }
    return 0;
}

/**
 * @brief Create a numpy array wrapping a hbUCPSysMem buffer with automatic memory management.
 *
 * This function wraps the memory buffer described by `hbUCPSysMem` into a numpy array.
 * The memory ownership is transferred to a Python capsule that will free the memory
 * using `hbUCPFree` when the numpy array is garbage collected.
 *
 * @param[in]  dtype    The numpy dtype of the array elements.
 * @param[in]  shape    The shape of the numpy array.
 * @param[in]  strides  The strides of the numpy array (in bytes).
 * @param[in]  sys_mem  Reference to the hbUCPSysMem struct containing the buffer.
 *
 * @return py::array A numpy array sharing the memory buffer.
 *
 * @note The function makes a heap copy of `hbUCPSysMem` and passes it to the capsule
 *       for lifecycle management to ensure safe memory release.
 */
py::array MakeNumpyArrayWithCapsule(
    const py::dtype& dtype,
    const std::vector<ssize_t>& shape,
    const std::vector<ssize_t>& strides,
    hbUCPSysMem &sys_mem
    )
{
    // Allocate copy of hbUCPSysMem to pass ownership to Python capsule
    hbUCPSysMem* mem_copy = new hbUCPSysMem(sys_mem );

    // Create a capsule that will free the hbUCPSysMem memory on destruction
    auto capsule = py::capsule(mem_copy, [](void* ptr) {
        auto* mem = static_cast<hbUCPSysMem*>(ptr);
        int ret = hbUCPFree(mem);
        if (ret != 0) {
            std::cerr << "Free memory failed. ret: " << ret << std::endl;
        }
        delete mem;
    });

    // Return numpy array that shares the memory buffer and owns the capsule
    return py::array(dtype, shape, strides, sys_mem .virAddr, capsule);
}

/**
 * @brief Prepare Python numpy output arrays from model output tensors.
 *
 * For each output tensor of the specified model, this function:
 * - Extracts the tensor shape and strides from the hbDNNTensor properties.
 * - Creates a corresponding numpy dtype based on the tensor's data type.
 * - Wraps the tensor's system memory buffer into a numpy array with correct shape and strides.
 * - Inserts the numpy array into the provided map keyed by the output tensor name.
 *
 * @param[in]  model_name         The name of the model whose outputs are being processed.
 * @param[out] model_output_arrays A map to store output tensor name to numpy array.
 * @param[in]  output_tensors A vector of hbDNNTensors representing the model output tensors.
 *
 * @return int32_t Returns 0 on success.
 */
int32_t HB_HBMRuntime::PrepareOutputArrays(std::string model_name, std::unordered_map<std::string, py::array> & model_output_arrays,
                            std::vector<hbDNNTensor> & output_tensors)
{
    for (int32_t output_tensor_idx{0}; output_tensor_idx < output_counts[model_name]; output_tensor_idx++) {
        std::string tensor_name = output_names[model_name][output_tensor_idx];

        hbDNNTensorProperties &output_properties{output_tensors[output_tensor_idx].properties};

        // Extract shape and strides for numpy array construction
        std::vector<ssize_t> shape(output_properties.validShape.numDimensions);
        std::vector<ssize_t> strides(output_properties.validShape.numDimensions);

        for (int i = 0; i < output_properties.validShape.numDimensions; ++i) {
            shape[i] = static_cast<ssize_t>(output_properties.validShape.dimensionSize[i]);
            strides[i] = static_cast<ssize_t>(output_properties.stride[i]);
        }

        // Create numpy dtype matching tensor data type
        py::dtype dtype = py::dtype(HbDNNTypeToNumpyFormat(output_properties.tensorType));

        // Create numpy array that wraps hbDNNTensor's system memory with automatic memory management
        py::array np_array = MakeNumpyArrayWithCapsule(dtype, shape, strides, output_tensors[output_tensor_idx].sysMem);

        // Store in output map with output tensor name as key
        model_output_arrays[tensor_name] = np_array;
    }
    return 0;
}

/**
 * @brief Determine the BPU core mask to be used for the given model based on user-provided core IDs.
 *
 * This function looks up the BPU core IDs assigned to a specific model and returns a 64-bit bitmask
 * indicating which cores the model should run on. If no valid cores are specified, it defaults to
 * HB_UCP_BPU_CORE_ANY to allow automatic scheduling.
 *
 * @param[in] model_name    The name of the model for which the BPU core mask is requested.
 * @param[in] bpu_cores  A map from model names to lists of user-specified BPU core IDs (valid values are 0~3 or -1).
 *
 * @return A 64-bit mask with bits set for each core ID the model should run on.
 *         Returns HB_UCP_BPU_CORE_ANY if no core is specified or if the list contains only -1.
 *
 * @throws std::runtime_error if any core ID is outside the valid range [0, 3] or not -1.
 */
uint64_t HB_HBMRuntime::GetBPUCoreMaskForModel(const std::string& model_name,
                                                const std::vector<int>& bpu_cores)
{
    if (bpu_cores.empty()) {
        // Empty list -> still means ANY
        return HB_UCP_BPU_CORE_ANY;
    }

    // Check for the special case: single -1 value = ANY
    if (bpu_cores.size() == 1 && bpu_cores[0] == -1) {
        return HB_UCP_BPU_CORE_ANY;
    }

    // Build core bitmask explicitly from user-specified core IDs
    uint64_t mask = 0;
    for (int32_t id : bpu_cores) {
        if (id < 0 || id > 3) {
            throw std::runtime_error("Invalid BPU core id " + std::to_string(id) +
                                     " for model " + model_name + ". Expected 0~3.");
        }
        mask |= (1ULL << id);  // set the bit for the core ID
    }

    // Note: do not mix explicit core mask with HB_UCP_BPU_CORE_ANY
    return mask;
}

/**
 * @brief Perform inference for a single model with given input/output tensors.
 *
 * This function submits an inference task to the DNN engine, sets scheduling parameters
 * such as priority, device ID, custom ID, and BPU core mask, then waits for task completion.
 * After inference, it updates output tensor properties and handles cache flushing.
 *
 * @param[in] model_name       Name of the model to run inference on.
 * @param[in] dnn_handle       Handle to the loaded DNN model.
 * @param[in,out] input_tensors   Prepared input tensors for the model.
 * @param[in,out] output_tensors  Output buffers where inference results will be stored.
 */
void HB_HBMRuntime::InferSingleModel(const std::string& model_name,
                 hbDNNHandle_t& dnn_handle,
                 std::vector<hbDNNTensor>& input_tensors,
                 std::vector<hbDNNTensor>& output_tensors)
{
    // Task handle for UCP scheduler
    hbUCPTaskHandle_t task_handle = nullptr;

    // Start timing the inference
    auto start = std::chrono::high_resolution_clock::now();

    // Submit the inference request to DNN engine
    HBDNN_CHECK_SUCCESS(hbDNNInferV2(&task_handle, output_tensors.data(), input_tensors.data(), dnn_handle),
                        " hbDNNInferV2 failed");

    // Configure scheduler parameters
    hbUCPSchedParam sched_param{};
    HB_UCP_INITIALIZE_SCHED_PARAM(&sched_param);
    sched_param.priority = model_sched_params[model_name].priority;
    sched_param.deviceId = model_sched_params[model_name].deviceId;
    sched_param.customId = model_sched_params[model_name].customId;
    sched_param.backend = GetBPUCoreMaskForModel(model_name, model_sched_params[model_name].bpu_cores);  // Set BPU core mask

    // Submit task to UCP for execution
    HBUCP_CHECK_SUCCESS(hbUCPSubmitTask(task_handle, &sched_param),
                        model_name + " hbUCPSubmitTask failed");

    // Wait for the task to complete (synchronously)
    HBUCP_CHECK_SUCCESS(hbUCPWaitTaskDone(task_handle, 0),
                        model_name + " hbUCPWaitTaskDone failed");

    // End timing and print elapsed time
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // std::cout << "hbDNNInferV2 inference time: " << duration_ms << " ms" << std::endl;

    // After inference, flush the memory and fetch updated output tensor properties
    for (int output_tensor_idx = 0; output_tensor_idx < output_tensors.size(); output_tensor_idx++) {
        // Invalidate cache for output buffer
        HBUCP_CHECK_SUCCESS(hbUCPMemFlush(&output_tensors[output_tensor_idx].sysMem, HB_SYS_MEM_CACHE_INVALIDATE), " ");

        std::string tensor_name = output_names[model_name][output_tensor_idx];
        hbDNNTensorProperties &output_properties = output_tensor_properties[model_name][tensor_name];

        // Retrieve runtime-updated output tensor properties (e.g., shape, size)
        HBDNN_CHECK_SUCCESS(hbDNNGetTaskOutputTensorProperties(&output_properties, task_handle, 0, output_tensor_idx),
                            model_name + " hbDNNGetTaskOutputTensorProperties failed!");
    }

    // Release task resources
    HBUCP_CHECK_SUCCESS(hbUCPReleaseTask(task_handle),
                        model_name + " hbUCPReleaseTask failed");
}

/**
 * @brief Launch inference tasks for all models in parallel using multithreading.
 *
 * Each model is assigned a separate thread to perform inference via `InferSingleModel`.
 *
 * @param[in] input_tensors   Map containing input tensors for each model.
 * @param[in,out] output_tensors Map to hold output tensors for each model after inference.
 *
 * @return Returns 0 on success.
 */
int32_t HB_HBMRuntime::LaunchInferenceTasks(
    std::unordered_map<std::string, std::vector<hbDNNTensor>> & input_tensors,
    std::unordered_map<std::string, std::vector<hbDNNTensor>> & output_tensors)
{
    std::vector<std::thread> threads;

    // Launch inference threads for each model
    for (const auto& model_pair : input_tensors) {
        const std::string& model_name = model_pair.first;

        // Start a new thread to perform inference for this model
        threads.emplace_back([this, model_name, &input_tensors, &output_tensors]() {
            InferSingleModel(
                model_name,
                dnn_handle_list.at(model_name),  // Fetch handle for the model
                input_tensors[model_name],        // Input tensor list
                output_tensors[model_name]        // Output tensor list (to be filled)
            );
        });
    }

    // Wait for all threads to complete before returning
    for (auto& thread : threads) {
        thread.join();
    }

    return 0;
}

/**
 * @brief Perform inference for all loaded models using provided input tensors.
 *
 * This function orchestrates the preparation, execution, and result collection for
 * multiple models in parallel. It handles input tensor preparation, output memory allocation,
 * launches inference tasks (in parallel via threads), gathers results as numpy arrays,
 * and frees allocated memory.
 *
 * @param[in] multi_input_tensors Nested map of input numpy arrays for each model.
 *                             Outer key: model name, Inner key: tensor name, Value: py::array.
 *
 * @return Nested map containing output numpy arrays for each model.
 *         Outer key: model name, Inner key: output tensor name, Value: py::array.
 */
std::unordered_map<std::string, std::unordered_map<std::string, py::array>>
HB_HBMRuntime::InferAllModels(
    std::unordered_map<std::string, std::unordered_map<std::string, py::array>>& multi_input_tensors)
{
    // Map to hold input tensor structures for each model
    std::unordered_map<std::string, std::vector<hbDNNTensor>> input_tensors;

    // Map to hold output tensor structures for each model
    std::unordered_map<std::string, std::vector<hbDNNTensor>> output_tensors;

    // Final result map: model name -> {tensor name -> py::array}
    std::unordered_map<std::string, std::unordered_map<std::string, py::array>> result;

    // Step 1: Prepare input and output tensors for each model
    for (const auto& model_pair : multi_input_tensors) {
        const std::string& model_name = model_pair.first;

        // Prepare input tensors (allocate and copy numpy data)
        PrepareInputTensor(model_name, input_tensors[model_name], multi_input_tensors[model_name]);

        // Prepare output tensors (allocate memory)
        PrepareOutputTensor(model_name, output_tensors[model_name]);
    }

    // Step 2: Launch all inference tasks in parallel
    LaunchInferenceTasks(input_tensors, output_tensors);

    // Step 3: Convert output tensors to numpy arrays and store in result
    for (const auto& model_pair : multi_input_tensors) {
        const std::string& model_name = model_pair.first;

        PrepareOutputArrays(model_name, result[model_name], output_tensors[model_name]);

        // Free input tensor memory (output tensor memory is freed automatically via numpy capsule)
        FreeTensorMem(input_tensors[model_name]);
    }

    return result;
}

/**
 * @brief Parse and validate the model name from the given extra arguments.
 *
 * This function extracts the "model_name" key from the provided extra arguments map.
 * If the key is present, it verifies that the value is a string and matches one of the loaded models.
 * If the key is absent, it assumes exactly one model is loaded and returns its name.
 * If multiple models are loaded but no model name is provided, it throws an error due to ambiguity.
 *
 * @param[in] extra_args       A map of extra arguments, where "model_name" may be included.
 *                   The value type is a variant that can hold multiple types.
 * @param[in] model_list A vector containing all currently loaded model names.
 *
 * @return A validated model name string to be used for inference or other operations.
 *
 * @throws std::runtime_error If:
 *         - "model_name" is present but not a string.
 *         - "model_name" is not found in the loaded model list.
 *         - "model_name" is missing and multiple models are loaded.
 */
std::string HB_HBMRuntime::ParseAndValidateModelName(
    const ExtraArgs& extra_args,
    const std::vector<std::string>& model_list)
{
    // Try to find "model_name" key in extra arguments
    auto it = extra_args.find("model_name");
    if (it != extra_args.end()) {
        // Check if the value is a string
        if (!std::holds_alternative<std::string>(it->second)) {
            throw std::runtime_error("extra_args['model_name'] must be a string.");
        }

        const std::string& model_name = std::get<std::string>(it->second);

        // Check if the provided model name is in the list of loaded models
        if (std::find(model_list.begin(), model_list.end(), model_name) == model_list.end()) {
            throw std::runtime_error("Provided model_name '" + model_name + "' is not in the loaded model list.");
        }

        return model_name;
    }

    // If no model name provided and multiple models are loaded, raise an error
    if (model_list.size() != 1) {
        throw std::runtime_error("extra_args does not provide model_name, but multiple models are loaded (" +
                                 std::to_string(model_list.size()) + ").");
    }

    // Default case: only one model is loaded, return it
    return model_list.front();
}

/**
 * @brief Validate input tensor names for one or more models before inference.
 *
 * This function checks that all model names and their corresponding input tensor names
 * in the user-provided `multi_input_tensors` are valid and registered.
 *
 * Specifically:
 * - It verifies that each model in `multi_input_tensors` exists in the registered `model_list`.
 * - It ensures each model has corresponding input name registrations in `input_names`.
 * - It checks that all user-provided input tensor names match the expected names.
 *
 * @param[in] multi_input_tensors A nested map of model name -> (input name -> numpy array),
 *                         representing the input tensors for inference.
 * @param[in] model_list       A list of model names that have been loaded and registered.
 * @param[in] input_names      A map of model name -> list of valid input tensor names.
 *
 * @throws std::runtime_error if:
 *         - A model name in `multi_input_tensors` is not found in `model_list`.
 *         - No input names are registered for a model.
 *         - Any input tensor name is not recognized for its model.
 *
 * @note This function performs structural validation only. It does not check tensor shapes or data types.
 */
void HB_HBMRuntime::CheckInputInfoValid(
    const std::unordered_map<std::string, std::unordered_map<std::string, py::array>>& multi_input_tensors,
    const std::vector<std::string>& model_list,
    const std::unordered_map<std::string, std::vector<std::string>>& input_names)
{
    // Iterate over each model's input map
    for (const auto& [model_name, inputs] : multi_input_tensors) {
        // Check whether the model name is in the registered list
        if (std::find(model_list.begin(), model_list.end(), model_name) == model_list.end()) {
            throw std::runtime_error("Model name \"" + model_name + "\" is not in the registered model_names.");
        }

        // Check whether input names for this model are registered
        auto it = input_names.find(model_name);
        if (it == input_names.end()) {
            throw std::runtime_error("Input names for model \"" + model_name + "\" are not registered.");
        }

        const auto& expected_input_names = it->second;

        // Validate all provided input tensor names against the registered ones
        for (const auto& [input_name, _] : inputs) {
            if (std::find(expected_input_names.begin(), expected_input_names.end(), input_name) == expected_input_names.end()) {
                throw std::runtime_error("Input name \"" + input_name + "\" is invalid for model \"" + model_name + "\".");
            }
        }
    }
}

/**
 * @brief Validate or filter input models based on the "model_name" field in extra_args.
 *
 * This function enforces model selection rules depending on the presence of "model_name" in extra_args.
 * - If "model_name" is not provided, the function does nothing (all models are allowed).
 * - If "model_name" is provided, it verifies that the name exists in multi_input_tensors,
 *   and filters the map to retain only the selected model.
 *
 * This ensures that downstream inference logic is only applied to the user-specified model.
 *
 * @param[in,out] multi_input_tensors A mutable map of model name -> (input name -> numpy array).
 *                         May initially contain multiple models.
 * @param[in] extra_args       Extra arguments provided by the user. May optionally contain:
 *                         - "model_name" : string, specifies a single model to be used.
 *
 * @throws std::runtime_error if:
 *         - "model_name" is present but not a string
 *         - "multi_input_tensors" is empty when "model_name" is provided
 *         - The specified "model_name" does not match the only available model
 *         - The specified "model_name" is not present in the input map
 *
 * @note This function modifies multi_input_tensors in-place if filtering is applied.
 */
void HB_HBMRuntime::ValidateOrFilterModelName(
    std::unordered_map<std::string, std::unordered_map<std::string, py::array>>& multi_input_tensors,
    const ExtraArgs& extra_args)
{
    // Check if model_name is specified in extra_args
    auto model_name_it = extra_args.find("model_name");
    if (model_name_it == extra_args.end()) {
        // Lenient mode: allow multiple models to be used without restriction
        return;
    }

    // Validate that model_name is a string
    if (!std::holds_alternative<std::string>(model_name_it->second)) {
        throw std::runtime_error("extra_args['model_name'] must be a string.");
    }

    const std::string& model_name = std::get<std::string>(model_name_it->second);

    if (multi_input_tensors.empty()) {
        throw std::runtime_error("multi_input_tensors is empty, cannot match model_name: " + model_name);
    }

    // If only one model is provided but doesn't match the specified name
    if (multi_input_tensors.size() == 1) {
        const std::string& only_model = multi_input_tensors.begin()->first;
        if (only_model != model_name) {
            throw std::runtime_error("Only one model provided, but model_name argument \"" + model_name +
                                     "\" does not match multi_input_tensors model \"" + only_model + "\".");
        }
        return;  // The only model matches the requested name
    }

    // Validate the specified model exists in the map
    if (multi_input_tensors.find(model_name) == multi_input_tensors.end()) {
        throw std::runtime_error("Model name \"" + model_name + "\" not found in multi_input_tensors.");
    }

    // Filter out all models except the specified one
    for (auto it = multi_input_tensors.begin(); it != multi_input_tensors.end(); ) {
        if (it->first != model_name) {
            it = multi_input_tensors.erase(it);  // Remove unmatched model
        } else {
            ++it;
        }
    }
}

/**
 * @brief Set the scheduling priorities for models.
 *
 * Validates model names and priority values before updating the internal scheduling parameters.
 *
 * @param[in] priority  Map from model name to priority value (0-255).
 *
 * @throws std::runtime_error if a model name is invalid or priority is out of range.
 */
void HB_HBMRuntime::SetModelPriorities(const std::unordered_map<std::string, int>& priority)
{
    for (const auto& [model_name, value] : priority) {
        // Validate model existence
        if (std::find(model_names.begin(), model_names.end(), model_name) == model_names.end()) {
            throw std::runtime_error("Invalid model name in priority setting: \"" + model_name +
                                     "\". This model is not in the loaded model list.");
        }

        // Validate priority range
        if (value < 0 || value > 255) {
            throw std::runtime_error("Priority value for model \"" + model_name + "\" is out of valid range [0, 255]: " +
                                     std::to_string(value));
        }

        // Update the priority if valid
        model_sched_params[model_name].priority = value;
    }
}

/**
 * @brief Set the BPU core assignments for each model.
 *
 * This function validates the model names and BPU core IDs before updating the scheduling parameters.
 * Each model is assigned a vector of BPU cores to run on. Valid core IDs are in the range [0, 3].
 *
 * @param[in] bpu_cores A map from model name to a vector of BPU core IDs.
 *
 * @throws std::runtime_error If a model name is not found in the loaded model list or
 *         if any core ID is outside the valid range [0, 3].
 */
void HB_HBMRuntime::SetModelBpuCores(
    const std::unordered_map<std::string, std::vector<int32_t>>& bpu_cores)
{
    for (const auto& [model_name, core_ids] : bpu_cores) {
        // Check if the model name is valid
        if (std::find(model_names.begin(), model_names.end(), model_name) == model_names.end()) {
            throw std::runtime_error("Invalid model name in bpu_cores: \"" + model_name +
                                     "\". This model is not in the loaded model list.");
        }

        // Check if each core ID is within the valid range [0, 3]
        for (int32_t core_id : core_ids) {
            if (core_id < 0 || core_id > 3) {
                throw std::runtime_error("Invalid BPU core ID for model \"" + model_name + "\": " +
                                         std::to_string(core_id) + " (valid range: [0, 3])");
            }
        }
        // Update the BPU cores assignment for the model
        model_sched_params[model_name].bpu_cores = core_ids;
    }
}

/**
 * @brief Set custom IDs for each model.
 *
 * This function updates the customId field in the scheduling parameters
 * for the specified models after validating their existence.
 *
 * @param[in] custom_id A map from model name to custom ID value.
 *
 * @throws std::runtime_error If any model name is not in the loaded model list.
 */
void HB_HBMRuntime::SetCustomIds(const std::unordered_map<std::string, int64_t>& custom_id)
{
    for (const auto& [model_name, value] : custom_id) {
        if (std::find(model_names.begin(), model_names.end(), model_name) == model_names.end()) {
            throw std::runtime_error("Invalid model name in custom_id setting: \"" + model_name + "\". "
                                     "This model is not in the loaded model list.");
        }

        model_sched_params[model_name].customId = value;
    }
}

/**
 * @brief Set device IDs for each model.
 *
 * This function updates the deviceId field in the scheduling parameters
 * for the specified models after verifying their presence.
 *
 * @param[in] device_id A map from model name to device ID value.
 *
 * @throws std::runtime_error If any model name is not in the loaded model list.
 */
void HB_HBMRuntime::SetDeviceIds(const std::unordered_map<std::string, uint32_t>& device_id)
{
    for (const auto& [model_name, value] : device_id) {
        if (std::find(model_names.begin(), model_names.end(), model_name) == model_names.end()) {
            throw std::runtime_error("Invalid model name in device_id setting: \"" + model_name + "\". "
                                     "This model is not in the loaded model list.");
        }

        model_sched_params[model_name].deviceId = value;
    }
}

/**
 * @brief Set scheduling parameters for multiple models.
 *
 * This function updates scheduling parameters such as priority, BPU cores,
 * custom IDs, and device IDs for models if the corresponding optional
 * arguments are provided.
 *
 * @param[in] priority   Optional map of model names to priority values. Updates model priorities if set.
 * @param[in] bpu_cores  Optional map of model names to vectors of BPU core IDs. Updates BPU core allocation.
 * @param[in] custom_id  Optional map of model names to custom ID values. Updates model custom IDs.
 * @param[in] device_id  Optional map of model names to device ID values. Updates model device IDs.
 */
void HB_HBMRuntime::SetSchedulingParams(
        const std::optional<std::unordered_map<std::string, int32_t>>& priority,
        const std::optional<std::unordered_map<std::string, std::vector<int32_t>>>& bpu_cores,
        const std::optional<std::unordered_map<std::string, int64_t>>& custom_id,
        const std::optional<std::unordered_map<std::string, uint32_t>>& device_id)
{
    if (priority.has_value()) {  // Set model priorities if provided
        SetModelPriorities(priority.value());
    }
    if (bpu_cores.has_value()) {  // Set BPU core allocations if provided
        SetModelBpuCores(bpu_cores.value());
    }
    if (custom_id.has_value()) {  // Set custom IDs if provided
        SetCustomIds(custom_id.value());
    }
    if (device_id.has_value()) {  // Set device IDs if provided
        SetDeviceIds(device_id.value());
    }
}

/**
 * @brief Run inference for a single model with a single input tensor.
 *
 * This simplified API is intended for models that accept exactly one input tensor.
 * It automatically selects the correct model (if only one is loaded), wraps the input
 * tensor into a tensor map, and delegates to the more general `run()` interface.
 *
 * @param[in] input_tensor A single input tensor (NumPy array). Must match expected shape and dtype.
 * @param[in] extra_args   Optional arguments such as:
 *                     - "model_name" (required if multiple models are loaded)
 *
 * @return A nested map from model_name to a map of output_name -> output tensor (NumPy array).
 *
 * @throws std::runtime_error if:
 *         - More than one model is loaded but "model_name" is not specified.
 *         - The selected model has more than one input.
 *         - The specified model name is invalid or not found.
 */
std::unordered_map<std::string, std::unordered_map<std::string, py::array>>
HB_HBMRuntime::run(py::array input_tensor, const ExtraArgs& extra_args)
{
    // Extract the model name, or validate against the only one
    std::string model_name = ParseAndValidateModelName(extra_args, model_names);

    // Check that this model has exactly one input tensor
    if (input_names[model_name].size() != 1) {
        throw std::runtime_error("Single-input run() requires the model to have exactly 1 input, but model '" +
                                 model_name + "' has " + std::to_string(input_names[model_name].size()) + " inputs.");
    }

    // Wrap input_tensor into a map: {input_name -> tensor}
    std::unordered_map<std::string, py::array> input_tensors;
    input_tensors[input_names[model_name][0]] = input_tensor;

    // Inject model name into extra_args to ensure correctness downstream
    ExtraArgs modified_args = extra_args;
    modified_args["model_name"] = model_name;

    // Call the generic run() method that supports multiple inputs per model
    return run(input_tensors, modified_args);
}

/**
 * @brief Run inference for a single model with multiple input tensors.
 *
 * This API is designed for models that require multiple input tensors. The user provides
 * a flat map of input tensor name to NumPy array. Internally, the inputs are wrapped
 * into a nested structure and passed to the generic multi-model `run()` function.
 *
 * @param[in] input_tensors A flat map: input_name -> py::array (NumPy) for a single model.
 * @param[in] extra_args Additional arguments. Required to contain "model_name" if multiple models are loaded.
 *
 * @return A nested map: model_name -> (output_name -> py::array)
 *
 * @throws std::runtime_error if:
 *         - "model_name" is not specified when multiple models are loaded.
 *         - The given model name is not in the loaded model list.
 */
std::unordered_map<std::string, std::unordered_map<std::string, py::array>>
HB_HBMRuntime::run(std::unordered_map<std::string, py::array>& input_tensors,
                   const ExtraArgs& extra_args)
{
    // Extract model name from extra_args or validate against model_names
    std::string model_name = ParseAndValidateModelName(extra_args, model_names);

    // Wrap input tensors into nested structure: { model_name: { input_name: input_tensor } }
    std::unordered_map<std::string, std::unordered_map<std::string, py::array>> nested_input_info;
    nested_input_info[model_name] = input_tensors;

    // Ensure "model_name" is available in extra_args for downstream processing
    ExtraArgs modified_args = extra_args;
    modified_args["model_name"] = model_name;

    // Call the generic multi-model, multi-input run()
    return run(nested_input_info, modified_args);
}

/**
 * @brief Run inference for one or multiple models with full input specification.
 *
 * This is the core multi-model inference entry point.
 * The user provides:
 *   - A nested input map: model_name -> (input_name -> numpy array)
 *   - Optionally, extra arguments such as:
 *       - "model_name" : string (to select a specific model)
 *
 * Internally, this function:
 *   1. Validates the input tensor names and models
 *   2. Optionally filters to a specific model if "model_name" is set
 *   3. Launches inference tasks (may be multi-threaded)
 *   4. Returns inference results as py::array (NumPy) in a nested map
 *
 * @threadsafe This function is guarded by an atomic flag and is not reentrant.
 *
 * @param[in] multi_input_tensors A nested input map:
 *        model_name -> (input_name -> py::array)
 * @param[in] extra_args Optional arguments for inference control. Supported keys:
 *        - "model_name" (std::string)
 *
 * @return A nested output map:
 *         model_name -> (output_name -> py::array)
 *
 * @throws std::runtime_error if:
 *         - Function is already running
 *         - Input/model name is invalid
 */
std::unordered_map<std::string, std::unordered_map<std::string, py::array>>
HB_HBMRuntime::run(std::unordered_map<std::string, std::unordered_map<std::string, py::array>>& multi_input_tensors,
                   const ExtraArgs& extra_args)
{
    // Ensure this function is not re-entered concurrently
    bool expected = false;
    if (!running_flag.compare_exchange_strong(expected, true)) {
        throw std::runtime_error("run() already in use");
    }

    // Automatically clear the running flag when function exits
    struct Guard {
        std::atomic<bool>& flag;
        ~Guard() { flag = false; }
    } guard{running_flag};

    // Step 1: Check that provided model/input names are valid
    CheckInputInfoValid(multi_input_tensors, model_names, input_names);

    // Step 2: Optionally filter multi_input_tensors by model_name
    ValidateOrFilterModelName(multi_input_tensors, extra_args);

    // Step 3: Perform inference and return results
    return InferAllModels(multi_input_tensors);
}

/**
 * @brief Get the version string of the underlying hbDNN library.
 *
 * This function returns the version of the hbDNN runtime as a standard string.
 *
 * @return A std::string representing the version, e.g., "3.5.0".
 */
const std::string HB_HBMRuntime::GetVersion() {
    return std::string(hbDNNGetVersion());
}

/**
 * @brief Load the model name list and model count from the packed DNN handle.
 *
 * This function queries the packed DNN handle to retrieve all sub-model names and the total model count.
 * The names are stored in `model_names` and the count in `model_count`. A flag `is_load_model_name_and_count`
 * is set to indicate that model metadata has been successfully loaded.
 *
 * @note This must be called before accessing individual model handles or metadata.
 */
void HB_HBMRuntime::LoadModelNameListAndCount() {
    const char** name_array = nullptr;

    // Query the DNN engine for model names and count
    HBDNN_CHECK_SUCCESS(hbDNNGetModelNameList(&name_array, &model_count, dnn_packed_handle),
                        "Failed to get model names.");

    // Copy each name into the model_names vector
    for (int i = 0; i < model_count; ++i) {
        model_names.emplace_back(name_array[i]);
    }

    // Set the internal flag to true
    is_load_model_name_and_count = true;
}

/**
 * @brief Load DNN model handles for each sub-model in the packed model.
 *
 * This function retrieves a runtime handle (`hbDNNHandle_t`) for each model contained
 * in the packed DNN model (`dnn_packed_handle`) and stores them in `dnn_handle_list`.
 * These handles are required to perform inference and query model metadata later.
 *
 * @note This function automatically calls LoadModelNameListAndCount() if model names are not yet loaded.
 */

void HB_HBMRuntime::LoadModelHandles()
{
    // If model names and count are not yet loaded, retrieve them first
    if (!is_load_model_name_and_count)
        LoadModelNameListAndCount();

    // Iterate through all loaded model names
    for (int i = 0; i < model_count; i++) {
        hbDNNHandle_t handle;

        // Get the DNN handle for the specific model from the packed handle
        // and store it in the internal handle map (dnn_handle_list)
        HBDNN_CHECK_SUCCESS(
            hbDNNGetModelHandle(&handle, dnn_packed_handle, model_names[i].c_str()),
            std::string("Failed to get handle for model: ") + model_names[i].c_str()
        );

        // Save the handle in a map for later inference usage
        dnn_handle_list[model_names[i]] = handle;
    }

    // Mark that model handles have been successfully loaded
    is_load_dnn_handle = true;
}

/**
 * @brief Load input metadata for all models, including tensor properties, shape, dtype, quantization, stride, and descriptions.
 *
 * This function iterates through all loaded model handles and collects detailed input information
 * such as names, tensor properties, quantization parameters, and descriptions. These are stored in
 * internal data structures for later use during inference and input validation.
 *
 * @note This must be called after loading model names and model handles.
 */
void HB_HBMRuntime::LoadInputInfo()
{
    // Ensure model names and handles are loaded
    if (!is_load_model_name_and_count)
        LoadModelNameListAndCount();
    if (!is_load_dnn_handle)
        LoadModelHandles();

    // Loop over all registered model handles
    for (const auto& pair : dnn_handle_list) {
        const std::string& model_name = pair.first;
        hbDNNHandle_t handle = pair.second;
        std::vector<std::string> inputs;

        int32_t count = 0;

        // Query the number of input tensors for this model
        HBDNN_CHECK_SUCCESS(hbDNNGetInputCount(&count, handle),
                            "Failed to get input count for model: " + model_name);

        // Iterate through each input tensor index
        for (int32_t i = 0; i < count; ++i) {
            const char* input_name_cstr = nullptr;

            // Get input name
            HBDNN_CHECK_SUCCESS(hbDNNGetInputName(&input_name_cstr, handle, i),
                                "Failed to get input name for model: " + model_name + ", index: " + std::to_string(i));

            // Get input tensor properties (shape, dtype, quantization, etc.)
            hbDNNTensorProperties props;
            HBDNN_CHECK_SUCCESS(hbDNNGetInputTensorProperties(&props, handle, i),
                                "Failed to get tensor properties for model: " + model_name + ", input: " + input_name_cstr);

            // Get optional input description string (if available)
            const char* input_desc_cstr = nullptr;
            uint32_t desc_size = 0;
            int32_t desc_type;
            HBDNN_CHECK_SUCCESS(hbDNNGetInputDesc(&input_desc_cstr, &desc_size, &desc_type, handle, i),
                                "Failed to get input desc for index " + std::to_string(i));

            inputs.emplace_back(input_name_cstr);  // Store input name list

            // Cache all relevant input tensor information
            input_tensor_properties[model_name][input_name_cstr] = props;

            input_shapes[model_name][input_name_cstr] = std::vector<int32_t>(
                props.validShape.dimensionSize,
                props.validShape.dimensionSize + props.validShape.numDimensions);

            input_dtypes[model_name][input_name_cstr] = static_cast<hbDNNDataType>(props.tensorType);

            input_quants[model_name][input_name_cstr].quant_type = props.quantiType;
            input_quants[model_name][input_name_cstr].axis = props.quantizeAxis;
            input_quants[model_name][input_name_cstr].scale = std::vector<float>(
                props.scale.scaleData,
                props.scale.scaleData + props.scale.scaleLen);
            input_quants[model_name][input_name_cstr].zero_point = std::vector<int32_t>(
                props.scale.zeroPointData,
                props.scale.zeroPointData + props.scale.zeroPointLen);

            input_strides[model_name][input_name_cstr].assign(
                props.stride,
                props.stride + props.validShape.numDimensions);

            // Store optional description if available
            if (desc_type == HB_DNN_DESC_TYPE_STRING && input_desc_cstr != nullptr && desc_size > 0) {
                input_descs[model_name][input_name_cstr] = std::string(input_desc_cstr, desc_size);
            } else {
                input_descs[model_name][input_name_cstr] = "";
            }
        }

        // Finalize input name list and count for this model
        input_counts[model_name] = count;
        input_names[model_name] = std::move(inputs);
    }
}

/**
 * @brief Load output metadata for all models, including tensor properties, shape, dtype, quantization, stride, and descriptions.
 *
 * This function queries all loaded models to extract information about each output tensor,
 * and stores these details in internal mappings for later access and validation.
 *
 * @note This function assumes that model names and handles have already been loaded.
 */
void HB_HBMRuntime::LoadOutputInfo()
{
    // Ensure model names and handles are available
    if (!is_load_model_name_and_count)
        LoadModelNameListAndCount();
    if (!is_load_dnn_handle)
        LoadModelHandles();

    // Iterate over all model handles
    for (const auto& pair : dnn_handle_list) {
        const std::string& model_name = pair.first;
        hbDNNHandle_t handle = pair.second;
        std::vector<std::string> outputs;

        int32_t count = 0;

        // Query number of output tensors for this model
        HBDNN_CHECK_SUCCESS(hbDNNGetOutputCount(&count, handle),
                            "Failed to get output count for model: " + model_name);

        // Loop over each output tensor
        for (int32_t i = 0; i < count; ++i) {
            const char* output_name_cstr = nullptr;

            // Retrieve output tensor name
            HBDNN_CHECK_SUCCESS(hbDNNGetOutputName(&output_name_cstr, handle, i),
                                "Failed to get output name for model: " + model_name + ", index: " + std::to_string(i));

            // Retrieve tensor properties (shape, dtype, quantization, etc.)
            hbDNNTensorProperties props;
            HBDNN_CHECK_SUCCESS(hbDNNGetOutputTensorProperties(&props, handle, i),
                                "Failed to get tensor properties for model: " + model_name + ", output: " + output_name_cstr);

            // Retrieve optional tensor description
            const char* output_desc_cstr = nullptr;
            uint32_t desc_size = 0;
            int32_t desc_type;
            HBDNN_CHECK_SUCCESS(hbDNNGetOutputDesc(&output_desc_cstr, &desc_size, &desc_type, handle, i),
                                "Failed to get output desc for index " + std::to_string(i));

            outputs.emplace_back(output_name_cstr);  // Store output name

            // Cache output tensor metadata
            output_tensor_properties[model_name][output_name_cstr] = props;

            output_shapes[model_name][output_name_cstr] = std::vector<int32_t>(
                props.validShape.dimensionSize,
                props.validShape.dimensionSize + props.validShape.numDimensions);

            output_dtypes[model_name][output_name_cstr] = static_cast<hbDNNDataType>(props.tensorType);

            output_quants[model_name][output_name_cstr].quant_type = props.quantiType;
            output_quants[model_name][output_name_cstr].axis = props.quantizeAxis;
            output_quants[model_name][output_name_cstr].scale = std::vector<float>(
                props.scale.scaleData,
                props.scale.scaleData + props.scale.scaleLen);
            output_quants[model_name][output_name_cstr].zero_point = std::vector<int32_t>(
                props.scale.zeroPointData,
                props.scale.zeroPointData + props.scale.zeroPointLen);

            output_strides[model_name][output_name_cstr].assign(
                props.stride,
                props.stride + props.validShape.numDimensions);

            // Save description if available
            if (desc_type == HB_DNN_DESC_TYPE_STRING && output_desc_cstr != nullptr && desc_size > 0) {
                output_descs[model_name][output_name_cstr] = std::string(output_desc_cstr, desc_size);
            } else {
                output_descs[model_name][output_name_cstr] = "";
            }
        }

        // Finalize output name list and count
        output_counts[model_name] = count;
        output_names[model_name] = std::move(outputs);
    }
}

/**
 * @brief Load and cache the description string for each loaded model.
 *
 * This function retrieves the model description metadata from the runtime
 * for all loaded models, storing the result as a string in an internal map.
 * If the description is not available or empty, an empty string is stored instead.
 *
 * @note This function assumes that model names and handles are already loaded.
 */
void HB_HBMRuntime::LoadModelDesc()
{
    // Ensure model names and handles are available
    if (!is_load_model_name_and_count)
        LoadModelNameListAndCount();
    if (!is_load_dnn_handle)
        LoadModelHandles();

    // Iterate over each model handle to get the description
    for (const auto& pair : dnn_handle_list) {
        const std::string& model_name = pair.first;
        hbDNNHandle_t handle = pair.second;

        const char* desc_cstr = nullptr;
        uint32_t desc_size = 0;
        int32_t desc_type = 0;

        // Retrieve model description from the handle
        HBDNN_CHECK_SUCCESS(hbDNNGetModelDesc(&desc_cstr, &desc_size, &desc_type, handle),
                            "Failed to get model desc for model: " + model_name);

        // Store description string if valid, else store empty string
        if (desc_type == HB_DNN_DESC_TYPE_STRING && desc_cstr != nullptr && desc_size > 0) {
            model_descs[model_name] = std::string(desc_cstr, desc_size);
        } else {
            model_descs[model_name] = "";
        }
    }
}

/**
 * @brief Load and cache the HBM (Hierarchical Bayesian Model) description for each model file.
 *
 * This function retrieves the HBM description string for each model file from the packed
 * handle and stores it in an internal map keyed by the model filename.
 * If the description is not available or empty, an empty string is stored instead.
 *
 * @param[in] model_files Vector of model filenames corresponding to loaded models.
 */
void HB_HBMRuntime::LoadHBMDesc(const std::vector<std::string>& model_files)
{
    for (size_t idx = 0; idx < model_files.size(); ++idx) {
        const std::string& filename = model_files[idx];

        const char* desc_cstr = nullptr;
        uint32_t desc_size = 0;
        int32_t desc_type = 0;

        // Retrieve HBM description from the packed model handle by index
        HBDNN_CHECK_SUCCESS(hbDNNGetHBMDesc(&desc_cstr, &desc_size, &desc_type, dnn_packed_handle, idx),
                            "Failed to get HBM desc for file: " + filename);

        // Store the description string if valid, using desc_size to avoid truncation caused by '\0' in the content
        if (desc_type == HB_DNN_DESC_TYPE_STRING && desc_cstr != nullptr && desc_size > 0) {
            HBM_descs[filename] = std::string(desc_cstr, desc_size);
        } else {
            HBM_descs[filename] = "";
        }
    }
}

/**
 * @brief Initialize a scheduling parameter structure with default values.
 *
 * @param[out] param Reference to the SchedParam structure to initialize.
 */
void HB_HBMRuntime::InitSchedParam(SchedParam& param)
{
    param.priority = 0;
    param.deviceId = 0;
    param.customId = 0;
    param.bpu_cores = std::vector<int32_t>{-1};  // Default core set to -1
}

/**
 * @brief Initialize scheduling parameters map for a list of models.
 *
 * For each model name in the provided vector, initializes a default
 * SchedParam and inserts it into the params_map with the model name as key.
 *
 * @param[out] params_map Reference to the map to populate with model scheduling parameters.
 * @param[in] model_names Vector of model names to initialize scheduling parameters for.
 */
void HB_HBMRuntime::InitSchedParamMapWithModels(std::unordered_map<std::string, SchedParam>& params_map,
                                const std::vector<std::string>& model_names) {
    for (const auto& name : model_names) {
        SchedParam param;
        InitSchedParam(param);
        params_map[name] = param;
    }
}


/**
 * @brief Load all necessary model parameters and metadata.
 *
 * This function sequentially loads:
 *  - The list of model names and their count,
 *  - The handles for each loaded model,
 *  - Input tensor information (names, shapes, data types, etc.),
 *  - Output tensor information (names, shapes, data types, etc.),
 *  - The descriptive string for each model,
 *  - The HBM (Hierarchical Bayesian Model) descriptions for the given model files.
 *
 * @param[in] model_files A vector of model file names corresponding to the loaded models.
 */
void HB_HBMRuntime::LoadParameters(const std::vector<std::string>& model_files)
{
    LoadModelNameListAndCount();
    LoadModelHandles();
    LoadInputInfo();
    LoadOutputInfo();
    LoadModelDesc();
    LoadHBMDesc(model_files);
}

/**
 * @brief Get the list of loaded model names.
 * @return Vector of model names.
 */
std::vector<std::string> HB_HBMRuntime::GetModelNames()
{
    return model_names;
}

/**
 * @brief Get the total number of loaded models.
 * @return Number of loaded models.
 */
int32_t HB_HBMRuntime::GetModelCount()
{
    return model_count;
}

/**
 * @brief Get the number of input tensors per model.
 * @return Map from model name to its input count.
 */
std::unordered_map<std::string, int32_t> HB_HBMRuntime::GetInputCounts()
{
    return input_counts;
}

/**
 * @brief Get the input tensor names per model.
 * @return Map from model name to a vector of input tensor names.
 */
std::unordered_map<std::string, std::vector<std::string>> HB_HBMRuntime::GetInputNames()
{
    return input_names;
}

/**
 * @brief Get the descriptive strings of input tensors per model.
 * @return Map from model name to map of input tensor name to its description string.
 */
std::unordered_map<std::string, std::unordered_map<std::string, std::string>> HB_HBMRuntime::GetInputDescs()
{
    return input_descs;
}

/**
 * @brief Get the input tensor shapes per model.
 * @return Map from model name to map of input tensor name to shape vector.
 */
std::unordered_map<std::string, std::unordered_map<std::string, std::vector<int32_t>>> HB_HBMRuntime::GetInputShapes()
{
    return input_shapes;
}

/**
 * @brief Get the input tensor data types per model.
 * @return Map from model name to map of input tensor name to data type enum.
 */
std::unordered_map<std::string, std::unordered_map<std::string, hbDNNDataType>> HB_HBMRuntime::GetInputDtpyes()
{
    return input_dtypes;
}

/**
 * @brief Get the quantization parameters for input tensors per model.
 * @return Map from model name to map of input tensor name to QuantParams struct.
 */
std::unordered_map<std::string, std::unordered_map<std::string, QuantParams>> HB_HBMRuntime::GetInputQuants()
{
    return input_quants;
}

/**
 * @brief Get the strides for input tensors per model.
 * @return Map from model name to map of input tensor name to strides vector.
 */
std::unordered_map<std::string, std::unordered_map<std::string, std::vector<int64_t>>> HB_HBMRuntime::GetInputStrides()
{
    return input_strides;
}

/**
 * @brief Get the number of output tensors per model.
 * @return Map from model name to its output count.
 */
std::unordered_map<std::string, int32_t> HB_HBMRuntime::GetOutputCounts()
{
    return output_counts;
}

/**
 * @brief Get the output tensor names per model.
 * @return Map from model name to vector of output tensor names.
 */
std::unordered_map<std::string, std::vector<std::string>> HB_HBMRuntime::GetOutputNames()
{
    return output_names;
}

/**
 * @brief Get descriptive strings of output tensors per model.
 * @return Map from model name to map of output tensor name to description string.
 */
std::unordered_map<std::string, std::unordered_map<std::string, std::string>> HB_HBMRuntime::GetOutputDescs()
{
    return output_descs;
}

/**
 * @brief Get the output tensor shapes per model.
 * @return Map from model name to map of output tensor name to shape vector.
 */
std::unordered_map<std::string, std::unordered_map<std::string, std::vector<int32_t>>> HB_HBMRuntime::GetOutputShapes()
{
    return output_shapes;
}

/**
 * @brief Get the output tensor data types per model.
 * @return Map from model name to map of output tensor name to data type enum.
 */
std::unordered_map<std::string, std::unordered_map<std::string, hbDNNDataType>> HB_HBMRuntime::GetOutputDtpyes()
{
    return output_dtypes;
}

/**
 * @brief Get quantization parameters for output tensors per model.
 * @return Map from model name to map of output tensor name to QuantParams struct.
 */
std::unordered_map<std::string, std::unordered_map<std::string, QuantParams>> HB_HBMRuntime::GetOutputQuants()
{
    return output_quants;
}

/**
 * @brief Get strides for output tensors per model.
 * @return Map from model name to map of output tensor name to strides vector.
 */
std::unordered_map<std::string, std::unordered_map<std::string, std::vector<int64_t>>> HB_HBMRuntime::GetOutputStrides()
{
    return output_strides;
}

/**
 * @brief Get descriptive strings for each loaded model.
 * @return Map from model name to model description string.
 */
std::unordered_map<std::string, std::string> HB_HBMRuntime::GetModelDescs()
{
    return model_descs;
}

/**
 * @brief Get HBM descriptive strings for each loaded model file.
 * @return Map from model filename to HBM description string.
 */
std::unordered_map<std::string, std::string> HB_HBMRuntime::GetHBMDescs()
{
    return HBM_descs;
}

/**
 * @brief Get the scheduling parameters for all models.
 *
 * @return std::unordered_map<std::string, SchedParam>
 *         A map from model name (string) to its scheduling parameters (SchedParam).
 */
std::unordered_map<std::string, SchedParam> HB_HBMRuntime::GetModelSchedParams()
{
    return model_sched_params;
}

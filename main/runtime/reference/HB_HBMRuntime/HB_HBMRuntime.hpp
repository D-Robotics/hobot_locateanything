// Copyright (c) 2025 D-Robotics Co,.Ltd. All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of D-Robotics Co,.Ltd. This is proprietary information owned by
// D-Robotics Co,.Ltd. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of D-Robotics Co,.Ltd.

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <optional>
#include <variant>
#include <thread>
#include <unordered_map>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <algorithm>
#include <atomic>

#include "HB_RuntimeUtils.hpp"
#include "hb_ucp.h"
#include "hb_dnn.h"
#include "hb_ucp_status.h"
#include "hb_dnn_status.h"

/**
 * @brief Macro to check the return code of a DNN function and throw an exception if it fails.
 *
 * This macro wraps a DNN API call, checks its return status, and throws a std::runtime_error
 * with a descriptive error message if the call does not return 0 (success).
 *
 * It also attaches a user-provided context string to help identify where the error occurred.
 *
 * @param func_call     The DNN function call to be evaluated. Must return an int32_t status code.
 * @param context_str   A string (typically function name or description) providing context for the error message.
 *
 * @throws std::runtime_error If the return code is non-zero, including the error code and description.
 *
 * @note This macro uses a `do { ... } while (0)` block to ensure safe usage with semicolons and in conditionals.
 */
#define HBDNN_CHECK_SUCCESS(func_call, context_str)                                                    \
    do {                                                                                               \
        int32_t __err_code = (func_call);                                                              \
        if (__err_code != 0) {                                                                         \
            const char* __err_desc = hbDNNGetErrorDesc(__err_code);                                    \
            throw std::runtime_error(std::string("DNN Error (code: ") + std::to_string(__err_code) +   \
                                     ", desc: " + __err_desc + ") " + (context_str));                  \
        }                                                                                              \
    } while (0)

/**
 * @brief Macro to check the return code of a UCP function and throw an exception on failure.
 *
 * This macro wraps a UCP API call, checks whether it succeeded (i.e., returned 0),
 * and throws a std::runtime_error if it failed. The error message includes the
 * error code, description (retrieved from hbUCPGetErrorDesc), and a user-defined context string.
 *
 * @param func_call     The UCP function call to be checked. Must return an int32_t status code.
 * @param context_str   A string providing context for the error, such as the function name or purpose.
 *
 * @throws std::runtime_error If the return code is non-zero.
 *
 * @note The macro uses a `do { ... } while (0)` block for safe expansion in all contexts.
 *
 * @see hbUCPGetErrorDesc
 */
#define HBUCP_CHECK_SUCCESS(func_call, context_str)                                                    \
    do {                                                                                               \
        int32_t __err_code = (func_call);                                                              \
        if (__err_code != 0) {                                                                         \
            const char* __err_desc = hbUCPGetErrorDesc(__err_code);                                    \
            throw std::runtime_error(std::string("UCP Error (code: ") + std::to_string(__err_code) +   \
                                     ", desc: " + __err_desc + ") " + (context_str));                  \
        }                                                                                              \
    } while (0)

/**
 * @brief Aligns a given unsigned integer value to the specified boundary.
 *
 * This function rounds up the value @p w to the nearest multiple of @p alignment.
 * Commonly used for memory alignment, especially in hardware or low-level optimization scenarios.
 *
 * @param w          The input value to align.
 * @param alignment  The alignment boundary, must be a power of 2.
 * @return The smallest multiple of @p alignment that is greater than or equal to @p w.
 *
 * @note This function assumes that @p alignment is a power of 2.
 */
inline uint32_t ALIGNED_2E(uint32_t const w, uint32_t const alignment) {
  return ((w + (alignment - 1U)) & (~(alignment - 1U)));
}

/**
 * @brief Aligns a value to the next multiple of 64.
 *
 * This macro uses ALIGNED_2E internally to round a value up to a 64-byte boundary.
 *
 * @param w  The value to align (can be an integer expression).
 * @return The aligned value.
 */
#define ALIGN_64(w) ALIGNED_2E(static_cast<uint32_t>(w), 64U)

/**
 * @brief Checks if a pointer is null and returns a specified error code if it is.
 *
 * This macro is used for early-exit validation of pointers. If the pointer @p ptr is null,
 * it logs an error to std::cerr and returns @p code.
 *
 * @param ptr   The pointer to check.
 * @param code  The return value to use if @p ptr is null.
 *
 * @note The macro expands into a block using `{}`; it should be used as a standalone statement.
 */
#define LOGE_AND_RETURN_IF_NULL(ptr, code) \
{                                        \
    if ((ptr) == nullptr) {                \
        std::cerr << #ptr << " is null pointer" << std::endl; \
        return (code);                       \
    }                                      \
}

namespace py = pybind11;

typedef struct {
    std::vector<float> scale;
    std::vector<int32_t> zero_point;
    hbDNNQuantiType quant_type;
    int32_t axis;
} QuantParams;

typedef struct SchedParam {
    int32_t priority;
    int64_t customId;
    std::vector<int32_t> bpu_cores;
    uint32_t deviceId;
} SchedParam;

using ExtraArgValue = std::variant<int, float, double,
        bool, std::string, std::unordered_map<std::string, int32_t>, std::unordered_map<std::string, std::vector<int32_t>>>;
using ExtraArgs = std::unordered_map<std::string, ExtraArgValue>;

class HB_HBMRuntime {
private:
    /// Indicates whether the inference runtime is actively running.
    std::atomic<bool> running_flag{false};

    /// Packed handle for managing multiple DNN models.
    hbDNNPackedHandle_t dnn_packed_handle;

    /// Mapping from model name to its corresponding DNN handle.
    std::unordered_map<std::string, hbDNNHandle_t> dnn_handle_list;

    /// List of loaded model names.
    std::vector<std::string> model_names;

    /// Number of models loaded.
    int32_t model_count;

    /// Mapping from model name to number of input tensors.
    std::unordered_map<std::string, int32_t> input_counts;

    /// Mapping from model name to list of input tensor names.
    std::unordered_map<std::string, std::vector<std::string>> input_names;

    /// Mapping from model -> tensor name -> input tensor description (string).
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> input_descs;

    /// Mapping from model -> tensor name -> input tensor properties.
    std::unordered_map<std::string, std::unordered_map<std::string, hbDNNTensorProperties>> input_tensor_properties;

    /// Mapping from model -> tensor name -> input tensor shape.
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<int32_t>>> input_shapes;

    /// Mapping from model -> tensor name -> input tensor data type.
    std::unordered_map<std::string, std::unordered_map<std::string, hbDNNDataType>> input_dtypes;

    /// Mapping from model -> tensor name -> quantization parameters.
    std::unordered_map<std::string, std::unordered_map<std::string, QuantParams>> input_quants;

    /// Mapping from model -> tensor name -> input tensor strides.
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<int64_t>>> input_strides;

    /// Mapping from model name to number of output tensors.
    std::unordered_map<std::string, int32_t> output_counts;

    /// Mapping from model name to list of output tensor names.
    std::unordered_map<std::string, std::vector<std::string>> output_names;

    /// Mapping from model -> tensor name -> output tensor description (string).
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> output_descs;

    /// Mapping from model -> tensor name -> output tensor properties.
    std::unordered_map<std::string, std::unordered_map<std::string, hbDNNTensorProperties>> output_tensor_properties;

    /// Mapping from model -> tensor name -> output tensor shape
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<int32_t>>> output_shapes;

    /// Mapping from model -> tensor name -> output tensor data type.
    std::unordered_map<std::string, std::unordered_map<std::string, hbDNNDataType>> output_dtypes;

    /// Mapping from model -> tensor name -> quantization parameters.
    std::unordered_map<std::string, std::unordered_map<std::string, QuantParams>> output_quants;

    /// Mapping from model -> tensor name -> output tensor strides.
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<int64_t>>> output_strides;

    /// Mapping from model name to model description string.
    std::unordered_map<std::string, std::string> model_descs;

    /// Mapping from model name to original HBM file path or descriptor.
    std::unordered_map<std::string, std::string> HBM_descs;

    /// Map storing scheduling parameters for each model by model name
    std::unordered_map<std::string, SchedParam> model_sched_params;

    /// Flag indicating whether model name list and count have been loaded.
    bool is_load_model_name_and_count = false;

    /// Flag indicating whether model handles have been loaded.
    bool is_load_dnn_handle = false;

    /// Flag indicating whether model name list has been loaded.
    bool is_load_mode_name_list = false;

    /// Flag indicating whether input shapes have been loaded
    bool input_shape_ready = false;

    /**
    * @brief Load the model name list and model count from the packed DNN handle.
    *
    * This function queries the packed DNN handle to retrieve all sub-model names and the total model count.
    * The names are stored in `model_names` and the count in `model_count`. A flag `is_load_model_name_and_count`
    * is set to indicate that model metadata has been successfully loaded.
    *
    * @note This must be called before accessing individual model handles or metadata.
    */
    void LoadModelNameListAndCount();

    /**
    * @brief Load DNN model handles for each sub-model in the packed model.
    *
    * This function retrieves a runtime handle (`hbDNNHandle_t`) for each model contained
    * in the packed DNN model (`dnn_packed_handle`) and stores them in `dnn_handle_list`.
    * These handles are required to perform inference and query model metadata later.
    *
    * @note This function automatically calls LoadModelNameListAndCount() if model names are not yet loaded.
    */
    void LoadModelHandles();

    /**
    * @brief Load and cache the description string for each loaded model.
    *
    * This function retrieves the model description metadata from the runtime
    * for all loaded models, storing the result as a string in an internal map.
    * If the description is not available or empty, an empty string is stored instead.
    *
    * @note This function assumes that model names and handles are already loaded.
    */
    void LoadModelDesc();

    /**
    * @brief Load and cache the HBM (Hierarchical Bayesian Model) description for each model file.
    *
    * This function retrieves the HBM description string for each model file from the packed
    * handle and stores it in an internal map keyed by the model filename.
    * If the description is not available or empty, an empty string is stored instead.
    *
    * @param[in] model_files Vector of model filenames corresponding to loaded models.
    */
    void LoadHBMDesc(const std::vector<std::string>& model_files);

    /**
    * @brief Load input metadata for all models, including tensor properties, shape, dtype, quantization, stride, and descriptions.
    *
    * This function iterates through all loaded model handles and collects detailed input information
    * such as names, tensor properties, quantization parameters, and descriptions. These are stored in
    * internal data structures for later use during inference and input validation.
    *
    * @note This must be called after loading model names and model handles.
    */
    void LoadInputInfo();

    /**
    * @brief Load output metadata for all models, including tensor properties, shape, dtype, quantization, stride, and descriptions.
    *
    * This function queries all loaded models to extract information about each output tensor,
    * and stores these details in internal mappings for later access and validation.
    *
    * @note This function assumes that model names and handles have already been loaded.
    */
    void LoadOutputInfo();

    /**
    * @brief Initialize a scheduling parameter structure with default values.
    *
    * @param[out] param Reference to the SchedParam structure to initialize.
    */
    void InitSchedParam(SchedParam& param);

    /**
    * @brief Initialize scheduling parameters map for a list of models.
    *
    * For each model name in the provided vector, initializes a default
    * SchedParam and inserts it into the params_map with the model name as key.
    *
    * @param[out] params_map Reference to the map to populate with model scheduling parameters.
    * @param[in] model_names Vector of model names to initialize scheduling parameters for.
    */
    void InitSchedParamMapWithModels(std::unordered_map<std::string, SchedParam>& params_map,
                                    const std::vector<std::string>& model_names);

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
    void LoadParameters(const std::vector<std::string>& model_files);

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
    std::string HbDNNTypeToNumpyFormat(int hb_type);

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
    void CheckInputDtypeCompatibility(const std::string& model_name, const std::string& tensor_name,
                                                 int expected_hb_type, const py::array& input_array);

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
    int32_t PrepareInputValidShape(std::string &model_name, std::string &input_tensor_name,
                                        hbDNNTensorShape &shape, py::array & input_array);

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
    int32_t PrepareInputStride(std::string &model_name, std::string &input_tensor_name,
                                        hbDNNTensorShape &shape,   int64_t *stride, py::array & input_array);

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
    int32_t PrepareInputTensorData(std::string &model_name, std::string &input_tensor_name,
                                                   hbDNNTensor &tensor, py::array & input_array);

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
    int32_t PrepareInputTensor(std::string model_name, std::vector<hbDNNTensor> &model_input_tensor,
                                std::unordered_map<std::string, py::array>& model_input_info);

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
    int32_t PrepareOutputTensor(std::string model_name, std::vector<hbDNNTensor> & model_output_tensor);

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
    int32_t PrepareOutputArrays(std::string model_name, std::unordered_map<std::string, py::array> & model_output_array,
                            std::vector<hbDNNTensor> & model_output_tensor);


    uint64_t GetBPUCoreMaskForModel(const std::string& model_name, const std::vector<int32_t>& bpu_cores);

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
    void InferSingleModel(const std::string& model_name,
                 hbDNNHandle_t& dnn_handle,
                 std::vector<hbDNNTensor>& input_tensors,
                 std::vector<hbDNNTensor>& output_tensors);

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
    int32_t LaunchInferenceTasks(std::unordered_map<std::string, std::vector<hbDNNTensor>> & input_tensor,
                        std::unordered_map<std::string, std::vector<hbDNNTensor>> & output_tensor);

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
    InferAllModels(std::unordered_map<std::string, std::unordered_map<std::string, py::array>>& multi_input_tensors);

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
    std::string ParseAndValidateModelName(const ExtraArgs& args, const std::vector<std::string>& model_list);

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
    void CheckInputInfoValid(const std::unordered_map<std::string, std::unordered_map<std::string, py::array>>& multi_input_tensors,
                            const std::vector<std::string>& model_list,
                            const std::unordered_map<std::string, std::vector<std::string>>& input_names);

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
    void ValidateOrFilterModelName(std::unordered_map<std::string, std::unordered_map<std::string, py::array>>& multi_input_tensors,
                                const ExtraArgs& extra_args);

    /**
    * @brief Parse and validate BPU core assignment from extra_args.
    *
    * This function reads the optional "bpu_cores" field from the user-provided extra_args
    * and constructs a mapping from model names to BPU core ID lists.
    * Each core ID must be in the range [0, 3]. If "bpu_cores" is not specified,
    * a default value of {-1} is assigned to each model, indicating no binding preference.
    *
    * @param[in] multi_input_tensors A nested map of model_name -> (input_name -> py::array),
    *                         representing all models and inputs involved in the current inference.
    * @param[in] extra_args       Optional configuration dictionary that may contain the "bpu_cores" field:
    *                         - key: model_name (string)
    *                         - value: list of target BPU core IDs (vector<int32_t>)
    *
    * @return A map from model_name to list of assigned BPU core IDs.
    *         If not specified, each model will have core list set to {-1}.
    *
    * @throws std::runtime_error if:
    *         - "bpu_cores" exists but is not of type std::unordered_map<std::string, vector<int32_t>>
    *         - Any BPU core ID is out of valid range [0, 3]
    *         - "bpu_cores" contains a model name not present in multi_input_tensors
    *
    * @note The returned core ID list may contain multiple elements.
    */
    std::unordered_map<std::string, std::vector<int32_t>>
    ParseBpuCoreIds(const std::unordered_map<std::string, std::unordered_map<std::string, py::array>>& multi_input_tensors,
                    const ExtraArgs& extra_args);

public:

    /**
     * @brief Construct runtime from a single model file.
     * @param[in] model_file Path to the model file (e.g., .bin or .hbm).
     */
    HB_HBMRuntime(const std::string& model_file);

    /**
     * @brief Construct runtime from multiple model files.
     * @param[in] model_files List of model paths to be loaded into the runtime.
     */
    HB_HBMRuntime(const std::vector<std::string>& model_files);


    /**
     * @brief Destructor. Automatically releases resources and allocated memory.
     */
    ~HB_HBMRuntime();

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
    void FreeTensorMem(std::vector<hbDNNTensor>& tensors);

    /**
    * @brief Set the scheduling priorities for models.
    *
    * Validates model names and priority values before updating the internal scheduling parameters.
    *
    * @param[in] priority  Map from model name to priority value (0-255).
    *
    * @throws std::runtime_error if a model name is invalid or priority is out of range.
    */
    void SetModelPriorities(const std::unordered_map<std::string, int>& priority);

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
    void SetModelBpuCores(const std::unordered_map<std::string, std::vector<int32_t>>& bpu_cores);

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
    void SetCustomIds(const std::unordered_map<std::string, int64_t>& custom_id);

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
    void SetDeviceIds(const std::unordered_map<std::string, uint32_t>& device_id);

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
    void SetSchedulingParams(
        const std::optional<std::unordered_map<std::string, int32_t>>& priority = std::nullopt,
        const std::optional<std::unordered_map<std::string, std::vector<int32_t>>>& bpu_cores = std::nullopt,
        const std::optional<std::unordered_map<std::string, int64_t>>& custom_id = std::nullopt,
        const std::optional<std::unordered_map<std::string, uint32_t>>& device_id = std::nullopt);

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
    run(py::array input_tensor, const ExtraArgs& extra_args);

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
    run(std::unordered_map<std::string, py::array>& input_tensors,
                    const ExtraArgs& extra_args);

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
    run(std::unordered_map<std::string, std::unordered_map<std::string, py::array>>& multi_input_tensors,
                    const ExtraArgs& extra_args);

    /**
    * @brief Get the version string of the underlying hbDNN library.
    *
    * This function returns the version of the hbDNN runtime as a standard string.
    *
    * @return A std::string representing the version, e.g., "3.5.0".
    */
    static const std::string GetVersion();

    /**
    * @brief Get the list of loaded model names.
    * @return Vector of model names.
    */
    std::vector<std::string> GetModelNames();

    /**
    * @brief Get the total number of loaded models.
    * @return Number of loaded models.
    */
    int32_t GetModelCount();

    /**
    * @brief Get the number of input tensors per model.
    * @return Map from model name to its input count.
    */
    std::unordered_map<std::string, int32_t> GetInputCounts();

    /**
    * @brief Get the input tensor names per model.
    * @return Map from model name to a vector of input tensor names.
    */
    std::unordered_map<std::string, std::vector<std::string>> GetInputNames();

    /**
    * @brief Get the descriptive strings of input tensors per model.
    * @return Map from model name to map of input tensor name to its description string.
    */
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> GetInputDescs();

    /**
    * @brief Get the input tensor shapes per model.
    * @return Map from model name to map of input tensor name to shape vector.
    */
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<int32_t>>> GetInputShapes();

    /**
    * @brief Get the input tensor data types per model.
    * @return Map from model name to map of input tensor name to data type enum.
    */
    std::unordered_map<std::string, std::unordered_map<std::string, hbDNNDataType>> GetInputDtpyes();

    /**
    * @brief Get the quantization parameters for input tensors per model.
    * @return Map from model name to map of input tensor name to QuantParams struct.
    */
    std::unordered_map<std::string, std::unordered_map<std::string, QuantParams>> GetInputQuants();

    /**
    * @brief Get the strides for input tensors per model.
    * @return Map from model name to map of input tensor name to strides vector.
    */
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<int64_t>>> GetInputStrides();

    /**
    * @brief Get the number of output tensors per model.
    * @return Map from model name to its output count.
    */
    std::unordered_map<std::string, int32_t> GetOutputCounts();

    /**
    * @brief Get the output tensor names per model.
    * @return Map from model name to vector of output tensor names.
    */
    std::unordered_map<std::string, std::vector<std::string>> GetOutputNames();

    /**
    * @brief Get descriptive strings of output tensors per model.
    * @return Map from model name to map of output tensor name to description string.
    */
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> GetOutputDescs();

    /**
    * @brief Get the output tensor shapes per model.
    * @return Map from model name to map of output tensor name to shape vector.
    */
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<int32_t>>> GetOutputShapes();

    /**
    * @brief Get the output tensor data types per model.
    * @return Map from model name to map of output tensor name to data type enum.
    */
    std::unordered_map<std::string, std::unordered_map<std::string, hbDNNDataType>> GetOutputDtpyes();

    /**
    * @brief Get quantization parameters for output tensors per model.
    * @return Map from model name to map of output tensor name to QuantParams struct.
    */
    std::unordered_map<std::string, std::unordered_map<std::string, QuantParams>> GetOutputQuants();

    /**
    * @brief Get strides for output tensors per model.
    * @return Map from model name to map of output tensor name to strides vector.
    */
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<int64_t>>> GetOutputStrides();

    /**
    * @brief Get descriptive strings for each loaded model.
    * @return Map from model name to model description string.
    */
    std::unordered_map<std::string, std::string> GetModelDescs();

    /**
    * @brief Get HBM descriptive strings for each loaded model file.
    * @return Map from model filename to HBM description string.
    */
    std::unordered_map<std::string, std::string> GetHBMDescs();

    /**
    * @brief Get the scheduling parameters for all models.
    *
    * @return std::unordered_map<std::string, SchedParam>
    *         A map from model name (string) to its scheduling parameters (SchedParam).
    */
    std::unordered_map<std::string, SchedParam> GetModelSchedParams();
};

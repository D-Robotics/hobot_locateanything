// Copyright (c) 2025 D-Robotics Co,.Ltd. All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of D-Robotics Co,.Ltd. This is proprietary information owned by
// D-Robotics Co,.Ltd. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of D-Robotics Co,.Ltd.
#pragma once

#include <filesystem>
#include <stdexcept>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "HB_HBMRuntime.hpp"

namespace py = pybind11;

/**
 * @brief Calculate the total number of elements in a given NumPy array.
 *
 * This function multiplies all dimensions of the input NumPy array
 * to determine the total number of scalar elements it contains.
 *
 * @param[in] arr A py::array object representing the input NumPy array.
 * @return int64_t The total number of elements in the array.
 *
 * @note This function assumes the input array has a valid shape.
 *       For example, an array with shape (3, 224, 224) returns 150528.
 */
int64_t get_array_total_elements(const py::array& arr) ;

/**
 * @brief Get the size in bytes of a single element of a given tensor data type.
 *
 * This function maps the enum value representing the tensor data type
 * to its corresponding element size in bytes.
 *
 * @param[in] type Integer representing the hbDNN tensor data type.
 *             Supported types include:
 *             - HB_DNN_TENSOR_TYPE_BOOL8, HB_DNN_TENSOR_TYPE_S8, HB_DNN_TENSOR_TYPE_U8 (1 byte)
 *             - HB_DNN_TENSOR_TYPE_F16, HB_DNN_TENSOR_TYPE_S16, HB_DNN_TENSOR_TYPE_U16 (2 bytes)
 *             - HB_DNN_TENSOR_TYPE_F32, HB_DNN_TENSOR_TYPE_S32, HB_DNN_TENSOR_TYPE_U32 (4 bytes)
 *             - HB_DNN_TENSOR_TYPE_F64, HB_DNN_TENSOR_TYPE_S64, HB_DNN_TENSOR_TYPE_U64 (8 bytes)
 *
 * @return int32_t Size in bytes of one element of the specified type.
 *
 * @throws std::runtime_error If the input type is not supported.
 */
int32_t get_element_size(int32_t type);

/**
 * @brief Calculate the product of elements in a dimension array.
 *
 * This function computes the total number of elements represented by
 * the shape dimensions of a tensor or array.
 *
 * @tparam T The integer type of the dimension elements (e.g., int, size_t).
 *
 * @param[in] dim Pointer to an array containing the size of each dimension.
 * @param[in] dim_num Number of dimensions (length of the dim array).
 *
 * @return int64_t The total number of elements (product of all dimension sizes).
 *
 * @note The result may overflow if the product exceeds int64_t range.
 */
template <typename T>
int64_t get_prod_size(T const *dim, uint32_t dim_num);

/**
 * @brief Recursively copy input tensor data to output buffer with padding based on strides.
 *
 * This function copies data from the input buffer to the output buffer,
 * handling multi-dimensional tensors where output strides may include padding.
 * It uses recursion to handle each dimension and copies data block-wise.
 *
 * @param[out] output_ptr Pointer to the output buffer where padded data will be written.
 * @param[in] input_ptr Pointer to the input data buffer.
 * @param[in] dim_num Number of dimensions of the tensor.
 * @param[in] dim Pointer to an array representing the size of each dimension.
 * @param[in] stride Pointer to an array representing the stride (in bytes) for each dimension in the output buffer.
 * @param[in] element_size Size in bytes of each individual element in the tensor.
 *
 * @note If the tensor is 1-dimensional, a simple memcpy is used.
 *       For higher dimensions, the function recurses over each sub-dimension.
 */
void add_padding_core(void *output_ptr, void const *input_ptr, uint32_t dim_num,
                      uint32_t const *dim, int64_t const *stride,
                      uint32_t element_size);

/**
 * @brief Copy input tensor data into output buffer with padding according to strides.
 *
 * This function serves as a wrapper that performs basic null pointer checks
 * before calling the recursive core function `add_padding_core` to copy
 * and pad the tensor data.
 *
 * @param[out] output Pointer to the output buffer where the padded tensor data will be stored.
 * @param[in] input Pointer to the input tensor data buffer.
 * @param[in] dim_num Number of dimensions of the tensor.
 * @param[in] dim Pointer to an array holding the size of each dimension.
 * @param[in] stride Pointer to an array holding the stride (in bytes) for each dimension in the output buffer.
 * @param[in] element_size Size (in bytes) of each tensor element.
 *
 * @return int32_t Returns 0 on success, -1 if any input pointer is null.
 */
int32_t add_padding(void *output, void const *input, uint32_t dim_num,
                    const uint32_t *dim, const int64_t *stride,
                    uint32_t element_size);

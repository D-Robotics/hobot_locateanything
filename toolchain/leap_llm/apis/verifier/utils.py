from typing import Any, OrderedDict, Union

import numpy as np

from .types import Colors, TensorInfo


def calculate_cosine(
    vec1: Union[np.ndarray[Any, Any], None], vec2: Union[np.ndarray[Any, Any], None]
) -> Union[float, None]:
    """Calculate the cosine similarity between two numpy arrays.

    Args:
        vec1 (np.ndarray): the first numpy array
        vec2 (np.ndarray): the second numpy array

    Returns:
        Union[float, None]: the cosine similarity between two vectors

    """
    if vec1 is None or vec2 is None:
        return None

    if vec1.flatten().shape != vec2.flatten().shape:
        return None

    if np.all(vec1 == 0) and np.all(vec2 == 0):
        return 1.0

    v1 = vec1.flatten().astype(np.float64, copy=False)
    v2 = vec2.flatten().astype(np.float64, copy=False)

    dot_product = np.dot(v1, v2)
    norm_vec1 = np.linalg.norm(v1)
    norm_vec2 = np.linalg.norm(v2)
    if norm_vec1 == 0 and norm_vec2 == 0:
        return 1.0

    if norm_vec1 == 0 or norm_vec2 == 0:
        return None

    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity

    # MSE
    # mse = np.mean((v1 - v2) ** 2)
    # return mse


def print_colored(text: str, color: str = Colors.WHITE) -> None:
    """Print colored text"""
    print(f"{color}{text}{Colors.END}")


def to_json_serializable(obj):
    """Convert an object to a JSON serializable object."""
    if isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return list(obj)
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    else:
        return str(obj)


def cast_to_tensor_info(
    input: Union[dict, list, tuple, Any],
) -> "OrderedDict[str, TensorInfo]":
    """Cast raw outputs (dict, list, tuple, or single value)
       to OrderedDict of TensorInfo.

    Args:
        input: The input data to convert. Can be:
            - dict: Each key-value pair becomes a TensorInfo
            - list/tuple: Each item becomes a TensorInfo with name "output_{idx}"
            - single value: Becomes a TensorInfo with name "output_0"

    Returns:
        OrderedDict[str, TensorInfo]: Converted tensor information
    """
    from collections import OrderedDict

    tensor_dict: "OrderedDict[str, TensorInfo]" = OrderedDict()

    if isinstance(input, dict):
        for key, value in input.items():
            if isinstance(value, TensorInfo):
                tensor_dict[str(key)] = value
            else:
                tensor_dict[str(key)] = TensorInfo(value, name=str(key))
    elif isinstance(input, (list, tuple)):
        if len(input) > 0:
            tensor_dict["output_0"] = TensorInfo(input[0], name="output_0")
    else:
        tensor_dict["output_0"] = TensorInfo(input, name="output_0")

    return tensor_dict

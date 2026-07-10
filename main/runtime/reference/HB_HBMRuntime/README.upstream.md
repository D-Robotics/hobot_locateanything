# hbm_runtime

hbm_runtime 是一个统一的推理执行接口，支持在 BPU 上加载和执行量化后的模型。本项目提供 Python 接口（通过 pybind11 绑定），便于在多种开发环境中使用。

## 环境要求
本项目基于 Python 编写并依赖多个第三方库。请确保您的环境满足以下要求：

### Python 环境
  Python 版本：建议使用 Python 3.10.x（当前已在 3.10.12 下测试通过）

### python依赖库

| 组件                   | 版本要求       |
| --------------------- | -------------- |
| **CMake**             | ≥ 3.22.1       |
| **pybind11**          | ≥ 3.0.0        |
| **scikit-build-core** | ≥ 0.11.5       |
| **build**             | ≥ 1.2.2.post1  |
| **wheel**             | ≥ 0.37.1       |

#### 安装依赖

```bash
# 系统依赖
sudo apt update

# Python 构建依赖
pip install build scikit-build-core pybind11 wheel
```

### 动态库依赖
| 库名          | 来源路径           | 说明                     |
| ------------- | ----------------- | ------------------------ |
| `libhbucp.so` | `/usr/hobot/lib/` | UCP 通信库，芯片间通信接口 |
| `libdnn.so`   | `/usr/hobot/lib/` | DNN 推理引擎运行库        |


## 项目结构说明

```bash
.
├── CMakeLists.txt                # CMake 构建配置文件
├── README.md                     # 项目说明文档（当前文件）
├── build.sh                      # 一键构建脚本
├── hbm_runtime                   # Python 接口目录
│   ├── HB_HBMRuntime.pyi         # Python 类型提示文件
│   ├── __init__.py               # Python 包入口
│   └── py.typed                  # PEP 561 标记，声明类型完整性
├── include                       # C++ 头文件
│   ├── HB_HBMRuntime.hpp
│   └── HB_RuntimeUtils.hpp
├── pyproject.toml                # Python 打包配置文件
└── src                           # C++ 源码目录
    ├── HBMRuntimeBinding.cc      # pybind11 绑定代码
    ├── HB_HBMRuntime.cc
    └── HB_RuntimeUtils.cc
```

## 使用方法
  本项目提供了统一的构建脚本 build.sh，用于快速完成构建、安装和清理操作。

### 构建 wheel 包
```bash
./build.sh build
```
等价于执行 python -m build --wheel
输出文件将生成在 dist/ 目录下，文件名形如：HB_HBMRuntime-<version>-py3-none-any.whl

### 安装 Python 包
```bash
./build.sh install
```
使用 pip install 安装当前目录下的包（包括 C++ 扩展模块），无需重新构建。

### 清理构建缓存
```bash
./build.sh clean
```
删除 build/、dist/、*.egg-info/、_skbuild/ 等中间构建产物。

### 默认行为
若不传入参数，则默认为 build：
```bash
./build.sh
```
等价于：

```
./build.sh build
```

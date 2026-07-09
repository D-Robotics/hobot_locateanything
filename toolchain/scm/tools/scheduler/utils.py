"""
工具函数模块
包含各种实用工具函数
"""

import logging

import sys
from typing import Dict, Any, Optional
import yaml

import os


logger = logging.getLogger(__name__)


def load_yaml_config(config_path: str) -> Optional[Dict[Any, Any]]:
    """
    加载YAML配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        Dict: 配置字典或None
    """
    try:
        if not os.path.exists(config_path):
            logger.error(f"配置文件不存在: {config_path}")
            return None

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            logger.debug(f"成功加载配置文件: {config_path}")
            return config

    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return None


def validate_required_fields(config: Dict, required_fields: list) -> bool:
    """
    验证配置中的必需字段

    Args:
        config: 配置字典
        required_fields: 必需字段列表

    Returns:
        bool: 验证是否通过
    """
    for field in required_fields:
        if field not in config:
            logger.error(f"缺少必需字段: {field}")
            return False
    return True


def get_env_var(var_name: str, default: str = None) -> str:
    """
    获取环境变量值

    Args:
        var_name: 环境变量名称
        default: 默认值

    Returns:
        str: 环境变量值
    """
    return os.environ.get(var_name, default)


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """
    设置日志配置

    Args:
        log_level: 日志级别
        log_file: 日志文件路径
    """
    # 从配置中获取日志设置，如果没有则使用传入参数
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"无效的日志级别: {log_level}")

    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file))
    elif log_file is None:  # 如果没有显式指定日志文件，则从配置中获取
        # 这里可以添加从配置文件获取日志文件路径的逻辑
        pass

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    logger.debug("日志配置已设置")


def format_duration(seconds: float) -> str:
    """
    格式化持续时间

    Args:
        seconds: 秒数

    Returns:
        str: 格式化的持续时间字符串
    """
    if seconds < 60:
        return f"{seconds:.2f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}分钟"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}小时"


def merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """
    合并两个字典

    Args:
        dict1: 第一个字典
        dict2: 第二个字典

    Returns:
        Dict: 合并后的字典
    """
    merged = dict1.copy()
    merged.update(dict2)
    return merged


def generate_version(base_version: str,
                    is_dailybuild: bool = False,
                    is_test: bool = False,
                    build_number: int = None) -> str:
    """
    根据模式生成版本号

    Args:
        base_version: 基础版本号（如 "1.0.0"）
        is_dailybuild: 是否是dailybuild模式
        is_test: 是否是测试模式
        build_number: 构建号（测试模式需要）

    Returns:
        str: 生成的版本号

    版本号格式:
        - Release版本: {base_version}
        - Dailybuild版本: {base_version}.daily.{YYYYMMDD}
        - Test版本: {base_version}.post.{build_number}.dev.{YYYYMMDD}
    """
    from datetime import datetime

    today = datetime.now().strftime("%Y%m%d")

    if is_test and build_number is not None:
        return f"{base_version}.post.{build_number}.dev.{today}"
    elif is_dailybuild:
        return f"{base_version}.daily.{today}"
    else:
        return base_version

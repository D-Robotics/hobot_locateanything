"""
配置管理器
加载和解析配置文件
"""

import logging
import os
from typing import Dict, Optional

import yaml

logger = logging.getLogger(__name__)


class ConfigManager:
    """配置管理器类"""

    def __init__(self, config_path: str = "scm/tools/scheduler/config.yaml"):
        """
        初始化配置管理器

        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = {}
        self._load_config()

    def _load_config(self):
        """加载配置文件"""
        try:
            with open(self.config_path) as f:
                self.config = yaml.safe_load(f) or {}

            logger.info("配置文件加载成功")
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise

    def get_project_config(self, project_name: str) -> Optional[Dict]:
        """
        获取指定项目的配置

        Args:
            project_name: 项目名称

        Returns:
            Dict: 项目配置字典或None
        """
        try:
            projects = self.config.get("projects", {})
            project_config = projects.get(project_name)

            if not project_config:
                logger.warning(f"未找到项目 {project_name} 的配置")
                return None

            logger.debug(f"获取到项目 {project_name} 的配置")
            return project_config
        except Exception as e:
            logger.error(f"获取项目 {project_name} 配置失败: {e}")
            return None

    def get_jenkins_config(self) -> Dict:
        """
        获取Jenkins配置

        Returns:
            Dict: Jenkins配置字典
        """
        return self.config.get("jenkins", {})

    def get_logging_config(self) -> Dict:
        """
        获取日志配置

        Returns:
            Dict: 日志配置字典
        """
        return self.config.get("logging", {})

    def get_deps_version(self, version_key: str = "OE_LLM_VERSION") -> Optional[str]:
        """
        从 deps_version.conf 读取指定版本号

        Args:
            version_key: 版本号键名，默认 OE_LLM_VERSION

        Returns:
            Optional[str]: 版本号字符串，失败返回None
        """
        try:
            # deps_version.conf 位于项目根目录
            version_file = os.path.join(
                # 从 scheduler 目录向上 3 层到达项目根目录
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(self.config_path)))),
                "deps_version.conf",
            )

            if not os.path.exists(version_file):
                logger.warning(f"版本文件不存在: {version_file}")
                return None

            with open(version_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        if key.strip() == version_key:
                            version = value.strip()
                            logger.info(f"从 deps_version.conf 读取到 {version_key}: {version}")
                            return version

            logger.warning(f"未在 deps_version.conf 中找到 {version_key}")
            return None

        except Exception as e:
            logger.error(f"读取版本文件失败: {e}")
            return None

    def validate_config(self) -> bool:
        """
        验证配置的有效性

        Returns:
            bool: 配置是否有效
        """
        try:
            # 验证项目配置
            projects = self.config.get("projects", {})
            if not projects:
                logger.warning("项目配置为空")
                return False

            # 验证每个项目的必要字段
            for project_name, config in projects.items():
                required_fields = ["default_board_ip", "compile_yaml_path", "eval_yaml_path"]
                for field in required_fields:
                    if field not in config:
                        logger.error(f"项目 {project_name} 缺少必要字段: {field}")
                        return False

            # 验证Jenkins配置
            jenkins_config = self.config.get("jenkins", {})
            required_jenkins_fields = ["url", "username", "api_token", "compile_job_name", "eval_job_name"]
            for field in required_jenkins_fields:
                if field not in jenkins_config:
                    logger.error(f"Jenkins配置缺少必要字段: {field}")
                    return False

            logger.info("配置验证通过")
            return True
        except Exception as e:
            logger.error(f"配置验证失败: {e}")
            return False

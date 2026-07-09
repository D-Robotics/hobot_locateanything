"""
结果解析模块
解析特定目录下的tar.gz文件，提取性能指标信息
"""

import logging
from .config_manager import ConfigManager
import tarfile
import os
import csv
from typing import Dict, List, Optional
from pathlib import Path


logger = logging.getLogger(__name__)


class ResultParser:
    """结果解析器类"""

    def __init__(self, config_path: str = "scm/tools/scheduler/config.yaml"):
        """
        初始化结果解析器

        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config_manager = ConfigManager(config_path)

    def parse_results(self, project_name: str, eval_job_name: str = "",
                     build_number: int = None) -> Dict:
        """
        解析评测结果

        Args:
            project_name: 项目名称
            eval_job_name: 评测job名称
            build_number: 构建号

        Returns:
            Dict: 解析后的结果字典
        """
        logger.info(f"开始解析评测结果 - 项目: {project_name}, Job: {eval_job_name}, 构建号: {build_number}")

        # TODO: Phase 2 - 实现真实的评测结果解析逻辑
        # 1. 从Jenkins下载评测结果文件
        # 2. 解析tar.gz文件
        # 3. 提取性能指标
        # 4. 导出CSV文件

        logger.warning("评测结果解析功能尚未完全实现（Phase 2任务）")
        return {}

    def parse_results_legacy(self, tar_gz_paths: Optional[List[str]] = None) -> Dict:
        """
        解析tar.gz文件中的结果（遗留方法，用于向后兼容）

        Args:
            tar_gz_paths: tar.gz文件路径列表，如果为None则扫描默认目录

        Returns:
            Dict: 解析后的结果字典
        """
        try:
            if tar_gz_paths is None:
                # TODO: 实现默认目录扫描逻辑
                # 这里使用示例路径，实际应从配置中获取
                tar_gz_paths = self._scan_default_directory()

            results = {}

            for tar_gz_path in tar_gz_paths:
                try:
                    parsed_data = self._parse_single_tar_gz(tar_gz_path)
                    if parsed_data:
                        results[tar_gz_path] = parsed_data
                except Exception as e:
                    logger.error(f"解析文件 {tar_gz_path} 时出错: {e}")
                    # 继续处理其他文件

            logger.info(f"成功解析 {len(results)} 个文件")
            return results

        except Exception as e:
            logger.error(f"批量解析结果时出错: {e}")
            return {}

    def _scan_default_directory(self) -> List[str]:
        """
        扫描默认目录中的tar.gz文件

        Returns:
            List[str]: tar.gz文件路径列表
        """
        # TODO: 实现实际的目录扫描逻辑
        # 这里返回示例路径
        return [
            "/special/path/version/project/model1.tar.gz",
            "/special/path/version/project/model2.tar.gz"
        ]

    def _parse_single_tar_gz(self, tar_gz_path: str) -> Optional[Dict]:
        """
        解析单个tar.gz文件

        Args:
            tar_gz_path: tar.gz文件路径

        Returns:
            Dict: 解析后的数据字典
        """
        try:
            if not os.path.exists(tar_gz_path):
                logger.error(f"文件不存在: {tar_gz_path}")
                return None

            # 解压tar.gz文件
            extracted_dir = f"/tmp/extracted_{os.path.basename(tar_gz_path)}"
            os.makedirs(extracted_dir, exist_ok=True)

            with tarfile.open(tar_gz_path, "r:gz") as tar:
                tar.extractall(path=extracted_dir)

            # 查找性能指标文件（示例）
            performance_files = self._find_performance_files(extracted_dir)

            parsed_data = {}
            for perf_file in performance_files:
                try:
                    data = self._parse_performance_file(perf_file)
                    if data:
                        parsed_data.update(data)
                except Exception as e:
                    logger.error(f"解析性能文件 {perf_file} 时出错: {e}")

            # 清理临时目录
            self._cleanup_temp_directory(extracted_dir)

            return parsed_data

        except Exception as e:
            logger.error(f"解析单个tar.gz文件 {tar_gz_path} 时出错: {e}")
            return None

    def _find_performance_files(self, directory: str) -> List[str]:
        """
        在目录中查找性能指标文件

        Args:
            directory: 目录路径

        Returns:
            List[str]: 性能文件路径列表
        """
        # TODO: 实现实际的文件查找逻辑
        # 这里返回示例文件路径
        return [
            os.path.join(directory, "performance_metrics.txt"),
            os.path.join(directory, "results.json")
        ]

    def _parse_performance_file(self, file_path: str) -> Optional[Dict]:
        """
        解析性能指标文件

        Args:
            file_path: 文件路径

        Returns:
            Dict: 解析后的数据字典
        """
        try:
            if not os.path.exists(file_path):
                return None

            _, ext = os.path.splitext(file_path)
            if ext == ".txt":
                return self._parse_txt_file(file_path)
            elif ext == ".json":
                return self._parse_json_file(file_path)
            else:
                logger.warning(f"不支持的文件类型: {file_path}")
                return None

        except Exception as e:
            logger.error(f"解析性能文件 {file_path} 时出错: {e}")
            return None

    def _parse_txt_file(self, file_path: str) -> Dict:
        """
        解析文本格式的性能文件

        Args:
            file_path: 文件路径

        Returns:
            Dict: 解析后的数据字典
        """
        # TODO: 实现实际的文本解析逻辑
        # 这里返回示例数据
        return {
            "model_name": "example_model",
            "throughput": "1000",
            "latency": "5.2",
            "accuracy": "0.95"
        }

    def _parse_json_file(self, file_path: str) -> Dict:
        """
        解析JSON格式的性能文件

        Args:
            file_path: 文件路径

        Returns:
            Dict: 解析后的数据字典
        """
        # TODO: 实现实际的JSON解析逻辑
        # 这里返回示例数据
        return {
            "model_name": "example_model",
            "throughput": 1000,
            "latency": 5.2,
            "accuracy": 0.95
        }

    def save_results_to_csv(self, results: Dict, csv_path: str) -> bool:
        """
        将结果保存为CSV文件

        Args:
            results: 结果字典
            csv_path: CSV文件路径

        Returns:
            bool: 保存是否成功
        """
        try:
            if not results:
                logger.warning("没有结果需要保存")
                return False

            # 准备CSV数据
            csv_data = []
            header = ["model_name", "throughput", "latency", "accuracy"]

            for model_name, metrics in results.items():
                row = [
                    metrics.get("model_name", "unknown"),
                    metrics.get("throughput", ""),
                    metrics.get("latency", ""),
                    metrics.get("accuracy", "")
                ]
                csv_data.append(row)

            # 写入CSV文件
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header)
                writer.writerows(csv_data)

            logger.info(f"结果已保存到CSV文件: {csv_path}")
            return True

        except Exception as e:
            logger.error(f"保存CSV文件时出错: {e}")
            return False

    def _cleanup_temp_directory(self, directory: str):
        """
        清理临时目录

        Args:
            directory: 临时目录路径
        """
        try:
            import shutil
            if os.path.exists(directory):
                shutil.rmtree(directory)
                logger.debug(f"已清理临时目录: {directory}")
        except Exception as e:
            logger.error(f"清理临时目录时出错: {e}")

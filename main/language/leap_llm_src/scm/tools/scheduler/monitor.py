"""
任务监控模块
监控Jenkins任务状态
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
from .jenkins_client import JenkinsClient
from .config_manager import ConfigManager

logger = logging.getLogger(__name__)


class Monitor:
    """任务监控器类"""

    def __init__(self, config_path: str = "scm/tools/scheduler/config.yaml"):
        """
        初始化监控器

        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.jenkins_client = JenkinsClient(config_path)
        self.config_manager = ConfigManager(config_path)

        # 从配置文件获取轮询间隔和超时时间
        jenkins_config = self.config_manager.get_jenkins_config()
        self.poll_interval = jenkins_config.get('poll_interval', 30)
        self.compile_timeout = jenkins_config.get('compile_timeout', 36000)
        self.eval_timeout = jenkins_config.get('eval_timeout', 7200)

        logger.debug(f"监控器初始化完成，轮询间隔: {self.poll_interval}秒")

    def wait_for_job_completion(self, job_name: str, build_number: int,
                                timeout: Optional[int] = None) -> bool:
        """
        等待Job完成

        Args:
            job_name: Job名称
            build_number: 构建号
            timeout: 超时时间（秒），如果为None则使用配置文件中的默认值

        Returns:
            bool: Job是否成功完成
        """
        if timeout is None:
            # 根据job类型获取默认超时
            if 'compile' in job_name.lower():
                timeout = self.compile_timeout
            else:
                timeout = self.eval_timeout

        start_time = time.time()
        logger.info(f"开始监控Job {job_name}#{build_number} 的完成状态，超时: {timeout}秒")

        while True:
            # 检查是否超时
            elapsed = time.time() - start_time
            if elapsed > timeout:
                logger.error(f"Job {job_name}#{build_number} 超时，超过 {timeout} 秒")
                # 停止job
                self.jenkins_client.stop_job(job_name, build_number)
                return False

            # 获取Job状态
            status = self.jenkins_client.get_job_status(job_name, build_number)
            logger.info(f"Job {job_name}#{build_number} 当前状态: {status} (已运行: {elapsed:.1f}秒)")

            if status == "SUCCESS":
                logger.info(f"Job {job_name}#{build_number} 成功完成")
                return True
            elif status in ["FAILURE", "UNSTABLE", "ABORTED"]:
                logger.error(f"Job {job_name}#{build_number} 执行失败: {status}")
                # 输出失败信息
                self._log_failure_info(job_name, build_number)
                return False
            elif status == "IN_PROGRESS":
                logger.debug(f"Job {job_name}#{build_number} 正在运行中，等待 {self.poll_interval} 秒后重试")
                time.sleep(self.poll_interval)
            elif status in ["QUEUED", "PENDING"]:
                logger.debug(f"Job {job_name}#{build_number} 在队列中等待，等待 {self.poll_interval} 秒后重试")
                time.sleep(self.poll_interval)
            else:
                logger.warning(f"Job {job_name}#{build_number} 状态未知: {status}")
                time.sleep(self.poll_interval)

    def wait_for_multiple_jobs(self, jobs: List[Tuple[str, int]],
                              timeout: Optional[int] = None) -> Dict[str, str]:
        """
        监控多个Job的完成状态

        Args:
            jobs: Job列表，每个元素是 (job_name, build_number) 元组
            timeout: 总体超时时间（秒）

        Returns:
            Dict[str, str]: job_id: status 字典
        """
        if timeout is None:
            timeout = self.jenkins_client.get_job_timeout('compile') * 2

        job_dict = {f"{job_name}_{build_number}": (job_name, build_number) for job_name, build_number in jobs}
        results = {job_id: "PENDING" for job_id in job_dict.keys()}
        start_time = time.time()

        logger.info(f"开始监控 {len(jobs)} 个Job")

        while True:
            # 检查是否超时
            if time.time() - start_time > timeout:
                logger.error(f"监控超时，超过 {timeout} 秒")
                for job_id in job_dict:
                    if results[job_id] in ["PENDING", "IN_PROGRESS"]:
                        results[job_id] = "TIMEOUT"
                        # 停止超时的job
                        job_name, build_number = job_dict[job_id]
                        self.jenkins_client.stop_job(job_name, build_number)
                break

            # 检查所有Job的状态
            all_finished = True
            for job_id in job_dict:
                job_name, build_number = job_dict[job_id]

                if results[job_id] in ["PENDING", "IN_PROGRESS", "QUEUED"]:
                    status = self.jenkins_client.get_job_status(job_name, build_number)
                    results[job_id] = status

                    logger.debug(f"Job {job_id} 状态: {status}")

                    if status in ["PENDING", "IN_PROGRESS", "QUEUED"]:
                        all_finished = False
                    elif status in ["FAILURE", "UNSTABLE", "ABORTED"]:
                        # 输出失败信息
                        self._log_failure_info(job_name, build_number)

            if all_finished:
                logger.info("所有Job已完成")
                break

            logger.debug(f"等待 {self.poll_interval} 秒后重试...")
            time.sleep(self.poll_interval)

        return results

    def _log_failure_info(self, job_name: str, build_number: int):
        """
        输出Job失败信息

        Args:
            job_name: Job名称
            build_number: 构建号
        """
        try:
            console_output = self.jenkins_client.get_console_output(job_name, build_number)
            if console_output:
                # 输出最后100行
                lines = console_output.split('\n')
                logger.error(f"Job {job_name}#{build_number} 最后100行日志:")
                for line in lines[-100:]:
                    logger.error(f"  {line}")
        except Exception as e:
            logger.warning(f"无法获取Job {job_name}#{build_number} 的失败输出: {e}")

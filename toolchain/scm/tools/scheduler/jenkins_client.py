# Copyright (c) 2024 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.
"""
简化的 Jenkins 客户端

提供简洁直观的 API 来操作 Jenkins Job。

使用示例:
    # 1. 初始化
    client = JenkinsClient("scm/tools/scheduler/config.yaml")

    # 2. 触发 Job
    build_number = client.build_job('toolchain/integration/job-name',
                                    {'branch': 'master', 'model': 'qwen2_5'})

    # 3. 查询状态
    status = client.get_job_status('toolchain/integration/job-name', 123)
    # 返回: SUCCESS, FAILURE, IN_PROGRESS, UNSTABLE, ABORTED, UNKNOWN, ERROR

    # 4. 获取构建详情
    info = client.get_build_info('toolchain/integration/job-name', 123)
    # 返回: {'number': 123, 'result': 'SUCCESS', 'timestamp': ..., 'parameters': [...]}

    # 5. 停止 Job
    client.stop_job('job_name', 123)

    # 6. 获取控制台输出
    output = client.get_console_output('job_name', 123)
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import jenkins
import yaml


class JenkinsClient:
    """简化的 Jenkins 客户端

    提供简洁的 API 来操作 Jenkins Job。

    Attributes:
        jenkins_url: Jenkins 服务地址
        username: 用户名
        server: Jenkins API 客户端
    """

    def __init__(self, config_path: Optional[str] = None, **kwargs):
        """初始化 Jenkins 客户端

        Args:
            config_path: 配置文件路径（YAML），如果为 None 则使用参数初始化
            **kwargs: 直接传入的配置参数，当 config_path 为 None 时使用

        Raises:
            ValueError: 如果配置无效或缺少必要参数

        配置文件格式:
            jenkins:
                url: "https://jenkins.example.com"
                username: "your-username"
                token_env: "JENKINS_TOKEN"  # 或直接用 token: "your-token"
                timeout: 30
        """
        self.jenkins_url = None
        self.username = None
        self.token = None
        self.timeout = 30

        if config_path:
            self._load_from_config(config_path)
        elif kwargs:
            self._load_from_kwargs(kwargs)
        else:
            raise ValueError("Either config_path or parameters must be provided")

        self.server = jenkins.Jenkins(
            self.jenkins_url,
            username=self.username,
            password=self.token,
            timeout=self.timeout
        )

        logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)

    def _load_from_config(self, config_path: str):
        """从配置文件加载配置

        Args:
            config_path: 配置文件路径
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        jenkins_config = config.get("jenkins", {})
        self.jenkins_url = jenkins_config.get("url")
        self.username = jenkins_config.get("username")
        token_env = jenkins_config.get("token_env")
        token_direct = jenkins_config.get("token") or jenkins_config.get("api_token")
        self.timeout = jenkins_config.get("timeout", 30)

        if not self.jenkins_url or not self.username:
            raise ValueError("jenkins.url and jenkins.username are required in config")

        self.token = token_direct or os.getenv(token_env) if token_env else os.getenv("JENKINS_TOKEN") or os.getenv("JENKINS_API_TOKEN")

        if not self.token and not token_direct:
            raise ValueError("jenkins.token or jenkins.token_env must be provided")

    def _load_from_kwargs(self, kwargs: Dict[str, Any]):
        """从参数加载配置

        Args:
            kwargs: 参数字典
        """
        self.jenkins_url = kwargs.get("jenkins_url")
        self.username = kwargs.get("username")
        self.token = kwargs.get("token") or os.getenv("JENKINS_TOKEN") or os.getenv("JENKINS_API_TOKEN")
        self.timeout = kwargs.get("timeout", 30)

        if not self.jenkins_url or not self.username:
            raise ValueError("jenkins_url and username are required")

        if not self.token:
            raise ValueError("token must be provided or set in JENKINS_TOKEN env var")

    # ==================== 基础操作 ====================

    def build_job(self, job_name: str, parameters: Optional[Dict[str, Any]] = None) -> int:
        """触发 Jenkins Job

        Args:
            job_name: Job 名称
            parameters: 构建参数

        Returns:
            int: 构建编号 (build_number)

        Raises:
            jenkins.JenkinsException: 触发失败时抛出

        Example:
            >>> client = JenkinsClient("config.yaml")
            >>> build_number = client.build_job('my-job', {'branch': 'main'})
            >>> print(build_number)
            123
        """
        logging.info(f"Building job: {job_name} with params: {parameters}")
        queue_id = self.server.build_job(job_name, parameters or {})

        build_number = self._get_build_number_from_queue(queue_id)
        if not build_number:
            raise jenkins.JenkinsException(f"Failed to get build number for job: {job_name}")

        return build_number

    def get_job_status(self, job_name: str, build_number: int) -> str:
        """获取 Job 状态

        Args:
            job_name: Job 名称
            build_number: 构建编号

        Returns:
            str: 状态值
            - SUCCESS: 构建成功
            - FAILURE: 构建失败
            - IN_PROGRESS: 构建中
            - UNSTABLE: 不稳定
            - ABORTED: 中止
            - UNKNOWN: 未知状态
            - ERROR: 获取信息出错

        Example:
            >>> status = client.get_job_status('my-job', 123)
            >>> print(status)
            'SUCCESS'
        """
        try:
            build_info = self.server.get_build_info(job_name, build_number)
            building = build_info.get("building", False)
            result = build_info.get("result")

            if building:
                return "IN_PROGRESS"

            if result:
                if result == "SUCCESS":
                    return "SUCCESS"
                elif result == "FAILURE":
                    return "FAILURE"
                elif result == "UNSTABLE":
                    return "UNSTABLE"
                elif result == "ABORTED":
                    return "ABORTED"

            return "UNKNOWN"

        except Exception as e:
            logging.error(f"Error getting job status: {e}")
            return "ERROR"

    def get_build_info(self, job_name: str, build_number: int) -> Dict[str, Any]:
        """获取构建详细信息

        Args:
            job_name: Job 名称
            build_number: 构建编号

        Returns:
            Dict: 构建信息字典，包含:
            - number: 构建编号
            - result: 构建结果
            - building: 是否正在构建
            - timestamp: 开始时间戳
            - duration: 持续时间（毫秒）
            - url: 构建URL
            - parameters: 构建参数列表

        Example:
            >>> info = client.get_build_info('my-job', 123)
            >>> print(f"Result: {info['result']}, Duration: {info['duration']}ms")
        """
        try:
            build_info = self.server.get_build_info(job_name, build_number)

            params = []
            for action in build_info.get("actions", []):
                if "parameters" in action:
                    params = [{"name": p["name"], "value": p["value"]} for p in action["parameters"]]
                    break

            return {
                "number": build_number,
                "result": build_info.get("result"),
                "building": build_info.get("building", False),
                "timestamp": build_info.get("timestamp"),
                "duration": build_info.get("duration"),
                "url": build_info.get("url"),
                "parameters": params
            }
        except Exception as e:
            logging.error(f"Error getting build info: {e}")
            return {}

    def stop_job(self, job_name: str, build_number: int) -> bool:
        """停止正在运行的 Job

        Args:
            job_name: Job 名称
            build_number: 构建编号

        Returns:
            bool: 成功返回 True，失败返回 False

        Example:
            >>> client.stop_job('my-job', 123)
            True
        """
        try:
            self.server.stop_build(job_name, build_number)
            logging.info(f"Stopped job: {job_name}:{build_number}")
            return True
        except Exception as e:
            logging.error(f"Failed to stop job {job_name}:{build_number}: {e}")
            return False

    def get_console_output(self, job_name: str, build_number: int) -> str:
        """获取构建控制台输出

        Args:
            job_name: Job 名称
            build_number: 构建编号

        Returns:
            str: 控制台输出文本

        Example:
            >>> output = client.get_console_output('my-job', 123)
            >>> print(output)
            # 控制台完整日志...
        """
        try:
            return self.server.get_build_console_output(job_name, build_number)
        except Exception as e:
            logging.error(f"Error getting console output: {e}")
            return ""

    # ==================== 扩展操作 ====================

    def retry_job(
        self,
        job_name: str,
        previous_build_number: int,
        parameters: Optional[Dict[str, Any]] = None
    ) -> int:
        """重新运行失败的 Job

        Args:
            job_name: Job 名称
            previous_build_number: 上次失败的构建编号
            parameters: 可选的新参数，如果为 None 则使用上次参数

        Returns:
            int: 新的构建编号

        Example:
            >>> client.retry_job('my-job', 123)
            124
        """
        if parameters is None:
            build_info = self.get_build_info(job_name, previous_build_number)
            parameters = {p["name"]: p["value"] for p in build_info.get("parameters", [])}

        return self.build_job(job_name, parameters)

    def wait_for_completion(
        self,
        job_name: str,
        build_number: int,
        timeout: int = 3600,
        interval: int = 30
    ) -> str:
        """等待构建完成

        Args:
            job_name: Job 名称
            build_number: 构建编号
            timeout: 超时时间（秒）
            interval: 轮询间隔（秒）

        Returns:
            str: 最终状态

        Raises:
            TimeoutError: 超时抛出

        Example:
            >>> status = client.wait_for_completion('my-job', 123, timeout=1800)
            >>> print(status)
            'SUCCESS'
        """
        import time

        elapsed = 0
        while elapsed < timeout:
            status = self.get_job_status(job_name, build_number)

            if status not in ["IN_PROGRESS", "UNKNOWN"]:
                return status

            logging.info(f"Job {job_name}:{build_number} is {status}... ({elapsed}s/{timeout}s)")
            time.sleep(interval)
            elapsed += interval

        raise TimeoutError(f"Job {job_name}:{build_number} did not complete within {timeout}s")

    def is_job_running(self, job_name: str) -> bool:
        """检查 Job 是否正在运行

        Args:
            job_name: Job 名称

        Returns:
            bool: 正在运行返回 True

        Example:
            >>> if client.is_job_running('my-job'):
            ...     print("Job is still running...")
        """
        try:
            job_info = self.server.get_job_info(job_name)
            builds = job_info.get("builds", [])

            for build in builds[:5]:
                build_info = self.server.get_build_info(job_name, build["number"])
                if build_info.get("building", False):
                    return True

            return False

        except Exception as e:
            logging.error(f"Error checking if job is running: {e}")
            return False

    def get_last_build(self, job_name: str) -> Optional[int]:
        """获取最后一次构建编号

        Args:
            job_name: Job 名称

        Returns:
            Optional[int]: 构建编号，无构建返回 None

        Example:
            >>> last_build = client.get_last_build('my-job')
            >>> if last_build:
            ...     print(f"Last build: {last_build}")
        """
        try:
            job_info = self.server.get_job_info(job_name)
            last_build = job_info.get("lastBuild")
            return last_build.get("number") if last_build else None
        except Exception as e:
            logging.error(f"Error getting last build: {e}")
            return None

    # ==================== 私有方法 ====================

    def _get_build_number_from_queue(self, queue_id: int, max_retries: int = 30, retry_interval: int = 3) -> Optional[int]:
        """从队列 ID 获取构建编号

        Args:
            queue_id: 队列 ID
            max_retries: 最大重试次数
            retry_interval: 重试间隔（秒）

        Returns:
            Optional[int]: 构建编号
        """
        for attempt in range(max_retries):
            try:
                queue_item = self.server.get_queue_item(queue_id)
                if "executable" in queue_item and "number" in queue_item["executable"]:
                    return queue_item["executable"]["number"]
            except Exception as e:
                logging.debug(f"Waiting for build number... ({attempt + 1}/{max_retries})")

            import time
            time.sleep(retry_interval)

        return None


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python jenkins_client.py <config_file> [command] [args...]")
        print("\nCommands:")
        print("  build <job_name> [param1=value1 ...]")
        print("  status <job_name> <build_number>")
        print("  info <job_name> <build_number>")
        print("  stop <job_name> <build_number>")
        print("  console <job_name> <build_number>")
        print("  running <job_name>")
        sys.exit(1)

    config_path = sys.argv[1]
    command = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        client = JenkinsClient(config_path)

        if command == "build" and len(sys.argv) >= 4:
            job_name = sys.argv[3]
            params = {}
            for arg in sys.argv[4:]:
                if "=" in arg:
                    k, v = arg.split("=", 1)
                    params[k] = v
            build_num = client.build_job(job_name, params if params else None)
            print(f"Build triggered: {job_name}:{build_num}")

        elif command == "status" and len(sys.argv) == 5:
            job_name = sys.argv[3]
            build_num = int(sys.argv[4])
            status = client.get_job_status(job_name, build_num)
            print(f"Status: {status}")

        elif command == "info" and len(sys.argv) == 5:
            job_name = sys.argv[3]
            build_num = int(sys.argv[4])
            info = client.get_build_info(job_name, build_num)
            print(f"Build Info:")
            for k, v in info.items():
                print(f"  {k}: {v}")

        elif command == "stop" and len(sys.argv) == 5:
            job_name = sys.argv[3]
            build_num = int(sys.argv[4])
            result = client.stop_job(job_name, build_num)
            print(f"Stop result: {result}")

        elif command == "console" and len(sys.argv) == 5:
            job_name = sys.argv[3]
            build_num = int(sys.argv[4])
            output = client.get_console_output(job_name, build_num)
            print("Console output:")
            print(output)

        elif command == "running" and len(sys.argv) == 4:
            job_name = sys.argv[3]
            is_running = client.is_job_running(job_name)
            print(f"Running: {is_running}")

        else:
            print("Invalid command or arguments")
            sys.exit(1)

    except Exception as e:
        logging.error(f"Error: {e}")
        sys.exit(1)

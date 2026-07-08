"""
主调度器类
负责协调整个编译和评测流程
"""

import argparse
import logging
import os
import re
import sys
from datetime import datetime
from typing import Dict, Optional

from scm.tools.scheduler.config_manager import ConfigManager
from scm.tools.scheduler.jenkins_client import JenkinsClient
from scm.tools.scheduler.monitor import Monitor
from scm.tools.scheduler.parser import ResultParser
from scm.tools.scheduler.utils import setup_logging

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logger = logging.getLogger(__name__)


class Scheduler:
    """主调度器类"""

    def __init__(self, config_path: str = "scm/tools/scheduler/config.yaml"):
        """
        初始化调度器

        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config_manager = ConfigManager(config_path)
        self.jenkins_client = JenkinsClient(config_path)
        self.monitor = Monitor(config_path)
        self.parser = ResultParser(config_path)

        # 存储任务信息
        self.job_queue = []
        self.results = {}

    def schedule_compile_and_test(
        self,
        project_name: str,
        is_dailybuild: bool = False,
        skip_compile: bool = False,
        skip_eval: bool = False,
        ppl_test: bool = False,
        branch_name: str = "",
    ) -> bool:
        """
        调度编译和评测流程

        Args:
            project_name: 项目名称
            is_dailybuild: 是否是dailybuild
            skip_compile: 是否跳过编译阶段
            skip_eval: 是否跳过评测阶段
            ppl_test: 是否运行PPL测试
            branch_name: 可选，覆盖 compile/eval 的分支配置

        Returns:
            bool: 执行是否成功
        """
        logger.info(f"开始调度项目 {project_name} 的编译和评测流程")
        logger.info(
            f"参数 - dailybuild: {is_dailybuild}, "
            f"skip_compile: {skip_compile}, "
            f"skip_eval: {skip_eval}, "
            f"ppl_test: {ppl_test}, "
            f"branch: {branch_name or 'config-default'}"
        )

        try:
            # 1. 获取项目配置
            project_config = self.config_manager.get_project_config(project_name)
            if not project_config:
                logger.error(f"无法获取项目 {project_name} 的配置")
                return False
            project_config = self._apply_branch_override(project_config, branch_name)

            # 2. 获取Jenkins配置
            jenkins_config = self.config_manager.get_jenkins_config()
            compile_job_name = jenkins_config.get("compile_job_name")
            eval_job_name = jenkins_config.get("eval_job_name")

            logger.info(f"编译Job: {compile_job_name}")
            logger.info(f"评测Job: {eval_job_name}")

            # 3. 获取基础版本号 - 优先从 deps_version.conf 读取
            compile_config = project_config.get("compile", {})
            base_version = self.config_manager.get_deps_version("OE_LLM_VERSION")

            # 如果读取失败，使用配置文件中的默认值
            if not base_version:
                base_version = compile_config.get("default_model_version", "1.0.0")
                logger.warning(f"使用配置文件中的版本号: {base_version}")
            else:
                logger.info(f"使用 deps_version.conf 中的版本号: {base_version}")

            actual_model_version = base_version
            compile_build_numbers = {}

            # 传给编译job的MODEL_VERSION使用base_version，由Jenkins job内部动态修改
            model_version = base_version
            logger.info(f"传给编译job的MODEL_VERSION: {model_version}")

            # 4. 如果PPL测试，发出警告（暂不支持）
            if ppl_test:
                logger.warning("PPL测试暂不支持，将按照普通模式执行")

            # 5. 执行编译（如果未跳过）
            if not skip_compile:
                if ppl_test:
                    # PPL测试：编译两次（启用/禁用PPL）
                    # 第一次：不启用PPL
                    logger.info("触发编译job（不启用PPL）...")
                    compile_params = self._prepare_compile_params(
                        project_config, model_version, enable_ppl=False, is_dailybuild=is_dailybuild
                    )
                    build_number_no_ppl = self._trigger_compile_job(compile_job_name, compile_params)
                    if build_number_no_ppl:
                        compile_build_numbers["no_ppl"] = (compile_job_name, build_number_no_ppl)
                    else:
                        logger.error("编译job（不启用PPL）启动失败")
                        return False

                    # 第二次：启用PPL
                    logger.info("触发编译job（启用PPL）...")
                    compile_params = self._prepare_compile_params(
                        project_config, model_version, enable_ppl=True, is_dailybuild=is_dailybuild
                    )
                    build_number_with_ppl = self._trigger_compile_job(compile_job_name, compile_params)
                    if build_number_with_ppl:
                        compile_build_numbers["with_ppl"] = (compile_job_name, build_number_with_ppl)
                    else:
                        logger.error("编译job（启用PPL）启动失败")
                        return False

                else:
                    # 普通编译：编译一次
                    logger.info("触发编译job...")
                    compile_params = self._prepare_compile_params(
                        project_config,
                        model_version,
                        enable_ppl=compile_config.get("enable_ppl", False),
                        is_dailybuild=is_dailybuild,
                    )
                    build_number = self._trigger_compile_job(compile_job_name, compile_params)
                    if build_number:
                        compile_build_numbers["main"] = (compile_job_name, build_number)
                    else:
                        logger.error("编译job启动失败")
                        return False

                # 6. 根据build_number和日期生成实际的MODEL_VERSION（拼接版本）
                main_job_name, main_build_number = compile_build_numbers.get(
                    "main", next(iter(compile_build_numbers.values())) if compile_build_numbers else (None, None)
                )
                if main_build_number:
                    # 拼接逻辑：MODEL_VERSION + ".post" + BUILD_NUMBER + ".dev" + 短日期(YYYYMMDD)
                    short_date = datetime.now().strftime("%Y%m%d")
                    actual_model_version = f"{base_version}.post{main_build_number}.dev{short_date}"
                    logger.info(f"生成的实际模型版本号: {actual_model_version}")
                else:
                    logger.warning("无法获取build_number，使用base_version")
                    actual_model_version = base_version

                # 7. 打印JFS-Public保存路径
                project_name_key = compile_config.get("project", project_name)
                jfs_path = self._get_jfs_public_path(actual_model_version, project_name_key)
                logger.info(f"JFS-Public保存路径: {jfs_path}")

                # 8. 监控编译job直到完成
                logger.info("等待所有编译job完成...")
                for _, (job_name, bn) in compile_build_numbers.items():
                    if not self.monitor.wait_for_job_completion(job_name, bn):
                        logger.error(f"编译job {job_name}#{bn} 执行失败")
                        return False

                logger.info("所有编译job已完成")
            else:
                logger.info("跳过编译阶段")
                # 跳过编译时，使用配置中的版本号作为实际版本
                actual_model_version = base_version
                compile_build_numbers = {}

            # 9. 执行评测（如果未跳过）
            if not skip_eval:
                logger.info("开始触发评测job")
                hbdk_model_compile_version = self._resolve_hbdk_compiler_version(
                    compile_build_numbers=compile_build_numbers,
                    is_dailybuild=is_dailybuild,
                )
                main_job_name, main_build_number = compile_build_numbers.get(
                    "main", next(iter(compile_build_numbers.values())) if compile_build_numbers else (None, None)
                )
                compile_job_url = self._get_job_build_url(main_job_name, main_build_number)
                eval_params = self._prepare_eval_params(
                    project_config,
                    actual_model_version,
                    is_dailybuild=is_dailybuild,
                    hbdk_model_compile_version=hbdk_model_compile_version,
                    compile_job_url=compile_job_url,
                )
                eval_build_number = self._trigger_eval_job(eval_job_name, eval_params)

                if eval_build_number:
                    # 8. 监控评测job直到完成
                    if not self.monitor.wait_for_job_completion(eval_job_name, eval_build_number):
                        logger.error(f"评测job {eval_job_name}#{eval_build_number} 执行失败")
                        return False

                    logger.info(f"评测job {eval_job_name}#{eval_build_number} 已完成")

                    # 9. 解析评测结果（如果配置了解析器）
                    try:
                        logger.info("开始解析评测结果")
                        parsed_results = self.parser.parse_results(project_name, eval_job_name, eval_build_number)
                        self.results = parsed_results
                        logger.info("评测结果解析完成")
                    except Exception as e:
                        logger.warning(f"解析评测结果时出错（不影响主流程）: {e}")
                else:
                    logger.error("评测job启动失败")
                    return False
            else:
                logger.info("跳过评测阶段")

            logger.info(f"项目 {project_name} 的编译和评测流程完成")
            return True

        except Exception as e:
            logger.error(f"调度过程中发生错误: {e}", exc_info=True)
            return False

    def _prepare_compile_params(
        self, project_config: Dict, model_version: str, enable_ppl: bool = False, is_dailybuild: bool = False
    ) -> Dict[str, str]:
        """
        准备编译job的参数

        Args:
            project_config: 项目配置
            model_version: 模型版本号
            enable_ppl: 是否启用PPL
            is_dailybuild: 是否是dailybuild

        Returns:
            Dict[str, str]: 编译参数
        """
        compile_config = project_config.get("compile", {})

        params = {
            "CONFIG_PATH": compile_config.get("config_path", ""),
            "MODEL_VERSION": model_version,
            "PROJECT": compile_config.get("project", ""),
            "ENABLE_PPL": str(enable_ppl).lower(),
            "IS_TEST": str(compile_config.get("is_test", False)).lower(),
            "IS_DAILYBUILD": str(is_dailybuild).lower(),
            "UPLOAD_JFS_PUBLIC": str(compile_config.get("upload_jfs_public", False)).lower(),
        }

        # 如果有branch配置
        if "branch" in compile_config:
            params["BRANCH_NAME"] = compile_config["branch"]

        logger.debug(f"编译参数: {params}")
        return params

    def _apply_branch_override(self, project_config: Dict, branch_name: str) -> Dict:
        """如传入 branch_name，则统一覆盖 compile/evaluation 的 branch。"""
        if not branch_name:
            return project_config
        compile_config = project_config.setdefault("compile", {})
        eval_config = project_config.setdefault("evaluation", {})
        compile_config["branch"] = branch_name
        eval_config["branch"] = branch_name
        logger.info(f"使用指定分支覆盖 compile/eval branch: {branch_name}")
        return project_config

    def _prepare_eval_params(
        self,
        project_config: Dict,
        performance_version: str,
        is_dailybuild: bool = False,
        hbdk_model_compile_version: str = "",
        compile_job_url: str = "",
    ) -> Dict[str, str]:
        """
        准备评测job的参数

        Args:
            project_config: 项目配置
            performance_version: 性能测试版本号
            is_dailybuild: 是否是dailybuild

        Returns:
            Dict[str, str]: 评测参数
        """
        eval_config = project_config.get("evaluation", {})

        # 根据 is_dailybuild 动态设置 BUILD_TYPE
        # dailybuild 模式使用 "test"，正常模式使用配置文件中的值（默认 release）
        build_type = "test" if is_dailybuild else eval_config.get("build_type", "release")

        params = {
            "BOARD_IP": eval_config.get("board_ip", ""),
            "BOARD_USER": eval_config.get("board_user", ""),
            "BOARD_PASSWD": eval_config.get("board_passwd", ""),
            "BOARD_SSH_PORT": eval_config.get("board_ssh_port", "22"),
            "PROJECT": eval_config.get("project", ""),
            "PERFORMANCE_VERSION": performance_version,
            "BUILD_TYPE": build_type,
            "IS_DAILYBUILD": str(is_dailybuild).lower(),
            "SPECIFIED_MODEL": eval_config.get("specified_model", ""),
            "HBDK_MODEL_COMPILE_VERSION": hbdk_model_compile_version,
            "COMPILE_JOB_URL": compile_job_url,
        }

        # 如果有branch配置
        if "branch" in eval_config:
            params["gitlabSourceRepoBranch"] = eval_config["branch"]

        logger.debug(f"评测参数: {params}")
        return params

    def _resolve_hbdk_compiler_version(
        self,
        compile_build_numbers: Dict[str, tuple[str, int]],
        is_dailybuild: bool,
    ) -> str:
        """
        获取编译实际使用的 HBDK compiler 版本号。

        Args:
            compile_build_numbers: 编译job信息
            is_dailybuild: 是否是dailybuild模式

        Returns:
            str: hbdk compiler 版本号
        """
        main_job_name, main_build_number = compile_build_numbers.get(
            "main", next(iter(compile_build_numbers.values())) if compile_build_numbers else (None, None)
        )
        if main_job_name and main_build_number:
            try:
                console_output = self.jenkins_client.get_console_output(main_job_name, main_build_number) or ""
                marker_matches = re.findall(
                    r"HBDK_MODEL_COMPILE_VERSION_RESOLVED=([0-9A-Za-z.+_:-]+)",
                    console_output,
                )
                if marker_matches:
                    resolved = marker_matches[-1]
                    logger.info(f"从固定标记解析到 HBDK compiler 版本: {resolved}")
                    return resolved
            except Exception as e:
                logger.warning(f"解析编译日志中的 HBDK 版本失败: {e}")

        # dailybuild 下未解析到时标记为 unknown(--pre)
        if is_dailybuild:
            return "unknown(--pre)"

        return "unknown"

    def _get_job_build_url(self, job_name: Optional[str], build_number: Optional[int]) -> str:
        """获取 Jenkins job 构建链接。"""
        if not job_name or not build_number:
            return ""
        try:
            build_info = self.jenkins_client.get_build_info(job_name, build_number) or {}
            build_url = build_info.get("url", "")
            if build_url:
                return build_url
        except Exception as e:
            logger.warning(f"获取 Jenkins 构建链接失败: {e}")

        # fallback: 拼接 Jenkins URL
        base_url = self.config_manager.get_jenkins_config().get("url", "").rstrip("/")
        if not base_url:
            return ""
        return f"{base_url}/job/{job_name.replace('/', '/job/')}/{build_number}/"

    def _get_jfs_public_path(self, model_version: str, project: str) -> str:
        """
        计算JFS-Public保存路径

        Args:
            model_version: 模型版本号
            project: 项目名称

        Returns:
            str: JFS-Public保存路径
        """
        version_type = "test" if "post" in model_version else "release"
        return f"/jfs-public/openexplorer_llm/{version_type}/models/{model_version}/{project}"

    def _trigger_compile_job(self, job_name: str, params: Dict[str, str]) -> Optional[int]:
        """
        触发编译job

        Args:
            job_name: Job名称
            params: 参数字典

        Returns:
            Optional[int]: 构建号，失败返回None
        """
        build_number = self.jenkins_client.build_job(job_name, params)
        if build_number:
            logger.info(f"编译job {job_name} 已启动，构建号: {build_number}")
        else:
            logger.error(f"编译job {job_name} 启动失败")
        return build_number

    def _trigger_eval_job(self, job_name: str, params: Dict[str, str]) -> Optional[int]:
        """
        触发评测job

        Args:
            job_name: Job名称
            params: 参数字典

        Returns:
            Optional[int]: 构建号，失败返回None
        """
        build_number = self.jenkins_client.build_job(job_name, params)
        if build_number:
            logger.info(f"评测job {job_name} 已启动，构建号: {build_number}")
        else:
            logger.error(f"评测job {job_name} 启动失败")
        return build_number

    def parse_results(self, project_name: str, project_config: Dict):
        """
        解析评测结果

        Args:
            project_name: 项目名称
            project_config: 项目配置
        """
        pass


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="OpenExplorer LLM 调度器 - 自动化LLM模型编译和评测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 普通编译和评测
  python3 -u scm/tools/scheduler/scheduler.py --project public

  # Dailybuild模式
  python3 -u scm/tools/scheduler/scheduler.py --project public --is-dailybuild

  # 跳过编译阶段（仅评测）
  python3 -u scm/tools/scheduler/scheduler.py --project public --skip-compile

  # 跳过评测阶段（仅编译）
  python3 -u scm/tools/scheduler/scheduler.py --project public --skip-eval

  # 启用PPL对比测试
  python3 -u scm/tools/scheduler/scheduler.py --project public --ppl-test

  # 使用自定义配置文件
  python3 -u scm/tools/scheduler/scheduler.py --project public --config my_config.yaml
        """,
    )

    # 必需参数
    parser.add_argument(
        "--project",
        required=True,
        choices=["public", "saturnv", "deeproute"],
        help="项目名称 (public/saturnv/deeproute)",
    )

    # 可选参数
    parser.add_argument("--is-dailybuild", action="store_true", help="是否是dailybuild模式")
    parser.add_argument("--branch", default="", help="可选，覆盖 compile/eval 使用的代码分支")

    parser.add_argument("--skip-compile", action="store_true", help="跳过编译阶段，直接进入评测（用于调试历史模型）")

    parser.add_argument("--skip-eval", action="store_true", help="跳过评测阶段（用于仅验证编译流程）")

    parser.add_argument(
        "--ppl-test",
        action="store_true",
        help="【暂不支持】是否运行PPL测试，如果运行，会单独编译模型并运行PPL测试（仅特定模型支持）",
    )

    parser.add_argument(
        "--config",
        default="scm/tools/scheduler/config.yaml",
        help="配置文件路径（默认: scm/tools/scheduler/config.yaml）",
    )

    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志级别（默认: INFO）"
    )

    return parser.parse_args()


def main():
    """主函数入口"""
    # 解析命令行参数
    args = parse_arguments()

    # 配置日志
    setup_logging(log_level=args.log_level)

    logger.info("=" * 60)
    logger.info("OpenExplorer LLM 调度器启动")
    logger.info("=" * 60)
    logger.info(f"项目: {args.project}")
    logger.info(f"是否Dailybuild: {args.is_dailybuild}")
    logger.info(f"跳过编译: {args.skip_compile}")
    logger.info(f"跳过评测: {args.skip_eval}")
    logger.info(f"PPL测试: {args.ppl_test}")
    logger.info(f"分支覆盖: {args.branch or 'config-default'}")
    logger.info(f"配置文件: {args.config}")
    logger.info("=" * 60)

    try:
        # 初始化调度器
        scheduler = Scheduler(config_path=args.config)

        # 执行调度
        success = scheduler.schedule_compile_and_test(
            project_name=args.project,
            is_dailybuild=args.is_dailybuild,
            skip_compile=args.skip_compile,
            skip_eval=args.skip_eval,
            ppl_test=args.ppl_test,
            branch_name=args.branch,
        )

        if success:
            logger.info("=" * 60)
            logger.info("调度流程成功完成")
            logger.info("=" * 60)
            return 0
        else:
            logger.error("=" * 60)
            logger.error("调度流程执行失败")
            logger.error("=" * 60)
            return 1

    except Exception as e:
        logger.error(f"调度器执行异常: {e}", exc_info=True)
        logger.error("=" * 60)
        logger.error("调度器异常退出")
        logger.error("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())

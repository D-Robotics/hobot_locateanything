# Copyright (c) 2024 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

"""简化的飞书通知类"""

import logging
import os
import uuid
from typing import Any

from larker import Larker

# 默认配置（支持环境变量覆盖）
DEFAULT_CARD_ID = os.getenv("FEISHU_CARD_ID", "AAqvPKuXotMHm")
DEFAULT_GROUP_NAME = os.getenv("FEISHU_GROUP_NAME", "模块间CICD调试")


class SimpleFeishuNotifier:
    """简化的飞书通知类，支持 Markdown 卡片通知和文本通知"""

    def __init__(self, larker: Larker | None = None):
        """
        初始化通知器

        Args:
            larker: Larker 实例，如果不提供则自动创建
        """
        self.larker = larker or Larker()
        self.client = self.larker.client

    def send_markdown_notification(
        self,
        card_id: str,
        markdown_text: str,
        group_name: str,
        title: str = "",
        **extra_variables: Any,
    ) -> bool:
        """
        发送 Markdown 卡片通知

        Args:
            card_id: 飞书卡片模板ID
            markdown_text: Markdown 格式的文本，会填入卡片的 content 变量
                Example: "这是一段 **粗体** 文本，包含[链接](https://example.com)"
            group_name: 接收通知的群组名称
            title: 卡片标题，会填入卡片的 title 变量
            **extra_variables: 额外的卡片变量，如 {"module": "Qwen2.5", "status": "成功"}

        Returns:
            bool: 发送是否成功

        Example:
            >>> notifier = SimpleFeishuNotifier()
            >>> notifier.send_markdown_notification(
            ...     card_id="cli_xxxx",
            ...     markdown_text="## 编译结果\\n- **状态**: 成功",
            ...     group_name="版本监控群",
            ...     title="编译完成"
            ... )
        """
        # 1. 获取群组ID
        group_id = self.larker.get_chat_id_by_name(group_name)
        if not group_id:
            logging.error(f"Cannot find group: {group_name}")
            return False

        # 2. 构建变量字典
        variables = {"content": markdown_text}
        if title:
            variables["title"] = title
        variables.update(extra_variables)

        # 3. 构建卡片消息
        message = {
            "type": "template",
            "data": {
                "template_id": card_id,
                "template_variable": variables,
            },
        }

        # 4. 发送（带重试）
        return self._send_with_retry(group_id, message, retry_time=3)

    def send_text_notification(
        self,
        message: str,
        group_name: str,
        retry_time: int = 3,
    ) -> bool:
        """
        发送简单文本通知

        Args:
            message: 文本消息内容
            group_name: 接收通知的群组名称
            retry_time: 发送失败重试次数

        Returns:
            bool: 发送是否成功
        """
        group_id = self.larker.get_chat_id_by_name(group_name)
        if not group_id:
            logging.error(f"Cannot find group: {group_name}")
            return False

        for attempt in range(retry_time):
            try:
                self.larker.send_message_simple(
                    receive_id_type="chat_id",
                    receive_id=group_id,
                    message=message,
                )
                return True
            except Exception as e:
                if attempt == retry_time - 1:
                    logging.error(f"Failed to send message: {e}")
                else:
                    logging.warning(f"Retry {attempt + 1}/{retry_time}: {e}")

        return False

    def _send_with_retry(self, group_id: str, message: dict, retry_time: int) -> bool:
        """
        带重试的卡片发送逻辑

        Args:
            group_id: 群组ID
            message: 消息内容
            retry_time: 重试次数

        Returns:
            bool: 发送是否成功
        """
        unique_id = str(uuid.uuid4())
        for attempt in range(retry_time):
            try:
                self.larker.send_message_card(
                    receive_id_type="chat_id",
                    receive_id=group_id,
                    message=message,
                    uuid=unique_id,
                )
                return True
            except Exception as e:
                if attempt == retry_time - 1:
                    logging.error(f"Failed to send card: {e}")
                else:
                    logging.warning(f"Retry {attempt + 1}/{retry_time}: {e}")

        return False

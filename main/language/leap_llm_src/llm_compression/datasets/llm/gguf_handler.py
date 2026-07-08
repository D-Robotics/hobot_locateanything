"""GGUF Handler for BFCL evaluation via llama-server OpenAI-compatible API.

Replaces LLMCompressionHandler for GGUF models, using HTTP requests
to llama-server's /v1/chat/completions endpoint.
"""

import copy
import re
import time
from types import SimpleNamespace
from typing import Any

import requests
from bfcl_eval.model_handler.base_handler import BaseHandler
from bfcl_eval.model_handler.utils import (
    default_decode_ast_prompting,
    default_decode_execute_prompting,
    system_prompt_pre_processing_chat_model,
)
from bfcl_eval.utils import contain_multi_turn_interaction


class GGUFHandler(BaseHandler):
    """BFCL Handler using llama-server's OpenAI-compatible API."""

    # Gemma4 tool call tags: <|tool_call>call:func_name(args)<tool_call|>
    _TOOL_CALL_RE = re.compile(r"<\|tool_call>call:(.+?)<tool_call\|>")

    def __init__(self, server_url: str, model_name: str, temperature: float = 0.001, max_tokens: int = 512):
        super().__init__(model_name, temperature, model_name, is_fc_model=False)
        self.server_url = server_url.rstrip("/")
        self.max_tokens = max_tokens

    def _strip_tool_call_tags(self, text):
        """Convert Gemma4 native tool call format to BFCL expected format.

        Gemma4: <|tool_call>call:func(args)<tool_call|>
        BFCL:   [func(args)]
        """
        calls = self._TOOL_CALL_RE.findall(text)
        if calls:
            return "[" + ",".join(calls) + "]"
        return text

    def inference(self, test_entry: dict, include_input_log: bool, exclude_state_log: bool = True):
        if contain_multi_turn_interaction(test_entry["id"]):
            return self.inference_multi_turn_prompting(test_entry, include_input_log, exclude_state_log)
        return self.inference_single_turn_prompting(test_entry, include_input_log)

    def decode_ast(self, result, language, has_tool_call_tag):
        return default_decode_ast_prompting(result, language, has_tool_call_tag)

    def decode_execute(self, result, has_tool_call_tag):
        return default_decode_execute_prompting(result, has_tool_call_tag)

    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions = test_entry["function"]
        test_entry["question"][0] = system_prompt_pre_processing_chat_model(
            copy.deepcopy(test_entry["question"][0]), functions, test_entry["id"]
        )
        return {"message": [], "function": functions}

    def _query_prompting(self, inference_data: dict):
        start_time = time.time()
        payload = {
            "messages": inference_data["message"],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        resp = requests.post(
            f"{self.server_url}/v1/chat/completions",
            json=payload,
            timeout=120,
        )
        if resp.status_code != 200:
            import json

            print(f"[GGUF] Request failed ({resp.status_code}): {resp.text[:500]}")
            print(f"[GGUF] Payload messages: {json.dumps(payload['messages'][:3], ensure_ascii=False)[:1000]}")
        resp.raise_for_status()
        data = resp.json()
        latency = time.time() - start_time

        msg = data["choices"][0]["message"]
        # llama-server's enable_thinking=false does not truly disable thinking for some models
        # (e.g. Qwen3). The actual output goes to reasoning_content while content is empty.
        # Fallback to reasoning_content to avoid silent information loss.
        output_text = msg["content"] or msg.get("reasoning_content", "")

        usage = data.get("usage", {})
        response = SimpleNamespace(
            choices=[SimpleNamespace(text=output_text)],
            usage=SimpleNamespace(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
            ),
        )
        return response, latency

    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        raw = api_response.choices[0].text
        return {
            "model_responses": self._strip_tool_call_tags(raw),
            "reasoning_content": "",
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

    def add_first_turn_message_prompting(self, inference_data: dict, first_turn_message: list) -> dict:
        inference_data["message"].extend(first_turn_message)
        return inference_data

    def _add_next_turn_user_message_prompting(self, inference_data: dict, user_message: list) -> dict:
        inference_data["message"].extend(user_message)
        return inference_data

    def _add_assistant_message_prompting(self, inference_data: dict, model_response_data: dict) -> dict:
        inference_data["message"].append({"role": "assistant", "content": model_response_data["model_responses"]})
        return inference_data

    def _add_execution_results_prompting(
        self, inference_data: dict, execution_results: list, model_response_data: dict
    ) -> dict:
        # llama-server does not support "tool" role, convert to "user" role
        for execution_result, decoded_model_response in zip(
            execution_results, model_response_data.get("model_responses_decoded", [])
        ):
            inference_data["message"].append(
                {"role": "user", "content": f"[Function result of {decoded_model_response}]: {execution_result}"}
            )
        return inference_data

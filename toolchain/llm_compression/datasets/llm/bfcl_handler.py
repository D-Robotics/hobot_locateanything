"""BFCL Handler for llm_compression framework.

Internal module used by BFCLDataset.
Inherits BFCL BaseHandler to reuse complete multi-turn inference pipeline.
"""

import copy
import time
from types import SimpleNamespace
from typing import Any

import torch
from bfcl_eval.model_handler.base_handler import BaseHandler
from bfcl_eval.model_handler.utils import (
    default_decode_ast_prompting,
    default_decode_execute_prompting,
    system_prompt_pre_processing_chat_model,
)
from bfcl_eval.utils import contain_multi_turn_interaction


class LLMCompressionHandler:
    """BFCL Handler implementation for llm_compression framework."""

    def __init__(self, model, q_model, model_name: str, temperature: float, dtype: torch.dtype, do_sample: bool):
        self._model = model
        self._q_model = q_model
        self._model_name = model_name
        self._temperature = temperature
        self._dtype = dtype
        self._do_sample = do_sample

        # Create handler instance, inheriting from BFCL BaseHandler
        # Note: @override decorator doesn't work with dynamically created classes
        class _Handler(BaseHandler):
            def __init__(h_self, model, q_model, model_name, temperature, dtype, do_sample):
                super().__init__(model_name, temperature, model_name, is_fc_model=False)
                h_self.llmc_model = model
                h_self.q_model = q_model
                h_self.dtype = dtype
                h_self.do_sample = do_sample

            def inference(h_self, test_entry: dict, include_input_log: bool, exclude_state_log: bool = True):
                if contain_multi_turn_interaction(test_entry["id"]):
                    return h_self.inference_multi_turn_prompting(test_entry, include_input_log, exclude_state_log)
                else:
                    return h_self.inference_single_turn_prompting(test_entry, include_input_log)

            def decode_ast(h_self, result, language, has_tool_call_tag):
                return default_decode_ast_prompting(result, language, has_tool_call_tag)

            def decode_execute(h_self, result, has_tool_call_tag):
                return default_decode_execute_prompting(result, has_tool_call_tag)

            def _pre_query_processing_prompting(h_self, test_entry: dict) -> dict:
                functions = test_entry["function"]
                test_entry_id = test_entry["id"]
                test_entry["question"][0] = system_prompt_pre_processing_chat_model(
                    copy.deepcopy(test_entry["question"][0]), functions, test_entry_id
                )
                return {"message": [], "function": functions}

            def _query_prompting(h_self, inference_data: dict):
                messages = inference_data["message"]
                inputs = h_self.q_model.input_preprocess(messages)

                start_time = time.time()
                generated_ids = h_self.llmc_model.generate(inputs, do_sample=h_self.do_sample)
                latency = time.time() - start_time

                output_text = h_self.q_model.output_postprocess(generated_ids)

                prompt_tokens = (
                    inputs["input_ids"].shape[-1] if hasattr(inputs, "__getitem__") and "input_ids" in inputs else 0
                )
                completion_tokens = getattr(generated_ids, "prompt_length", 0)
                if completion_tokens > 0:
                    completion_tokens = generated_ids.shape[-1]
                else:
                    completion_tokens = max(0, generated_ids.shape[-1] - prompt_tokens)

                response = SimpleNamespace(
                    choices=[SimpleNamespace(text=output_text)],
                    usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
                )
                return response, latency

            def _parse_query_response_prompting(h_self, api_response: Any) -> dict:
                model_response = api_response.choices[0].text
                return {
                    "model_responses": model_response,
                    "reasoning_content": "",
                    "input_token": api_response.usage.prompt_tokens,
                    "output_token": api_response.usage.completion_tokens,
                }

            def add_first_turn_message_prompting(h_self, inference_data: dict, first_turn_message: list) -> dict:
                inference_data["message"].extend(first_turn_message)
                return inference_data

            def _add_next_turn_user_message_prompting(h_self, inference_data: dict, user_message: list) -> dict:
                inference_data["message"].extend(user_message)
                return inference_data

            def _add_assistant_message_prompting(h_self, inference_data: dict, model_response_data: dict) -> dict:
                inference_data["message"].append(
                    {"role": "assistant", "content": model_response_data["model_responses"]}
                )
                return inference_data

            def _add_execution_results_prompting(
                h_self, inference_data: dict, execution_results: list, model_response_data: dict
            ) -> dict:
                for execution_result, decoded_model_response in zip(
                    execution_results, model_response_data.get("model_responses_decoded", [])
                ):
                    inference_data["message"].append(
                        {
                            "role": "tool",
                            "name": decoded_model_response,
                            "content": execution_result,
                        }
                    )
                return inference_data

        self._handler = _Handler(
            self._model, self._q_model, self._model_name, self._temperature, self._dtype, self._do_sample
        )

    def inference(self, test_entry: dict, include_input_log: bool, exclude_state_log: bool = True):
        """Dispatch to single-turn or multi-turn inference."""
        return self._handler.inference(test_entry, include_input_log, exclude_state_log)

    def decode_ast(self, result, language, has_tool_call_tag):
        """Decode result to AST format."""
        return self._handler.decode_ast(result, language, has_tool_call_tag)

    def decode_execute(self, result, has_tool_call_tag):
        """Decode result to execute format."""
        return self._handler.decode_execute(result, has_tool_call_tag)

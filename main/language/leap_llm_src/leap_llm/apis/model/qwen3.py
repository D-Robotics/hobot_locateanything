import gc
import os
from pathlib import Path

import torch

from leap_llm.apis.calibration.calibration import CalibrationDataPreparer
from leap_llm.apis.calibration.data_loader import load_message_data
from leap_llm.models.eagle3.model import Eagle3Draft
from leap_llm.models.qwen3.model import Qwen3
from leap_llm.nn.utils import standard_lm_name

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Qwen3Api:
    def __init__(
        self,
        input_model_path: str,
        output_model_path: str,
        calib_message_path: str = None,
        chunk_size: int = 512,
        cache_len: int = 4096,
        devices: list[str] = None,
        device: str = None,  # For backward compatibility
        dtype: str = "float32",
        model_type: str = "qwen3",
        w_bits: int = 8,
        mask_value: int = -32768,
        prefill_core_num: list[int] = None,
        decode_core_num: list[int] = None,
        march: str = "nash-p",
        # EAGLE3 speculative decoding parameters (all optional)
        speculative_algorithm: str = None,
        speculative_draft_model_path: str = None,
        speculative_num_steps: int = 7,
        speculative_eagle_topk: int = 8,
        speculative_num_draft_tokens: int = 32,
        decode_seq_len: int = None,
    ):
        if prefill_core_num is None:
            prefill_core_num = [1]
        if decode_core_num is None:
            decode_core_num = [1]
        self.input_model_path = input_model_path
        self.calib_msg_data = load_message_data(calib_message_path, model_type)
        self.chunk_size = chunk_size
        self.cache_len = cache_len
        # Support both devices (list) and device (single string) for backward compatibility
        if devices is not None:
            self.devices = devices if isinstance(devices, list) else [devices]
        elif device is not None:
            self.devices = [device] if isinstance(device, str) else device
        else:
            self.devices = ["cpu"]
        self.primary_device = self.devices[0]
        self.dtype = dtype
        self.w_bits = w_bits
        self.mask_value = mask_value
        self.model_type = model_type
        self.prefill_core_num = prefill_core_num
        self.decode_core_num = decode_core_num
        self.march = march

        # EAGLE3 speculative decoding config
        self.speculative_algorithm = speculative_algorithm
        self.speculative_draft_model_path = speculative_draft_model_path
        self.speculative_num_steps = speculative_num_steps
        self.speculative_eagle_topk = speculative_eagle_topk
        self.speculative_num_draft_tokens = speculative_num_draft_tokens
        self.decode_seq_len = decode_seq_len

        os.makedirs(output_model_path, exist_ok=True)
        self.output_model_path = standard_lm_name(
            input_model_path,
            output_model_path,
            chunk_size,
            cache_len,
            w_bits,
            march,
            prefill_core_num,
            decode_core_num,
        )

        self.draft_output_model_path = None
        if speculative_draft_model_path:
            draft_w_bits = 8
            self.draft_output_model_path = standard_lm_name(
                speculative_draft_model_path,
                output_model_path,
                chunk_size,
                cache_len,
                draft_w_bits,
                march,
                prefill_core_num,
                decode_core_num,
            )

        is_eagle3 = (
            speculative_algorithm is not None
            and speculative_algorithm.upper() == "EAGLE3"
        )
        self.qwen3_model = Qwen3.load_model(
            input_model_path,
            chunk_size=chunk_size,
            cache_len=cache_len,
            w_bits=w_bits,
            enable_eagle3=is_eagle3,
            num_draft_tokens=speculative_num_draft_tokens,
            decode_seq_len=decode_seq_len,
        )
        
        # Setup multi-GPU if needed
        self._setup_multi_gpu()

    def _setup_multi_gpu(self):
        """Setup multi-GPU model distribution"""
        primary_device = self.primary_device
        
        if len(self.devices) > 1 and primary_device != "cpu":
            # Multi-GPU mode
            num_layers = self.qwen3_model.config.num_hidden_layers
            num_devices = len(self.devices)
            layers_per_device = num_layers // num_devices
            remainder = num_layers % num_devices

            # Store multi-GPU info
            self.qwen3_model.model._layer_to_device = {}
            layer_idx = 0
            for device_idx, device_name in enumerate(self.devices):
                start_layer = layer_idx
                end_layer = (
                    layer_idx + layers_per_device + (1 if device_idx < remainder else 0)
                )

                # Move corresponding layers to current device
                for i in range(start_layer, end_layer):
                    self.qwen3_model.model.layers[i] = self.qwen3_model.model.layers[i].to(device=device_name)
                    self.qwen3_model.model._layer_to_device[i] = device_name

                layer_idx = end_layer

            # Move other components to first device
            self.qwen3_model.model.embed_tokens = self.qwen3_model.model.embed_tokens.to(device=primary_device)
            self.qwen3_model.model.norm = self.qwen3_model.model.norm.to(device=primary_device)
            self.qwen3_model.model.lm_head = self.qwen3_model.model.lm_head.to(device=primary_device)
            self.qwen3_model.model.cos = self.qwen3_model.model.cos.to(device=primary_device)
            self.qwen3_model.model.sin = self.qwen3_model.model.sin.to(device=primary_device)

            # Store multi-device info
            self.qwen3_model.model._multi_gpu_devices = self.devices

            print(
                f"Multi-device setup: {num_layers} layers distributed across {num_devices} devices"
            )
            layer_idx = 0
            for device_idx, device_name in enumerate(self.devices):
                start_layer = layer_idx
                end_layer = (
                    layer_idx + layers_per_device + (1 if device_idx < remainder else 0)
                )
                print(
                    f"  Device {device_name}: layers {start_layer}-{end_layer-1} ({end_layer-start_layer} layers)"
                )
                layer_idx = end_layer
        else:
            # Single device mode (CPU or single GPU)
            self.qwen3_model.model._multi_gpu_devices = None

    def _forward_multi_gpu(self, tokens, position_ids, attention_mask, caches):
        """Multi-device forward function"""
        if (
            not hasattr(self.qwen3_model.model, "_multi_gpu_devices")
            or self.qwen3_model.model._multi_gpu_devices is None
        ):
            # Single device mode, use original forward
            return self.qwen3_model.model.forward(
                tokens=tokens,
                position_ids=position_ids,
                attention_mask=attention_mask,
                caches=caches,
            )

        # Multi-device mode
        devices = self.qwen3_model.model._multi_gpu_devices
        primary_device = devices[0]

        # Ensure inputs are on first device
        tokens = tokens.to(device=primary_device)
        position_ids = position_ids.to(device=primary_device)
        attention_mask = attention_mask.to(device=primary_device)

        # Forward pass, transfer hidden_states between devices
        hidden_states = self.qwen3_model.model.embed_tokens(tokens)

        new_keys = []
        new_values = []
        # Prepare position embeddings (compute on first device)
        self.qwen3_model.model.cos = self.qwen3_model.model.cos.to(position_ids.device).to(hidden_states.dtype)
        self.qwen3_model.model.sin = self.qwen3_model.model.sin.to(position_ids.device).to(hidden_states.dtype)

        position_ids_expanded = position_ids.unsqueeze(-1).expand(
            -1, -1, self.qwen3_model.model.cos.size(-1)
        )
        cos = torch.gather(self.qwen3_model.model.cos, 1, position_ids_expanded)
        sin = torch.gather(self.qwen3_model.model.sin, 1, position_ids_expanded)

        position_embeddings = (cos, sin)

        # Split caches
        cache_keys = caches[: len(caches) // 2]
        cache_values = caches[len(caches) // 2 :]

        # Move caches to corresponding devices (if caches are not empty)
        if len(cache_keys) > 0 and hasattr(self.qwen3_model.model, "_layer_to_device"):
            for layer_idx in range(len(self.qwen3_model.model.layers)):
                target_device = self.qwen3_model.model._layer_to_device[layer_idx]
                if layer_idx < len(cache_keys):
                    cache_keys[layer_idx] = cache_keys[layer_idx].to(device=target_device)
                if layer_idx < len(cache_values):
                    cache_values[layer_idx] = cache_values[layer_idx].to(
                        device=target_device
                    )

        all_hidden_states = []
        num_layers = self.qwen3_model.config.num_hidden_layers
        enable_eagle3 = self.qwen3_model.config.enable_eagle3

        for layer_idx, decoder_layer in enumerate(self.qwen3_model.model.layers):
            target_device = self.qwen3_model.model._layer_to_device[layer_idx]
            # Move hidden_states to current layer's device
            hidden_states = hidden_states.to(device=target_device)
            position_embeddings_gpu = (
                position_embeddings[0].to(device=target_device),
                position_embeddings[1].to(device=target_device),
            )
            attention_mask_gpu = attention_mask.to(device=target_device)

            if enable_eagle3 and (
                layer_idx == 2
                or layer_idx == num_layers // 2
                or layer_idx == num_layers - 3
            ):
                all_hidden_states.append(hidden_states.to(device=primary_device))

            hidden_states, new_key, new_value = decoder_layer(
                hidden_states,
                attention_mask=attention_mask_gpu,
                position_embeddings=position_embeddings_gpu,
                cache_keys=cache_keys[layer_idx] if len(cache_keys) else None,
                cache_values=cache_values[layer_idx] if len(cache_values) else None,
            )
            new_keys.append(new_key)
            new_values.append(new_value)

        # Move hidden_states back to first device for norm and lm_head
        hidden_states = hidden_states.to(device=primary_device)
        hidden_states = self.qwen3_model.model.norm(hidden_states)
        token_logits = self.qwen3_model.model.lm_head(hidden_states)

        if enable_eagle3:
            fused_hidden_states = torch.cat(all_hidden_states, dim=-1)
            return token_logits, *new_keys, *new_values, fused_hidden_states

        return token_logits, *new_keys, *new_values

    def compile(self, vit_kwargs=None, llm_kwargs=None):
        if self.speculative_algorithm == "EAGLE3":
            self._compile_eagle3(llm_kwargs)
            return

        self._compile_standard(llm_kwargs)

    def _run_calibration(self):
        """Run calibration forward pass through the base model.

        Returns:
            None for standard mode. When enable_eagle3=True, returns a dict:
            {
                "fused_hidden_states_chunks": list of [bs, chunk_size, hidden_size*3],
                "input_ids_chunks": list of [bs, chunk_size],
                "position_ids_chunks": list of [bs, chunk_size],
                "attention_mask_chunks": list of [bs, chunk_size, context_len],
            }
        """
        device = self.primary_device if torch.cuda.is_available() and self.primary_device.startswith("cuda") else "cpu"
        dtype = torch.float32
        enable_eagle3 = self.qwen3_model.config.enable_eagle3
        num_hidden_layers = self.qwen3_model.config.num_hidden_layers

        if len(self.devices) > 1 and self.primary_device != "cpu":
            pass
        else:
            self.qwen3_model.model.to(device, dtype=dtype)

        self.qwen3_model.model.compile_mode(False)

        preparer = CalibrationDataPreparer(
            model_type="qwen3",
            model_dir=self.input_model_path,
            seq_len=self.chunk_size,
            kv_cache_len=self.cache_len,
            transpose_cache=True,
            device=self.primary_device,
            mask_value=self.mask_value,
            pos_mask_value=1,
            data_type=dtype,
            padding_side="left",
        )
        preparer.tokenizer.padding_side = "left"

        eagle3_calib_data = None
        if enable_eagle3:
            eagle3_calib_data = {
                "fused_hidden_states_chunks": [],
                "input_ids_chunks": [],
                "position_ids_chunks": [],
                "attention_mask_chunks": [],
            }

        for prompt in self.calib_msg_data:
            prompt = preparer.tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
                # enable_thinking=True,
                enable_thinking=False,
            )
            (
                input_chunks,
                causal_mask_chunks,
                position_ids_chunks,
                pask_key_value_list,
            ) = preparer.prepare_inputs(prompt)

            pask_key_value_list = [cache.unsqueeze(0) for cache in pask_key_value_list]

            has_layer_map = hasattr(self.qwen3_model.model, "_layer_to_device")
            if len(self.devices) > 1 and self.primary_device != "cpu" and has_layer_map:
                num_hidden = self.qwen3_model.config.num_hidden_layers
                for layer_idx in range(num_hidden):
                    target_device = self.qwen3_model.model._layer_to_device[layer_idx]
                    pask_key_value_list[layer_idx] = (
                        pask_key_value_list[layer_idx].to(device=target_device)
                    )
                    pask_key_value_list[layer_idx + num_hidden] = (
                        pask_key_value_list[layer_idx + num_hidden].to(device=target_device)
                    )

            for _i, (input_ids, attn_mask, position_ids) in enumerate(
                zip(input_chunks, causal_mask_chunks, position_ids_chunks)
            ):
                with torch.no_grad():
                    attn_mask = attn_mask.unsqueeze(0)
                    position_ids = position_ids.unsqueeze(0)

                    if len(self.devices) > 1 and self.primary_device != "cpu":
                        outputs = self._forward_multi_gpu(
                            input_ids.to(self.primary_device),
                            position_ids.to(self.primary_device),
                            attn_mask.to(self.primary_device),
                            pask_key_value_list,
                        )
                    else:
                        outputs = self.qwen3_model.model.forward(
                            input_ids.to(device),
                            position_ids.to(device),
                            attn_mask.to(device),
                            pask_key_value_list,
                        )

                if enable_eagle3:
                    fused_hs = outputs[num_hidden_layers * 2 + 1]
                    eagle3_calib_data["fused_hidden_states_chunks"].append(fused_hs.cpu())
                    eagle3_calib_data["input_ids_chunks"].append(input_ids.cpu())
                    eagle3_calib_data["position_ids_chunks"].append(position_ids.cpu())
                    eagle3_calib_data["attention_mask_chunks"].append(attn_mask.cpu())

                for z in range(0, num_hidden_layers * 2):
                    new_cache = outputs[z + 1]
                    past = pask_key_value_list[z]
                    slice_past = past[:, self.chunk_size :]
                    target_device = past.device
                    new_cache_on_device = new_cache.to(device=target_device)
                    pask_key_value_list[z] = torch.concat(
                        [slice_past, new_cache_on_device], dim=1
                    )

        len(outputs)
        logits = outputs[0]
        logits_slice = logits[:, -1, :]
        top_token_id = torch.argmax(logits_slice, dim=-1).item()
        logits_slice.flatten()[top_token_id]

        preparer.tokenizer.decode(
            top_token_id,
            skip_special_tokens=True,
        )

        if enable_eagle3 and eagle3_calib_data:
            all_input_ids = torch.cat(
                eagle3_calib_data["input_ids_chunks"], dim=-1
            )
            pad_token_id = preparer.tokenizer.pad_token_id
            non_pad_mask = (all_input_ids[0] != pad_token_id)
            first_valid = non_pad_mask.nonzero(as_tuple=True)[0][0].item()

            top_token_tensor = torch.tensor(
                [top_token_id], dtype=all_input_ids.dtype, device=all_input_ids.device
            )
            shifted_valid = torch.cat([all_input_ids[0, first_valid + 1:], top_token_tensor])
            all_input_ids[0, first_valid:] = shifted_valid

            eagle3_calib_data["input_ids_chunks"] = list(
                all_input_ids.split(self.chunk_size, dim=-1)
            )
            print(
                f"[EAGLE3 Calibration] input_ids shifted: "
                f"pad_token_id={pad_token_id}, first_valid_pos={first_valid}, "
                f"appended top_token_id={top_token_id}, "
                f"re-chunked into {len(eagle3_calib_data['input_ids_chunks'])} chunk(s)"
            )

        self.qwen3_model.model.compile_mode(True)
        self.qwen3_model.model.to("cpu", dtype=torch.float16)

        del pask_key_value_list
        del outputs
        del preparer
        del input_chunks, causal_mask_chunks, position_ids_chunks
        gc.collect()
        if device != "cpu":
            for dev in self.devices:
                if dev != "cpu" and dev.startswith("cuda"):
                    with torch.cuda.device(dev):
                        torch.cuda.empty_cache()
            print(f"[GPU Memory] Released CUDA cache on {self.devices} after calibration.")

        return eagle3_calib_data

    def _compile_standard(self, llm_kwargs):
        """Standard Qwen3 compilation flow (original behavior)."""
        self._run_calibration()
        self.qwen3_model.compile(
            stage="all",
            output_model_path=self.output_model_path,
            prefill_core_num=self.prefill_core_num,
            decode_core_num=self.decode_core_num,
            enable_vpu=True,
            **llm_kwargs,
        )

    def _compile_eagle3(self, llm_kwargs):
        """EAGLE3 speculative decoding compilation flow.

        Compiles both the target model (Qwen3 base with hidden states output)
        and the draft model (EAGLE3 lightweight prediction network).
        """
        print("[EAGLE3] Starting EAGLE3 compilation...")
        print(f"[EAGLE3]   draft_model_path: {self.speculative_draft_model_path}")
        print(f"[EAGLE3]   num_steps: {self.speculative_num_steps}")
        print(f"[EAGLE3]   eagle_topk: {self.speculative_eagle_topk}")
        print(f"[EAGLE3]   num_draft_tokens: {self.speculative_num_draft_tokens}")

        draft_w_bits = 8
        print(f"[EAGLE3] Draft model using w_bits={draft_w_bits} (hardcoded for small parameter count)")
        eagle3_draft = Eagle3Draft.load_model(
            draft_model_path=self.speculative_draft_model_path,
            target_model_path=self.input_model_path,
            chunk_size=self.chunk_size,
            cache_len=self.cache_len,
            eagle_topk=self.speculative_eagle_topk,
            speculative_num_steps=self.speculative_num_steps,
            w_bits=draft_w_bits,
            torch_dtype=torch.float32,
        )
        d2t_int32 = eagle3_draft.model.d2t.to(torch.int32).cpu().numpy()
        d2t_int32_path = os.path.join(str(Path(self.output_model_path).parent), "d2t.bin")
        d2t_int32.tofile(d2t_int32_path)
        print(f"[EAGLE3 Draft model] d2t saved to {d2t_int32_path}")

        eagle3_calib_data = self._run_calibration()
        self._run_draft_prefill(eagle3_draft, eagle3_calib_data)

        draft_output = self._compile_eagle3_draft(llm_kwargs, eagle3_draft)
        print(f"[EAGLE3]   draft model:  {draft_output}")

        self.qwen3_model.compile(
            stage="all",
            output_model_path=self.output_model_path,
            prefill_core_num=self.prefill_core_num,
            decode_core_num=self.decode_core_num,
            enable_vpu=True,
            **llm_kwargs,
        )

        print(f"[EAGLE3]   target model: {self.output_model_path}")
        print("[EAGLE3] Compilation finished.")

    def _run_draft_prefill(self, eagle3_draft: Eagle3Draft, eagle3_calib_data: dict):
        """Run draft model chunk prefill using fused_hidden_states from base model.

        Draft model is small, always runs on a single device (primary_device).
        """
        device = self.primary_device
        print(f"[EAGLE3 Draft Prefill] Using single device: {device}")
        draft_config = eagle3_draft.config

        eagle3_draft.model.to(device, dtype=torch.float32)
        eagle3_draft.model.compile_mode(False)
        eagle3_draft.model.eval()

        cache_key = torch.zeros(
            1, self.cache_len, draft_config.num_key_value_heads, draft_config.head_dim,
            dtype=torch.float32, device=device,
        )
        cache_value = torch.zeros_like(cache_key)
        draft_caches = [cache_key, cache_value]

        num_chunks = len(eagle3_calib_data["fused_hidden_states_chunks"])
        print(f"[EAGLE3 Draft Prefill] Running {num_chunks} chunk(s)...")

        for i in range(num_chunks):
            input_ids = eagle3_calib_data["input_ids_chunks"][i].to(device)
            position_ids = eagle3_calib_data["position_ids_chunks"][i].to(device)
            attention_mask = eagle3_calib_data["attention_mask_chunks"][i].to(device)
            fused_hs = eagle3_calib_data["fused_hidden_states_chunks"][i].to(device)

            with torch.no_grad():
                logits, new_key, new_value, output_hidden = eagle3_draft.model.forward(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    caches=draft_caches,
                    hidden_states=fused_hs,
                )

            draft_caches[0] = torch.cat(
                [draft_caches[0][:, self.chunk_size:], new_key], dim=1
            )
            draft_caches[1] = torch.cat(
                [draft_caches[1][:, self.chunk_size:], new_value], dim=1
            )

        last_hidden = output_hidden[:, -1]
        last_headout = logits[:, -1]
        top_k = self.speculative_eagle_topk
        last_p = torch.nn.functional.log_softmax(last_headout, dim=-1)
        top = torch.topk(last_p, top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values
        topk_p[0]

        d2t = eagle3_draft.model.d2t.to(device)
        if eagle3_draft.config.vocab_size == eagle3_draft.config.draft_vocab_size:
            input_ids = topk_index
        else:
            topk_index + d2t[topk_index]
            input_ids = topk_index + d2t[topk_index]

        last_hidden.unsqueeze(1).repeat(1, top_k, 1)

        eagle3_draft.model.compile_mode(True)
        eagle3_draft.model.to("cpu", dtype=torch.float16)

        del draft_caches, eagle3_calib_data
        gc.collect()
        if device != "cpu" and device.startswith("cuda"):
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
            print(f"[EAGLE3 Draft Prefill] Released CUDA cache on {device}.")

    def _compile_eagle3_draft(self, llm_kwargs, eagle3_draft: Eagle3Draft) -> str:
        """Compile the draft model (EAGLE3 lightweight prediction network).

        The draft model is already in compile_mode(True) after _run_draft_prefill().
        Prefill uses chunk_size, decode uses eagle_topk as seq_len (configured
        via Eagle3DraftConfig.prefill_seq_len / decode_seq_len at load time).

        Returns:
            Path to the compiled draft model.
        """
        draft_output_path = self.draft_output_model_path

        print(f"[EAGLE3 Draft Compile] output_path: {draft_output_path}")
        print(
            f"[EAGLE3 Draft Compile] prefill_seq_len={eagle3_draft.config.prefill_seq_len}, "
            f"decode_seq_len={eagle3_draft.config.decode_seq_len}, "
            f"context_len={eagle3_draft.config.context_len}"
        )

        result = eagle3_draft.compile(
            stage="all",
            output_model_path=draft_output_path,
            prefill_core_num=self.prefill_core_num,
            decode_core_num=self.decode_core_num,
            enable_vpu=True,
            **llm_kwargs,
        )

        print(f"[EAGLE3 Draft Compile] Done: {result}")
        return draft_output_path

    def get_quant_path(self) -> tuple[str, ...]:
        """Return compiled BC path(s).

        For EAGLE3 mode, returns (target_bc_path, draft_bc_path).
        For standard mode, returns (bc_path, None).
        """
        if self.speculative_algorithm == "EAGLE3":
            target_bc = str(Path(self.output_model_path).with_suffix(".prefill_convert.bc"))
            draft_bc = str(Path(self.draft_output_model_path).with_suffix(".prefill_convert.bc"))
            return target_bc, draft_bc

        return (
            str(Path(self.output_model_path).with_suffix(".prefill_convert.bc")),
            None,
        )

    def get_hbm_path(self) -> tuple[str, ...]:
        """Return compiled HBM path(s).

        For EAGLE3 mode, returns (target_hbm_path, draft_hbm_path).
        For standard mode, returns (hbm_path, None).
        """
        if self.speculative_algorithm == "EAGLE3":
            target_hbm = self.output_model_path
            draft_hbm = self.draft_output_model_path
            return target_hbm, draft_hbm

        return self.output_model_path, None

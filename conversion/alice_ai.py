from __future__ import annotations

from typing import Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


@ModelBase.register("AliceAIForCausalLM")
class AliceAIModel(TextModel):
    model_arch = gguf.MODEL_ARCH.ALICE_AI

    def set_vocab(self):
        self._set_vocab_sentencepiece()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])
        self.gguf_writer.add_expert_count(hparams["num_experts"])
        self.gguf_writer.add_expert_feed_forward_length(hparams["moe_intermediate_size"])
        self.gguf_writer.add_expert_shared_count(hparams.get("num_shared_experts", 0))
        self.gguf_writer.add_expert_weights_scale(1.0)
        self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
        self.gguf_writer.add_rope_dimension_count(
            int(
                self.hparams["head_dim"]
                * self.hparams.get("partial_rotary_factor", 0.25)
            )
        )
        self.gguf_writer.add_ssm_conv_kernel(
            hparams.get("linear_conv_kernel_dim", 4)
        )
        self.gguf_writer.add_kda_head_dim(hparams.get("linear_key_head_dim", 128))

        # KDA-specific: state_size = key_head_dim, group_count = num_key_heads,
        # inner_size = key_dim (num_k_heads * key_head_dim)
        self.gguf_writer.add_ssm_state_size(hparams.get("linear_key_head_dim", 128))
        self.gguf_writer.add_ssm_group_count(hparams.get("linear_num_key_heads", 1))
        self.gguf_writer.add_ssm_inner_size(
            hparams.get("linear_num_key_heads", 1) * hparams.get("linear_key_head_dim", 128)
        )

        # Attention residual block size (for KDA cross-layer attention)
        if "block_attn_res_block_size" in hparams:
            self.gguf_writer.add_attn_res_block_size(hparams["block_attn_res_block_size"])

        # Recurrent layer mask: true = linear_attention (KDA), false = full_attention
        if (layer_types := self.hparams.get("layer_types")) is not None:
            self.gguf_writer.add_recurrent_layers([t == "linear_attention" for t in layer_types])

    _experts: list[dict[str, Tensor]] | None = None

    # Zero-centered norms: add 1.0 to the weight
    ZERO_CENTERED = (
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
    )

    def is_zero_centered(self, name: str) -> bool:
        if name == "model.norm.weight":
            return True
        for suffix in self.ZERO_CENTERED:
            if name.endswith(suffix):
                return True
        return False

    # Plain norms: write as-is
    PLAIN_NORMS = (
        "linear_attn.o_norm.weight",
        "attn_res_norm_weight",
        "mlp_res_norm_weight",
        "res_norm_weight",
    )

    def is_plain_norm(self, name: str) -> bool:
        for marker in self.PLAIN_NORMS:
            if marker in name:
                return True
        return False

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        n_head = self.hparams["num_attention_heads"]
        n_kv_head = self.hparams["num_key_value_heads"]
        n_embd = self.hparams["hidden_size"]
        n_experts = self.hparams.get("num_experts", 0)

        # --- MoE experts: gate_up_proj [E, 2*inter, hidden] -> gate + up ---
        if name.endswith("mlp.experts.gate_up_proj.weight"):
            assert bid is not None
            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]
            self._experts[bid][name] = data_torch
            if len(self._experts[bid]) >= n_experts:
                gate = self._experts[bid].pop(name)
                inter = gate.shape[1] // 2
                gate_part = gate[:, :inter, :]
                up_part = gate[:, inter:, :]
                yield from super().modify_tensors(
                    gate_part,
                    self.format_tensor_name(gguf.MODEL_TENSOR.FFN_GATE_UP_EXP, bid),
                    bid,
                )
                yield from super().modify_tensors(
                    up_part,
                    self.format_tensor_name(gguf.MODEL_TENSOR.FFN_UP_EXP, bid),
                    bid,
                )
            return

        # --- MoE experts: down_proj [E, hidden, inter] -> down ---
        if name.endswith("mlp.experts.down_proj.weight"):
            assert bid is not None
            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]
            self._experts[bid][name] = data_torch
            if len(self._experts[bid]) >= n_experts:
                down = self._experts[bid].pop(name)
                yield from super().modify_tensors(
                    down,
                    self.format_tensor_name(gguf.MODEL_TENSOR.FFN_DOWN_EXP, bid),
                    bid,
                )
            return

        # --- Shared expert: gate_proj, up_proj, down_proj ---
        if name.endswith("mlp.shared_expert.gate_proj.weight"):
            yield from super().modify_tensors(
                data_torch,
                self.format_tensor_name(gguf.MODEL_TENSOR.FFN_GATE_SHEXP, bid),
                bid,
            )
            return

        if name.endswith("mlp.shared_expert.up_proj.weight"):
            yield from super().modify_tensors(
                data_torch,
                self.format_tensor_name(gguf.MODEL_TENSOR.FFN_UP_SHEXP, bid),
                bid,
            )
            return

        if name.endswith("mlp.shared_expert.down_proj.weight"):
            yield from super().modify_tensors(
                data_torch,
                self.format_tensor_name(gguf.MODEL_TENSOR.FFN_DOWN_SHEXP, bid),
                bid,
            )
            return

        # --- Shared expert gate: Linear hidden->1 -> FFN_GATE_INP_SHEXP ---
        if name.endswith("mlp.shared_expert_gate.weight"):
            yield from super().modify_tensors(
                data_torch,
                self.format_tensor_name(gguf.MODEL_TENSOR.FFN_GATE_INP_SHEXP, bid),
                bid,
            )
            return

        # --- e_score_correction_bias: 1-D buffer per layer -> FFN_EXP_PROBS_B ---
        if name.endswith("mlp.gate.e_score_correction.bias") or name.endswith("mlp.gate.e_score_correction_bias"):
            yield from super().modify_tensors(
                data_torch,
                self.format_tensor_name(gguf.MODEL_TENSOR.FFN_EXP_PROBS_B, bid),
                bid,
            )
            return

        # --- Zero-centered norms: add 1.0 ---
        if self.is_zero_centered(name):
            # model.norm.weight is the global output norm (bid=None)
            if name == "model.norm.weight":
                yield from super().modify_tensors(
                    data_torch + 1.0,
                    self.format_tensor_name(gguf.MODEL_TENSOR.OUTPUT_NORM, None),
                    None,
                )
                return
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_NORM, bid)
                   if "input_layernorm" in name else
                   self.format_tensor_name(gguf.MODEL_TENSOR.FFN_NORM, bid)
                   if "post_attention_layernorm" in name else
                   self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_Q_NORM, bid)
                   if "q_norm" in name else
                   self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_K_NORM, bid),
                   data_torch + 1.0)
            return

        # --- Plain norms: write as-is ---
        if self.is_plain_norm(name):
            # Map attn_res_norm_weight and mlp_res_norm_weight to their GGUF norms
            if "attn_res_norm_weight" in name:
                tensor_id = gguf.MODEL_TENSOR.ATTN_RES_NORM
            elif "mlp_res_norm_weight" in name:
                tensor_id = gguf.MODEL_TENSOR.FFN_RES_NORM
            elif "res_norm_weight" in name:
                # model.attnres_final.res_norm_weight
                tensor_id = gguf.MODEL_TENSOR.OUTPUT_RES_NORM
            else:
                # linear_attn.o_norm.weight
                tensor_id = gguf.MODEL_TENSOR.SSM_OUT_NORM
            yield from super().modify_tensors(
                data_torch,
                self.format_tensor_name(tensor_id, bid),
                bid,
            )
            return

        # --- res_proj: write separately (2-D [1, hidden]) ---
        if name.endswith("_res_proj.weight"):
            if "attnres_final.res_proj" in name:
                tensor_id = gguf.MODEL_TENSOR.OUTPUT_RES_SCORE
            elif "attn_res_proj" in name:
                tensor_id = gguf.MODEL_TENSOR.ATTN_RES_SCORE
            elif "mlp_res_proj" in name:
                tensor_id = gguf.MODEL_TENSOR.FFN_RES_SCORE
            else:
                yield from super().modify_tensors(data_torch, name, bid)
                return
            yield from super().modify_tensors(
                data_torch,
                self.format_tensor_name(tensor_id, bid),
                bid,
            )
            return

        # --- a_log_bias and dt_bias ---
        if name.endswith("a_log_bias"):
            yield from super().modify_tensors(
                data_torch,
                self.format_tensor_name(gguf.MODEL_TENSOR.SSM_A_LOG_BIAS, bid),
                bid,
            )
            return

        if name.endswith("dt_bias"):
            yield from super().modify_tensors(
                data_torch,
                self.format_tensor_name(gguf.MODEL_TENSOR.SSM_DT_BIAS, bid),
                bid,
            )
            return

        # --- ssm_conv1d_*: reshape 3D [dim, 1, kernel] -> 2D [dim, kernel] ---
        # to avoid being counted as expert tensors by the oracle
        if name.endswith(("q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight")):
            if data_torch.ndim == 3 and data_torch.shape[1] == 1:
                data_torch = data_torch.squeeze(1)
            yield from super().modify_tensors(
                data_torch,
                self.map_tensor_name(name),
                bid,
            )
            return

        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._experts is not None:
            leftover = [k for d in self._experts for k in d.keys()]
            if leftover:
                raise ValueError(f"Unprocessed experts: {leftover}")

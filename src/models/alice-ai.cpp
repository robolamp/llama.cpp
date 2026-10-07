#include "models.h"

#include <algorithm>
#include "llama-memory-recurrent.h"
#include "llama-memory-hybrid.h"

//
// AliceAI: hybrid KDA (linear) + full-attention layers with MoE.
// Layer pattern: every 4th layer (layer_idx % 4 == 3) is full attention.
// Closest reference: kimi-linear (KDA math) + kimi-k3 (residual mixing + MoE).
//

void llama_model_alice_ai::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,             hparams.ssm_d_conv);
    ml.get_key(LLM_KV_KDA_HEAD_DIM,                hparams.n_embd_head_kda);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,             hparams.ssm_n_group);

    // Determine which layers are recurrent (KDA) vs full-attention.
    // Try reading recurrent_layers from GGUF metadata first (for reduced oracle models).
    // Fall back to pattern: every 4th layer (layer_idx % 4 == 3) is full attention, rest are KDA.
    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
        // For reduced oracle models, detect layer type by checking tensor presence.
        // A layer is recurrent if it has SSM tensors AND no attention tensors.
        // If it has attention tensors (attn_q), it goes in the KV cache even if it also has SSM.
        for (uint32_t i = 0; i < hparams.n_layer(); ++i) {
            std::string attn_name = "blk." + std::to_string(i) + ".attn_q.weight";
            bool has_attn = ml.get_tensor_meta(attn_name.c_str()) != nullptr;
            hparams.is_recr_impl[i] = has_attn ? 0 : 1;
        }
    }

    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,        hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
    if (hparams.n_ff_shexp == 0) {
        hparams.n_ff_shexp = hparams.n_ff_exp() * std::max(1u, hparams.n_expert_shared);
    }
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,       hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,        hparams.expert_weights_norm, false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,         hparams.expert_gating_func);

    switch (hparams.n_layer()) {
        case 20: type = LLM_TYPE_2B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_alice_ai::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    if (n_expert == 0) {
        throw std::runtime_error(arch_name() + " model cannot have zero experts");
    }

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output_norm (plain RMSNorm, converter stored as-is)
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);

    // output.weight may be absent if tied to token_embd
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (output == nullptr) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    // Final mix tensors (global, not per-layer)
    output_res_score = create_tensor(tn(LLM_TENSOR_OUTPUT_RES_SCORE, "weight"), {n_embd}, 0);

    // output_res_norm (plain RMSNorm, converter stored as-is)
    output_res_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_RES_NORM, "weight"), {n_embd}, TENSOR_NOT_REQUIRED);

    // Read KDA parameters from GGUF metadata
    const int64_t head_k_dim = hparams.n_embd_head_kda;
    const int64_t head_v_dim = head_k_dim;
    const int64_t n_k_heads  = hparams.ssm_n_group;
    const int64_t d_conv     = hparams.ssm_d_conv;
    const int64_t d_inner    = head_k_dim * n_k_heads;

    // MoE intermediate size per expert
    const int64_t n_ff_exp = hparams.n_ff_exp() > 0 ? hparams.n_ff_exp() : n_ff / n_expert_used;

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        // Shared across both layer types
        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_norm  = create_tensor(tn(LLM_TENSOR_FFN_NORM,  "weight", i), {n_embd}, 0);
        layer.attn_res_score = create_tensor(tn(LLM_TENSOR_ATTN_RES_SCORE, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
        layer.ffn_res_score  = create_tensor(tn(LLM_TENSOR_FFN_RES_SCORE,  "weight", i), {n_embd}, 0);

        // attn_res_norm (plain RMSNorm, stored as-is by converter)
        layer.attn_norm_2 = create_tensor(tn(LLM_TENSOR_ATTN_RES_NORM, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
        // ffn_res_norm (plain RMSNorm, stored as-is by converter)
        layer.ffn_post_norm_2 = create_tensor(tn(LLM_TENSOR_FFN_RES_NORM, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);

        // Detect layer type by checking for SSM tensor presence in GGUF.
        // This handles reduced oracle models where the is_recr pattern may not match.
        bool has_ssm = ml.get_tensor_meta(tn(LLM_TENSOR_SSM_F_A, "weight", i).str().c_str()) != nullptr;
        bool has_attn = ml.get_tensor_meta(tn(LLM_TENSOR_ATTN_Q, "weight", i).str().c_str()) != nullptr;

        if (has_ssm) {
            // === KDA (linear attention) layer ===

            // Conv1d weights: check GGUF shape then create matching tensor
            auto load_conv = [&](llm_tensor tid) {
                auto name = tn(tid, "weight", i);
                ggml_tensor * existing = ml.get_tensor_meta(name.str().c_str());
                if (existing) {
                    return create_tensor(name, {existing->ne[0], existing->ne[1]}, 0);
                }
                return (ggml_tensor *)nullptr;
            };
            layer.ssm_q_conv = load_conv(LLM_TENSOR_SSM_CONV1D_Q);
            layer.ssm_k_conv = load_conv(LLM_TENSOR_SSM_CONV1D_K);
            layer.ssm_v_conv = load_conv(LLM_TENSOR_SSM_CONV1D_V);

            // KDA query/key/value/bias projections: ssm_q/k/v/b
            {
                std::string wq_name = "blk." + std::to_string(i) + ".ssm_q.weight";
                ggml_tensor * q_meta = ml.get_tensor_meta(wq_name.c_str());
                if (q_meta) {
                    layer.wq = create_tensor(tn(LLM_TENSOR_SSM_Q, "weight", i), {q_meta->ne[0], q_meta->ne[1]}, 0);
                } else {
                    layer.wq = create_tensor(tn(LLM_TENSOR_SSM_Q, "weight", i), {d_inner, n_embd}, TENSOR_NOT_REQUIRED);
                }
            }
            {
                std::string wk_name = "blk." + std::to_string(i) + ".ssm_k.weight";
                ggml_tensor * k_meta = ml.get_tensor_meta(wk_name.c_str());
                if (k_meta) {
                    layer.wk = create_tensor(tn(LLM_TENSOR_SSM_K, "weight", i), {k_meta->ne[0], k_meta->ne[1]}, 0);
                } else {
                    layer.wk = create_tensor(tn(LLM_TENSOR_SSM_K, "weight", i), {d_inner, n_embd}, TENSOR_NOT_REQUIRED);
                }
            }
            {
                std::string wv_name = "blk." + std::to_string(i) + ".ssm_v.weight";
                ggml_tensor * v_meta = ml.get_tensor_meta(wv_name.c_str());
                if (v_meta) {
                    layer.wv = create_tensor(tn(LLM_TENSOR_SSM_V, "weight", i), {v_meta->ne[0], v_meta->ne[1]}, 0);
                } else {
                    layer.wv = create_tensor(tn(LLM_TENSOR_SSM_V, "weight", i), {d_inner, n_embd}, TENSOR_NOT_REQUIRED);
                }
            }
            {
                std::string wb_name = "blk." + std::to_string(i) + ".ssm_b.weight";
                ggml_tensor * b_meta = ml.get_tensor_meta(wb_name.c_str());
                if (b_meta) {
                    layer.wb = create_tensor(tn(LLM_TENSOR_SSM_B, "weight", i), {b_meta->ne[0], b_meta->ne[1]}, 0);
                } else {
                    layer.wb = create_tensor(tn(LLM_TENSOR_SSM_B, "weight", i), {d_inner, n_k_heads}, TENSOR_NOT_REQUIRED);
                }
            }

            // KDA specific tensors — make all SSM-specific tensors optional
            layer.ssm_f_a = create_tensor(tn(LLM_TENSOR_SSM_F_A, "weight", i), {n_embd, head_v_dim}, TENSOR_NOT_REQUIRED);
            layer.ssm_f_b = create_tensor(tn(LLM_TENSOR_SSM_F_B, "weight", i), {head_v_dim, d_inner}, TENSOR_NOT_REQUIRED);
            layer.ssm_beta = create_tensor(tn(LLM_TENSOR_SSM_BETA, "weight", i), {n_embd, n_k_heads}, TENSOR_NOT_REQUIRED);

            // A_log: shape [n_k_heads] (1-D)
            layer.ssm_a = create_tensor(tn(LLM_TENSOR_SSM_A_LOG, i), {n_k_heads}, TENSOR_NOT_REQUIRED);

            // ssm_dt_bias: GGUF name is blk.{i}.ssm_dt_bias.weight, tensor mapping uses blk.{i}.ssm_dt.bias
            // Use get_tensor_meta to find the actual GGUF tensor name
            {
                std::string dt_name = "blk." + std::to_string(i) + ".ssm_dt_bias.weight";
                ggml_tensor * dt_meta = ml.get_tensor_meta(dt_name.c_str());
                if (dt_meta) {
                    layer.ssm_dt_b = create_tensor(tn(LLM_TENSOR_SSM_DT_BIAS, "weight", i), {dt_meta->ne[0]}, 0);
                } else {
                    layer.ssm_dt_b = create_tensor(tn(LLM_TENSOR_SSM_DT_BIAS, "weight", i), {d_inner}, TENSOR_NOT_REQUIRED);
                }
            }

            // Output gate: g_a_proj -> g_b_proj
            layer.ssm_g_a = create_tensor(tn(LLM_TENSOR_SSM_G_A, "weight", i), {n_embd, head_v_dim}, TENSOR_NOT_REQUIRED);
            layer.ssm_g_b = create_tensor(tn(LLM_TENSOR_SSM_G_B, "weight", i), {head_v_dim, head_v_dim * n_k_heads}, TENSOR_NOT_REQUIRED);

            // o_norm (plain RMSNorm, converter stored as-is)
            layer.ssm_o_norm = create_tensor(tn(LLM_TENSOR_SSM_OUT_NORM, "weight", i), {head_v_dim}, TENSOR_NOT_REQUIRED);

            // o_proj: [head_v_dim * n_k_heads, hidden] — GGUF name is blk.{i}.ssm_o.weight
            {
                std::string wo_name = "blk." + std::to_string(i) + ".ssm_o.weight";
                ggml_tensor * wo_meta = ml.get_tensor_meta(wo_name.c_str());
                if (wo_meta) {
                    layer.wo = create_tensor(tn(LLM_TENSOR_SSM_O, "weight", i), {wo_meta->ne[0], wo_meta->ne[1]}, 0);
                } else {
                    layer.wo = create_tensor(tn(LLM_TENSOR_SSM_O, "weight", i), {head_v_dim * n_k_heads, n_embd}, TENSOR_NOT_REQUIRED);
                }
            }

        } else if (has_attn) {
            // === Full attention layer ===
            // q_proj output = 2 * n_head * head_dim (query + gate folded)
            // Use get_tensor_meta to find actual GGUF shape since head_dim differs from KDA head_k_dim
            {
                std::string q_name = "blk." + std::to_string(i) + ".attn_q.weight";
                ggml_tensor * q_meta = ml.get_tensor_meta(q_name.c_str());
                if (q_meta) {
                    layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), {q_meta->ne[0], q_meta->ne[1]}, 0);
                } else {
                    layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_head * head_k_dim * 2}, TENSOR_NOT_REQUIRED);
                }
            }
            // For full-attn layers, K/V use head_dim (64) not head_k_dim (32)
            {
                std::string k_name = "blk." + std::to_string(i) + ".attn_k.weight";
                ggml_tensor * k_meta = ml.get_tensor_meta(k_name.c_str());
                if (k_meta) {
                    layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), {k_meta->ne[0], k_meta->ne[1]}, 0);
                }
            }
            {
                std::string v_name = "blk." + std::to_string(i) + ".attn_v.weight";
                ggml_tensor * v_meta = ml.get_tensor_meta(v_name.c_str());
                if (v_meta) {
                    layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), {v_meta->ne[0], v_meta->ne[1]}, 0);
                }
            }

             // Q/K RMSNorm: full-attn uses head_dim (64), KDA uses head_k_dim (32)
             {
                 std::string qn_name = "blk." + std::to_string(i) + ".attn_q_norm.weight";
                 ggml_tensor * qn_meta = ml.get_tensor_meta(qn_name.c_str());
                 if (qn_meta) {
                     layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {qn_meta->ne[0]}, 0);
                 }
             }
             {
                 std::string kn_name = "blk." + std::to_string(i) + ".attn_k_norm.weight";
                 ggml_tensor * kn_meta = ml.get_tensor_meta(kn_name.c_str());
                 if (kn_meta) {
                     layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {kn_meta->ne[0]}, 0);
                 }
             }

              // o_proj: full-attn layers use attn_output.weight in GGUF
               {
                    ggml_tensor * wo_meta = ml.get_tensor_meta(tn(LLM_TENSOR_ATTN_OUT, "weight", i).str().c_str());
                   if (wo_meta) {
                       layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {wo_meta->ne[0], wo_meta->ne[1]}, 0);
                   } else {
                       layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, TENSOR_NOT_REQUIRED);
                   }
               }
        } else {
            // Layer has neither SSM nor attention tensors (e.g., missing layer in reduced model).
            // Still create the shared tensors and MoE tensors so the layer count matches.
            // wq/wk/wv/wo remain nullptr — the graph builder will handle this.
        }

        // MoE tensors (all layers)
        layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, nullptr, i), {n_embd, n_expert}, 0);
        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, i), {n_expert}, 0);

        // fused gate_up_exps: GGUF shape [hidden, 2*inter, E]
        layer.ffn_gate_up_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, nullptr, i), {n_embd, n_ff_exp * 2, n_expert}, 0);
        // ffn_down_exps: GGUF shape [inter, hidden, E]
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, nullptr, i), {n_ff_exp, n_embd, n_expert}, 0);

        // Shared expert (may be absent if n_expert_shared == 0)
        const int64_t n_ff_shexp = hparams.n_ff_shexp;
        layer.ffn_gate_inp_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", i), {n_embd, 1}, TENSOR_NOT_REQUIRED);
        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_shexp}, TENSOR_NOT_REQUIRED);
        layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_shexp}, TENSOR_NOT_REQUIRED);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_shexp, n_embd}, TENSOR_NOT_REQUIRED);
    }
}

std::unique_ptr<llm_graph_context> llama_model_alice_ai::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

//
// Graph building
//

llama_model_alice_ai::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "model.embed_tokens", -1);

    auto * inp_rs = build_rs_inp_hybrid();

    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    (void)inp_out_ids;

    const int64_t n_head = hparams.n_head();
    const int64_t head_k_dim = hparams.n_embd_head_kda;
    const int64_t head_v_dim = head_k_dim;
    const int64_t n_k_heads = hparams.ssm_n_group;
    const int64_t d_conv = hparams.ssm_d_conv;
    const int64_t d_inner = head_k_dim * n_k_heads;
    const int64_t n_seqs = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;
    const int64_t n_tokens = n_seq_tokens * n_seqs;
    const float eps = hparams.f_norm_rms_eps;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_tokens);

    // Detect actual residual block size by scanning for layers with both SSM and attention tensors.
    // For reduced oracle models the 4-layer block pattern may not apply.
    // We use a dynamic block size: check if any layer has both SSM and attention.
    // If all layers have only one type, use block_size = n_layer (no block splitting).
    int32_t res_block_size = n_layer;
    {
        int32_t max_layer_idx = 0;
        for (int il = 0; il < n_layer; ++il) {
            const auto & layer = model.layers[il];
            bool has_ssm = layer.ssm_f_a != nullptr;
            bool has_attn = layer.wq != nullptr &&
                (layer.attn_q_norm != nullptr || layer.attn_k_norm != nullptr);
            if (has_ssm && has_attn) {
                max_layer_idx = il;
                break;
            }
            if (has_ssm || has_attn) {
                max_layer_idx = il;
            }
        }
        // If we found mixed layers, check if the 4-layer pattern applies
        // by looking for a layer at index 3 with attention-only (or vice versa)
        if (max_layer_idx >= 3) {
            res_block_size = 4;
        }
    }

    // completed_blocks tracking: stack of tensors for residual mixing
    // completed_stack: [n_embd, n_completed, n_tokens]
    // At layer 0: completed_stack = [inpL], partial = None
    // At layer 4: push partial -> completed_stack = [inpL_0, partial_0-3], partial = None
    // etc.
    ggml_tensor * completed_stack = ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens);
    ggml_tensor * partial = nullptr;

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        // Detect layer type by checking tensor presence (not hparams.is_recr which may be stale)
        bool has_ssm = layer.ssm_f_a != nullptr;
        bool has_attn = layer.wq != nullptr &&
            (layer.attn_q_norm != nullptr || layer.attn_k_norm != nullptr);
        bool is_recr_layer = has_ssm && !has_attn;
        bool is_attn_layer = has_attn && !has_ssm;
        bool is_mixed = has_ssm && has_attn;

        // --- Residual stream management (completed_blocks + partial) ---
        // At layer_idx > 0 && layer_idx % block_size == 0: push partial onto completed, reset partial.
        // Then mix completed + [partial] via _depth_softmax_mix.
        if (il > 0 && (il % res_block_size) == 0) {
            // Push partial onto completed stack
            if (partial) {
                ggml_tensor * p_3d = ggml_reshape_3d(ctx0, partial, n_embd, 1, n_tokens);
                completed_stack = completed_stack ? ggml_concat(ctx0, completed_stack, p_3d, 1) : p_3d;
            }
            partial = nullptr;
        }

        // Build attention input via _depth_softmax_mix(completed + [partial], res_score, res_norm)
        if (il == 0) {
            // Layer 0: no mixing, just use inputs_embeds directly
            cur = inpL;
        } else {
            // Mix completed blocks + partial
            // _depth_softmax_mix: RMSNorm each source with shared norm_weight,
            // scalar res_proj->1, softmax over depth, weighted sum of RAW sources
            ggml_tensor * sources = partial
                ? (completed_stack
                    ? ggml_concat(ctx0, completed_stack, ggml_reshape_3d(ctx0, partial, n_embd, 1, n_tokens), 1)
                    : ggml_reshape_3d(ctx0, partial, n_embd, 1, n_tokens))
                : completed_stack;

            // RMSNorm each source with shared res_norm weight
            ggml_tensor * normed = ggml_rms_norm(ctx0, sources, eps);
            normed = ggml_mul(ctx0, normed, layer.attn_norm_2);
            cb(normed, "attn_res_normed", il);

            // res_proj: scalar weights for each source
            ggml_tensor * scores = ggml_mul(ctx0, normed, layer.attn_res_score);
            scores = ggml_sum_rows(ctx0, scores);
            cb(scores, "attn_res_scores", il);

            // Softmax over depth axis
            scores = ggml_reshape_2d(ctx0, scores, ggml_nrows(scores) / n_tokens, n_tokens);
            ggml_tensor * probs = ggml_soft_max(ctx0, scores);
            cb(probs, "attn_res_probs", il);

            // Weighted sum of RAW sources
            // Split probs: first n_completed for completed_stack, last 1 for partial
            const int n_completed = (int) probs->ne[0];
            ggml_tensor * p_src = ggml_cont(ctx0, ggml_view_2d(ctx0, probs, n_completed, n_tokens, probs->nb[1], 0));
            ggml_tensor * p_part = nullptr;
            if (partial && n_completed < (int) probs->ne[0]) {
                p_part = ggml_cont(ctx0, ggml_view_2d(ctx0, probs, 1, n_tokens, probs->nb[1],
                                                                     probs->nb[0] * n_completed));
            }

            // HC pre for the stack
            ggml_tensor * out = ggml_dsv4_hc_pre(ctx0, sources, p_src);

            // Broadcast multiply for partial
            if (p_part) {
                ggml_tensor * p_part_3d = ggml_reshape_3d(ctx0, p_part, 1, n_tokens, 1);
                out = ggml_add(ctx0, out, ggml_mul(ctx0, partial, p_part_3d));
            }

            cur = out;
            cb(cur, "attn_mixed", il);
        }

        // Attention input norm (zero-centered, but already subtracted 1.0 on load)
        cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);
        ggml_build_forward_expand(gf, cur);

        if (is_recr_layer || is_mixed) {
            // === KDA (linear attention) layer ===
            const auto * mctx_cur = inp_rs->mctx;
            const auto kv_head = mctx_cur->get_head();

            // Get conv states from r_l tensor
            ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
            ggml_tensor * conv_state_all = build_rs(inp_rs, conv_states_all, hparams.n_embd_r(), n_seqs);

            const int64_t mem_size = mctx_cur->get_size();
            (void)mem_size;
            const int64_t K_rs = (int64_t) cparams.n_rs_seq + 1;
            (void)K_rs;

            // Causal conv1d over Q, K, V
            auto conv1d = [&](int qkv, ggml_tensor * proj_w, ggml_tensor * conv_w) {
                ggml_tensor * conv_state_x = ggml_view_3d(ctx0, conv_state_all, d_conv - 1, d_inner, n_seqs,
                    (d_conv - 1) * ggml_element_size(conv_state_all),
                    hparams.n_embd_r() * ggml_element_size(conv_state_all),
                    qkv * ((d_conv - 1) * d_inner) * ggml_element_size(conv_state_all));

                ggml_tensor * x_proj = proj_w ? ggml_mul_mat(ctx0, proj_w, cur) : cur;
                cb(x_proj, qkv==0?"kda_q_proj":qkv==1?"kda_k_proj":"kda_v_proj", il);
                ggml_tensor * x_3d = ggml_reshape_3d(ctx0, x_proj, d_inner, n_seq_tokens, n_seqs);
                ggml_tensor * conv_x = ggml_concat(ctx0, conv_state_x, ggml_transpose(ctx0, x_3d), 0);

                // Save last (d_conv-1) columns back to conv state
                ggml_tensor * last_conv_x = ggml_view_3d(ctx0, conv_x, d_conv - 1, d_inner, n_seqs,
                    conv_x->nb[1], conv_x->nb[2], n_seq_tokens * conv_x->nb[0]);
                ggml_build_forward_expand(gf,
                    ggml_cpy(ctx0, last_conv_x,
                        ggml_view_3d(ctx0, conv_states_all, d_conv - 1, d_inner, n_seqs,
                            (d_conv - 1) * ggml_element_size(conv_states_all),
                            hparams.n_embd_r() * ggml_element_size(conv_states_all),
                            (kv_head * hparams.n_embd_r() + qkv * ((d_conv - 1) * d_inner)) * ggml_element_size(conv_states_all))));

                ggml_tensor * conv_weight = ggml_reshape_2d(ctx0, conv_w, d_conv, d_inner);
                ggml_tensor * Xcur = ggml_ssm_conv(ctx0, conv_x, conv_weight);
                Xcur = ggml_reshape_2d(ctx0, Xcur, d_inner, n_tokens);
                Xcur = ggml_silu(ctx0, Xcur);
                cb(Xcur, qkv==0?"kda_q_conv":qkv==1?"kda_k_conv":"kda_v_conv", il);

                return ggml_reshape_4d(ctx0, Xcur, head_k_dim, n_k_heads, n_seq_tokens, n_seqs);
            };

            ggml_tensor * Qcur = conv1d(0, layer.wq, layer.ssm_q_conv);
            ggml_tensor * Kcur = conv1d(1, layer.wk, layer.ssm_k_conv);
            ggml_tensor * Vcur = conv1d(2, layer.wv, layer.ssm_v_conv);

              // Gate computation: g = -exp(a_log_bias) * softplus(alpha + dt_bias)
              ggml_tensor * f_a = ggml_mul_mat(ctx0, layer.ssm_f_a, cur);
              ggml_tensor * g1 = ggml_mul_mat(ctx0, layer.ssm_f_b, f_a);
              cb(g1, "kda_alpha", il);

                // ssm_dt_b is stored as [d_inner] = [head_k_dim * n_k_heads].
                // HF interprets it as [num_k_heads, head_k_dim] (row-major in h,d).
                // llama.cpp reshapes g1 to [n_k_heads, head_k_dim, n_tokens] then
                // permutes to [head_k_dim, n_k_heads, n_tokens] so g1[d,h,t] =
                // alpha[0,t,h,d]. ggml_add broadcasts ssm_dt_b to [head_k_dim,
                // n_k_heads, 1, 1] in C-order, giving ssm_dt_b[d,h] = flat[d *
                // n_k_heads + h]. To match HF's flat[h * head_k_dim + d] ordering,
                // we reshape ssm_dt_b to [n_k_heads, head_k_dim] first, then
                // transpose and flatten to get the correct flat ordering.
                // gate[d, h, t] = softplus(g1[d,h,t] + dt_broadcast[d,h]) * A[h]
                // = softplus(alpha[0,t,h,d] + dt_bias_hf[h,d]) * A[h]
                // = gate_hf[0, t, h, d]
              // ssm_dt_b is already h-major flat [h*head_k_dim + d], matching the
              // alpha row index (i = h*head_k_dim + d). A plain reshape_1d keeps
              // that ordering so the broadcast add lands dt_bias on the right
              // elements. (The old reshape_2d+permute made it d-major, mismatching.)
              ggml_tensor * dt_bias = ggml_reshape_1d(ctx0, layer.ssm_dt_b, head_k_dim * n_k_heads);
              g1 = ggml_add(ctx0, g1, dt_bias);
              g1 = ggml_softplus(ctx0, g1);
              g1 = ggml_reshape_3d(ctx0, g1, head_k_dim, n_k_heads, n_tokens);

              // Permute g1 from [n_k_heads, head_k_dim, n_tokens] to
              // [head_k_dim, n_k_heads, n_tokens] so subsequent reshapes
              // preserve the (h,d) element ordering for the chunking path.
              g1 = ggml_cont(ctx0, g1);

             // A_log: shape [n_k_heads], broadcast to [head_k_dim, n_k_heads, n_tokens]
             // HF: gate = -exp(a_log_bias) * softplus(...)
             ggml_tensor * A = ggml_reshape_3d(ctx0, layer.ssm_a, 1, n_k_heads, 1);
             A = ggml_exp(ctx0, A);          // exp(a_log_bias)
             A = ggml_neg(ctx0, A);          // -exp(a_log_bias)
             g1 = ggml_mul(ctx0, g1, A);
             cb(g1, "kda_g1", il);

               g1 = ggml_reshape_4d(ctx0, g1, head_k_dim, n_k_heads, n_seq_tokens, n_seqs);

            // Beta mixing coefficient (b_proj -> ssm_b.weight in GGUF)
            ggml_tensor * beta = ggml_mul_mat(ctx0, layer.wb, cur);
            beta = ggml_reshape_4d(ctx0, beta, 1, n_k_heads, n_seq_tokens, n_seqs);
            cb(beta, "kda_beta", il);
            cb(beta, "kda_beta_raw", il);
            beta = ggml_sigmoid(ctx0, beta);
            cb(beta, "kda_beta", il);
            // kda_allow_negative_eigenvalues=false -> don't multiply by 2

            // L2-normalize Q and K
            ggml_tensor * cur_3d = ggml_reshape_3d(ctx0, cur, cur->ne[0], n_seq_tokens, n_seqs);
            Qcur = build_gdn_l2_norm(ctx0, Qcur, eps);
            Kcur = build_gdn_l2_norm(ctx0, Kcur, eps);
            // k_l2 checkpoint: reference k_l2 = normalize(K), matches post-L2 Kcur exactly.
            cb(Kcur, "kda_k_l2", il);
            // q_l2_scaled = normalize(Q) * head_k_dim**-0.5; the scale is applied inside
            // build_delta_net (ggml_scale), so cb() the scaled Q there as "kda_q_l2".

            // KDA recurrence
            ggml_tensor * ssm_states_all = mctx_cur->get_s_l(il);
            ggml_tensor * state = build_rs(inp_rs, ssm_states_all, hparams.n_embd_s(), n_seqs);
            state = ggml_reshape_4d(ctx0, state, head_k_dim, head_k_dim, n_k_heads, n_seqs);

            // g1 is already [head_k_dim, n_k_heads, n_seq_tokens, n_seqs]
            // from the reshape at line 511. The fused ggml_gated_delta_net
            // kernel iterates over dim1 as the head index (iv1 = ir % H),
            // so dim1 must be n_k_heads, not 1. The non-fused chunking path
            // also expects head in dim1 (see delta-net-base.cpp line 42).
            cb(g1, "kda_g1", il);

            auto attn_out = build_delta_net(Qcur, Kcur, Vcur, g1, beta, state, il);

            ggml_tensor * output = ggml_cont(ctx0, attn_out.first);
            ggml_tensor * new_state = attn_out.second;
            cb(attn_out.first, "kda_delta_out", il);
            cb(new_state, "new_state", il);

            // Update recurrent states
            ggml_build_forward_expand(gf,
                ggml_cpy(ctx0, new_state,
                    ggml_view_1d(ctx0, ssm_states_all, hparams.n_embd_s() * n_seqs,
                        kv_head * hparams.n_embd_s() * ggml_element_size(ssm_states_all))));

            // Output gating: g2 = g_b(g_a(x)), then o_norm * sigmoid(g2)
            ggml_tensor * cur_2d = ggml_reshape_2d(ctx0, cur_3d, cur_3d->ne[0], n_seq_tokens * n_seqs);
            ggml_tensor * g_a = ggml_mul_mat(ctx0, layer.ssm_g_a, cur_2d);
            ggml_tensor * g2 = ggml_mul_mat(ctx0, layer.ssm_g_b, g_a);
            cb(g2, "kda_gate_raw", il);
            g2 = ggml_reshape_3d(ctx0, g2, head_v_dim, n_k_heads, n_seq_tokens * n_seqs);

             // output is already [head_v_dim, n_k_heads, n_tokens*n_seqs] with
             // head_v_dim as ne0. build_norm (RMS) normalizes over ne0 per column,
             // matching HF's o_norm over the last dim of [batch, seq, num_v_heads, head_v_dim].
             // Keep that ordering; do NOT permute (permuting scrambles the column
             // order so the gate multiply below lands on wrong elements).
             ggml_tensor * attn_out_3d = ggml_reshape_3d(ctx0, output, head_v_dim, n_k_heads, n_seq_tokens * n_seqs);
             ggml_tensor * attn_out_2d = ggml_reshape_2d(ctx0, attn_out_3d, head_v_dim, n_k_heads * n_seq_tokens * n_seqs);
              ggml_tensor * normed = build_norm(attn_out_2d, layer.ssm_o_norm, nullptr, LLM_NORM_RMS, il);
              ggml_tensor * o_norm = ggml_reshape_4d(ctx0, normed, head_v_dim, n_k_heads, n_seq_tokens, n_seqs);
               cb(normed, "kda_normed", il);
               cb(o_norm, "kda_o_norm", il);
             ggml_tensor * gate = ggml_sigmoid(ctx0, g2);
             cb(gate, "kda_gate", il);
             gate = ggml_reshape_2d(ctx0, gate, head_v_dim, n_k_heads * n_seq_tokens * n_seqs);
              ggml_tensor * gated = ggml_mul(ctx0, normed, gate);
              // View as [head_v_dim, n_k_heads, n_seq_tokens, n_seqs] so the
              // checkpoint matches HF's [batch, seq, num_v_heads, head_v_dim].
              gated = ggml_reshape_4d(ctx0, gated, head_v_dim, n_k_heads, n_seq_tokens, n_seqs);
              cb(gated, "kda_gated", il);

             // Output projection
            gated = ggml_cont_2d(ctx0, gated, d_inner, n_tokens);
            cur = ggml_mul_mat(ctx0, layer.wo, gated);
             cb(cur, "attn_output", il);

        } else if (is_attn_layer) {
            // === Full attention layer ===

            // Gate folded inside q_proj: q_proj output = [query, gate]
            ggml_tensor * QGcur = ggml_mul_mat(ctx0, layer.wq, cur);
            // View as [head_dim, n_head, 2, n_tokens], split into q and gate
            // wq shape is [n_embd, 2*n_head*head_dim], so mul_mat output is [2*n_head*head_dim, n_tokens]
            const int64_t attn_head_dim = layer.wq->ne[1] / (2 * n_head);
            QGcur = ggml_reshape_4d(ctx0, QGcur, attn_head_dim, n_head, 2, n_tokens);

            ggml_tensor * Qcur = ggml_view_4d(ctx0, QGcur, attn_head_dim, n_head, n_tokens, 1,
                                                QGcur->nb[1], QGcur->nb[2], QGcur->nb[3], 0);
            cb(Qcur, "Qcur_view", il);

            ggml_tensor * gate = ggml_view_4d(ctx0, QGcur, attn_head_dim, n_head, n_tokens, 1,
                                                QGcur->nb[1], QGcur->nb[2], QGcur->nb[3],
                                                attn_head_dim * ggml_element_size(QGcur));
            cb(gate, "gate", il);

            // K, V projections
            ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.wk, cur);
            ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.wv, cur);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            // Derive attention head_dim from weight tensor shapes
            // wk shape is [n_embd, n_head_kv * head_dim], so mul_mat output is [n_head_kv * head_dim, n_tokens]
            const int64_t attn_k_head_dim = layer.wk->ne[0] / n_head_kv;
            const int64_t attn_v_head_dim = layer.wv->ne[0] / n_head_kv;
            Kcur = ggml_reshape_3d(ctx0, Kcur, attn_k_head_dim, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, attn_v_head_dim, n_head_kv, n_tokens);

            // Q/K RMSNorm (already zero-centered: subtracted 1.0 on load)
            Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_normed", il);

            // Partial rotary: rotary_dim = head_dim * 0.25
            // rope_theta = 1e6
            const int64_t rotary_dim = attn_head_dim / 4;  // 16
            const float rope_freq_base = 1e6f;

            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                    rotary_dim, 0, 0, rope_freq_base, 1.0f,
                    1.0f, 1.0f, 0.0f, 0.0f);

            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr,
                    rotary_dim, 0, 0, rope_freq_base, 1.0f,
                    1.0f, 1.0f, 0.0f, 0.0f);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

             // Attention scale
             const float kq_scale = 1.0f / sqrtf((float) head_k_dim);

             auto * inp_attn = build_attn_inp_no_cache();

            cur = build_attn(inp_attn,
                        nullptr, nullptr, nullptr,
                        Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            cb(cur, "attn_pregate", il);

            // Apply sigmoid gate
            gate = ggml_cont_2d(ctx0, gate, attn_head_dim * n_head, n_tokens);
            gate = ggml_sigmoid(ctx0, gate);
            cb(gate, "gate_sigmoid", il);

            cur = ggml_mul(ctx0, cur, gate);
            cb(cur, "attn_gated", il);

            // Output projection
            cur = ggml_mul_mat(ctx0, layer.wo, cur);
            cb(cur, "attn_output", il);
        } else {
            // Layer has neither SSM nor attention tensors (missing/empty layer).
            // Pass through: cur remains unchanged as the layer output.
            // This handles reduced models where some layers have no tensors.
            cb(cur, "layer_pass", il);
        }

        // Residual: partial += attn_out
        partial = partial ? ggml_add(ctx0, partial, cur) : cur;
        cb(partial, "partial_attn", il);

        // MoE input: mix completed + [partial] via mlp_res_score + mlp_res_norm
        ggml_tensor * moe_mixed;
        if (il == 0) {
            // Layer 0: no mlp mixing yet (partial is the only block)
            moe_mixed = partial;
        } else {
            // Mix completed + [partial]
            ggml_tensor * sources = partial
                ? (completed_stack
                    ? ggml_concat(ctx0, completed_stack, ggml_reshape_3d(ctx0, partial, n_embd, 1, n_tokens), 1)
                    : ggml_reshape_3d(ctx0, partial, n_embd, 1, n_tokens))
                : completed_stack;

            // RMSNorm each source with shared res_norm weight (plain)
            ggml_tensor * normed = ggml_rms_norm(ctx0, sources, eps);
            normed = ggml_mul(ctx0, normed, layer.ffn_post_norm_2);
            cb(normed, "ffn_res_normed", il);

            // res_proj: scalar weights
            ggml_tensor * scores = ggml_mul(ctx0, normed, layer.ffn_res_score);
            scores = ggml_sum_rows(ctx0, scores);
            cb(scores, "ffn_res_scores", il);

            scores = ggml_reshape_2d(ctx0, scores, ggml_nrows(scores) / n_tokens, n_tokens);
            ggml_tensor * probs = ggml_soft_max(ctx0, scores);
            cb(probs, "ffn_res_probs", il);

            const int n_completed = (int) probs->ne[0];
            ggml_tensor * p_src = ggml_cont(ctx0, ggml_view_2d(ctx0, probs, n_completed, n_tokens, probs->nb[1], 0));
            ggml_tensor * p_part = nullptr;
            if (partial && n_completed < (int) probs->ne[0]) {
                p_part = ggml_cont(ctx0, ggml_view_2d(ctx0, probs, 1, n_tokens, probs->nb[1],
                                                                     probs->nb[0] * n_completed));
            }

            ggml_tensor * out = ggml_dsv4_hc_pre(ctx0, sources, p_src);

            if (p_part) {
                ggml_tensor * p_part_3d = ggml_reshape_3d(ctx0, p_part, 1, n_tokens, 1);
                out = ggml_add(ctx0, out, ggml_mul(ctx0, partial, p_part_3d));
            }

            moe_mixed = out;
            cb(moe_mixed, "moe_mixed", il);
        }

        // Post-attention norm (zero-centered, already subtracted 1.0 on load)
        moe_mixed = build_norm(moe_mixed, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(moe_mixed, "ffn_norm", il);

        // MoE FFN
        const int64_t n_ff_exp = hparams.n_ff_exp() > 0 ? hparams.n_ff_exp() : hparams.n_ff() / n_expert_used;

        // Router: sigmoid scores, top-k
        ggml_tensor * logits = ggml_mul_mat(ctx0, layer.ffn_gate_inp, moe_mixed);
        cb(logits, "ffn_moe_logits", il);

        ggml_tensor * moe_out = build_moe_ffn(moe_mixed,
            layer.ffn_gate_inp,
            layer.ffn_up_exps,
            layer.ffn_gate_exps,
            layer.ffn_down_exps,
            layer.ffn_exp_probs_b,
            n_expert, n_expert_used,
            LLM_FFN_SILU, true,
            hparams.expert_weights_scale,
            (llama_expert_gating_func_type) hparams.expert_gating_func, il,
            logits, layer.ffn_gate_up_exps,
            layer.ffn_up_exps_s,
            layer.ffn_gate_exps_s,
            layer.ffn_down_exps_s);
        cb(moe_out, "ffn_moe_out", il);

        // Shared expert: shared_gate * sigmoid(shared_gate_inp)
        if (layer.ffn_gate_inp_shexp) {
            ggml_tensor * shared_gate = build_lora_mm(layer.ffn_gate_inp_shexp, moe_mixed);
            cb(shared_gate, "shared_expert_gate", il);
            shared_gate = ggml_sigmoid(ctx0, shared_gate);
            cb(shared_gate, "shared_expert_gate_sigmoid", il);

            ggml_tensor * sh = build_ffn(moe_mixed,
                layer.ffn_up_shexp, nullptr, layer.ffn_up_shexp_s,
                layer.ffn_gate_shexp, nullptr, layer.ffn_gate_shexp_s,
                layer.ffn_down_shexp, nullptr, layer.ffn_down_shexp_s,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(sh, "ffn_shexp", il);

            sh = ggml_mul(ctx0, sh, shared_gate);
            cb(sh, "ffn_shexp_gated", il);

            moe_out = ggml_add(ctx0, moe_out, sh);
        }

        // Layer output = partial + moe_out
        cur = ggml_add(ctx0, partial, moe_out);
        cb(cur, "layer_out", il);

        inpL = cur;
    }

    cur = inpL;

    // Final attnres mix
    {
        ggml_tensor * sources = partial
            ? (completed_stack
                ? ggml_concat(ctx0, completed_stack, ggml_reshape_3d(ctx0, partial, n_embd, 1, n_tokens), 1)
                : ggml_reshape_3d(ctx0, partial, n_embd, 1, n_tokens))
            : completed_stack;

        ggml_tensor * normed = ggml_rms_norm(ctx0, sources, eps);
        normed = ggml_mul(ctx0, normed, model.output_res_norm);
        cb(normed, "final_res_normed", -1);

        ggml_tensor * scores = ggml_mul(ctx0, normed, model.output_res_score);
        scores = ggml_sum_rows(ctx0, scores);
        cb(scores, "final_res_scores", -1);

        scores = ggml_reshape_2d(ctx0, scores, ggml_nrows(scores) / n_tokens, n_tokens);
        ggml_tensor * probs = ggml_soft_max(ctx0, scores);
        cb(probs, "final_res_probs", -1);

        const int n_completed = (int) probs->ne[0];
        ggml_tensor * p_src = ggml_cont(ctx0, ggml_view_2d(ctx0, probs, n_completed, n_tokens, probs->nb[1], 0));
        ggml_tensor * p_part = nullptr;
        if (partial && n_completed < (int) probs->ne[0]) {
            p_part = ggml_cont(ctx0, ggml_view_2d(ctx0, probs, 1, n_tokens, probs->nb[1],
                                                                 probs->nb[0] * n_completed));
        }

        ggml_tensor * out = ggml_dsv4_hc_pre(ctx0, sources, p_src);

        if (p_part) {
            ggml_tensor * p_part_3d = ggml_reshape_3d(ctx0, p_part, 1, n_tokens, 1);
            out = ggml_add(ctx0, out, ggml_mul(ctx0, partial, p_part_3d));
        }

        cur = out;
        cb(cur, "attnres_final", -1);
    }

    // Final norm (plain RMSNorm, output_norm)
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // LM head
    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

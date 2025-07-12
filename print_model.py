from DSV.models.t2v_model import T2V_Model

if __name__ == "__main__":
    model = T2V_Model(
        in_channels=4,
        patch_size=2,
        sample_size=32,
        num_layers=36,
        activation_fn="gelu-approximate",
        norm_type="ada_norm_single",
        caption_channels=4096,
        cross_attention_dim=32 * 256,
        attention_head_dim=256,
        num_attention_heads=32,
        num_cross_attention_heads=32,
        return_dict=False,
        use_additional_conditions=False,
        video_length=16,
    ).bfloat16()
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")
    exit()


# T2V_Model(
#   (pos_embed): PatchEmbed(
#     (proj): Conv2d(4, 1024, kernel_size=(2, 2), stride=(2, 2))
#   )
#   (transformer_blocks): ModuleList(
#     (0-1): 2 x BasicTransformerBlock(
#       (norm1): FusedLayerNorm(torch.Size([1024]), eps=1e-05, elementwise_affine=True)
#       (attn1): Attention(
#         (to_q): Linear(in_features=1024, out_features=1024, bias=False)
#         (to_k): Linear(in_features=1024, out_features=1024, bias=False)
#         (to_v): Linear(in_features=1024, out_features=1024, bias=False)
#         (to_out): ModuleList(
#           (0): Linear(in_features=1024, out_features=1024, bias=True)
#           (1): Dropout(p=0.0, inplace=False)
#         )
#       )
#       (norm2): FusedLayerNorm(torch.Size([1024]), eps=1e-05, elementwise_affine=True)
#       (attn2): Attention(
#         (to_q): Linear(in_features=1024, out_features=1024, bias=False)
#         (to_k): Linear(in_features=1024, out_features=1024, bias=False)
#         (to_v): Linear(in_features=1024, out_features=1024, bias=False)
#         (to_out): ModuleList(
#           (0): Linear(in_features=1024, out_features=1024, bias=True)
#           (1): Dropout(p=0.0, inplace=False)
#         )
#       )
#       (ff): FeedForward(
#         (net): ModuleList(
#           (0): GEGLU(
#             (proj): LoRACompatibleLinear(in_features=1024, out_features=8192, bias=True)
#           )
#           (1): Dropout(p=0.0, inplace=False)
#           (2): LoRACompatibleLinear(in_features=4096, out_features=1024, bias=True)
#         )
#       )
#     )
#   )
#   (norm_out): FusedLayerNorm(torch.Size([1024]), eps=1e-06, elementwise_affine=False)
#   (proj_out): Linear(in_features=1024, out_features=16, bias=True)
#   (adaln_single): AdaLayerNormSingle(
#     (emb): CombinedTimestepSizeEmbeddings(
#       (time_proj): Timesteps()
#       (timestep_embedder): TimestepEmbedding(
#         (linear_1): LoRACompatibleLinear(in_features=256, out_features=1024, bias=True)
#         (act): SiLU()
#         (linear_2): LoRACompatibleLinear(in_features=1024, out_features=1024, bias=True)
#       )
#     )
#     (silu): SiLU()
#     (linear): Linear(in_features=1024, out_features=6144, bias=True)
#   )
#   (caption_projection): CaptionProjection(
#     (linear_1): Linear(in_features=1024, out_features=1024, bias=True)
#     (act_1): GELU(approximate='tanh')
#     (linear_2): Linear(in_features=1024, out_features=1024, bias=True)
#   )
# )

"""兼容 Transformers 新旧 ViT 实现的检查点参数名称。"""

VIT_PARAMETER_NAMES = (
    ("attention.q_proj.", "attention.attention.query."),
    ("attention.k_proj.", "attention.attention.key."),
    ("attention.v_proj.", "attention.attention.value."),
    ("attention.o_proj.", "attention.output.dense."),
    ("mlp.fc1.", "intermediate.dense."),
    ("mlp.fc2.", "output.dense."),
)


def align_vit_state_dict(state_dict, target_state):
    """只转换当前模型可识别的 ViT 名称，保留全部张量供后续严格加载。"""
    new_prefix = "encoder.model.layers."
    old_prefix = "encoder.model.encoder.layer."
    aligned = {}
    renamed_count = 0

    for source_key, value in state_dict.items():
        target_key = source_key
        if source_key not in target_state:
            if source_key.startswith(new_prefix):
                candidate = old_prefix + source_key[len(new_prefix) :]
                replacements = VIT_PARAMETER_NAMES
            elif source_key.startswith(old_prefix):
                candidate = new_prefix + source_key[len(old_prefix) :]
                replacements = [(old, new) for new, old in VIT_PARAMETER_NAMES]
            else:
                candidate = source_key
                replacements = ()

            for source_name, target_name in replacements:
                candidate = candidate.replace(source_name, target_name)

            if candidate in target_state:
                target_key = candidate
                renamed_count += 1

        # 两个原始参数不能映射到同一个位置，以免覆盖已经训练好的权重。
        if target_key in aligned:
            raise ValueError(f"Checkpoint parameters map to the same target: {target_key}")
        aligned[target_key] = value

    return aligned, renamed_count

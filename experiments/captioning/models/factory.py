def build_model(name: str, vocab_size: int, config: dict):
    common = {
        "vocab_size": vocab_size,
        "embedding_dim": config.get("embedding_dim", 512),
        "hidden_dim": config.get("hidden_dim", 512),
        "freeze_encoder": config.get("freeze_encoder", True),
        "resnet_path": config.get("resnet_path"),
    }
    if name == "cnn_gru":
        from .cnn_gru import CNNGRU
        return CNNGRU(**common)
    if name == "attention_rnn":
        from .attention_rnn import AttentionRNN
        return AttentionRNN(**common)

    transformer = {
        "embedding_dim": config.get("embedding_dim", 512),
        "layers": config.get("num_layers", 4),
        "heads": config.get("num_heads", 8),
        "max_length": config.get("max_length", 128),
        "freeze_encoder": config.get("freeze_encoder", True),
        "resnet_path": config.get("resnet_path"),
    }
    if name == "transformer":
        from .transformer_captioner import GridTransformer
        return GridTransformer(
            vocab_size=vocab_size,
            **transformer,
            use_visual_encoder=config.get("use_visual_encoder", True),
            use_2d_position=config.get("use_2d_position", True),
        )
    if name == "graph_transformer":
        from .graph_captioner import GraphTransformer
        return GraphTransformer(
            vocab_size=vocab_size,
            **transformer,
            graph_layers=config.get("graph_layers", 2),
            use_visual_encoder=config.get("graph_use_visual_encoder", False),
        )
    if name == "vit_transformer":
        from .vit_transformer import ViTTransformer
        transformer["vit_path"] = config.get("vit_path")
        transformer.pop("resnet_path", None)
        return ViTTransformer(vocab_size=vocab_size, **transformer)
    raise ValueError(f"Unknown caption model: {name}")

def build_model(name: str, vocab_size: int, config: dict):
    common = {
        "vocab_size": vocab_size,
        "embedding_dim": config.get("embedding_dim", 512),
        "hidden_dim": config.get("hidden_dim", 512),
        "freeze_encoder": config.get("freeze_encoder", True),
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
    }
    if name == "transformer":
        from .transformer_captioner import GridTransformer
        return GridTransformer(vocab_size=vocab_size, **transformer)
    if name == "graph_transformer":
        from .graph_captioner import GraphTransformer
        return GraphTransformer(vocab_size=vocab_size, **transformer)
    if name == "vit_transformer":
        from .vit_transformer import ViTTransformer
        return ViTTransformer(vocab_size=vocab_size, **transformer)
    raise ValueError(f"Unknown caption model: {name}")

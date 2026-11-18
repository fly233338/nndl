"""使用 ViT 块特征和 Transformer 解码器生成描述。"""

from .common import TransformerDecoder, ViTPatches, nn


class ViTTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 512,
        layers: int = 4,
        heads: int = 8,
        max_length: int = 128,
        freeze_encoder: bool = True,
        vit_path: str | None = None,
        global_fusion: str = "none",
        dropout: float = 0.1,
        norm_first: bool = False,
        feedforward_dim: int = 2048,
        activation: str = "relu",
        tie_embeddings: bool = False,
        decoding: dict | None = None,
    ):
        super().__init__()
        if global_fusion not in ("none", "concat", "gated"):
            raise ValueError("global_fusion must be none, concat or gated")
        self.encoder = ViTPatches(freeze_encoder, vit_path, include_cls=global_fusion != "none")
        self.decoder = TransformerDecoder(
            vocab_size,
            self.encoder.output_dim,
            embedding_dim,
            layers,
            heads,
            max_length,
            use_visual_encoder=False,
            use_2d_position=False,
            global_fusion=global_fusion,
            dropout=dropout,
            norm_first=norm_first,
            feedforward_dim=feedforward_dim,
            activation=activation,
            tie_embeddings=tie_embeddings,
            decoding=decoding,
        )

    def forward(self, images, tokens):
        return self.decoder(self.encoder(images), tokens)

    def generate(self, images, bos_id, eos_id, max_length, beam_size=1):
        return self.decoder.generate(self.encoder(images), bos_id, eos_id, max_length, beam_size)

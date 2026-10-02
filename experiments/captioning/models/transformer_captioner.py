from .common import ResNetGrid, TransformerDecoder, nn


class GridTransformer(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 512, layers: int = 4, heads: int = 8, max_length: int = 128, freeze_encoder: bool = True, resnet_path: str | None = None):
        super().__init__()
        self.encoder = ResNetGrid(freeze_encoder, resnet_path)
        self.decoder = TransformerDecoder(vocab_size, self.encoder.output_dim, embedding_dim, layers, heads, max_length)

    def forward(self, images, tokens):
        return self.decoder(self.encoder(images), tokens)

    def generate(self, images, bos_id, eos_id, max_length, beam_size=1):
        return self.decoder.generate(self.encoder(images), bos_id, eos_id, max_length, beam_size)

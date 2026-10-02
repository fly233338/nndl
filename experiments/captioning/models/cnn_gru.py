from .common import GRUDecoder, ResNetGlobal, nn


class CNNGRU(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 512, hidden_dim: int = 512, freeze_encoder: bool = True, resnet_path: str | None = None):
        super().__init__()
        self.encoder = ResNetGlobal(freeze_encoder, resnet_path)
        self.decoder = GRUDecoder(vocab_size, self.encoder.output_dim, embedding_dim, hidden_dim)

    def forward(self, images, tokens):
        return self.decoder(self.encoder(images), tokens)

    def generate(self, images, bos_id, eos_id, max_length, beam_size=1):
        return self.decoder.generate(self.encoder(images), bos_id, eos_id, max_length, beam_size)

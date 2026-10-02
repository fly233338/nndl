from .common import ResNetGrid, TransformerDecoder, nn, torch


class GridGAT(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.q = nn.Linear(feature_dim, hidden_dim)
        self.k = nn.Linear(feature_dim, hidden_dim)
        self.v = nn.Linear(feature_dim, hidden_dim)
        self.output_dim = hidden_dim

    def forward(self, features):
        n = features.size(1)
        side = int(n ** 0.5)
        q, k, v = self.q(features), self.k(features), self.v(features)
        scores = torch.matmul(q, k.transpose(1, 2)) / (q.size(-1) ** 0.5)
        adjacency = torch.zeros((side, side), device=features.device, dtype=torch.bool)
        for row in range(side):
            for col in range(side):
                idx = row * side + col
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
                    rr, cc = row + dr, col + dc
                    if 0 <= rr < side and 0 <= cc < side:
                        adjacency[idx, rr * side + cc] = True
        scores = scores.masked_fill(~adjacency, float("-inf"))
        return torch.matmul(scores.softmax(-1), v)


class GraphTransformer(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 512, layers: int = 4, heads: int = 8, max_length: int = 128, freeze_encoder: bool = True, resnet_path: str | None = None):
        super().__init__()
        self.encoder = ResNetGrid(freeze_encoder, resnet_path)
        self.graph = GridGAT(self.encoder.output_dim, embedding_dim)
        self.decoder = TransformerDecoder(vocab_size, embedding_dim, embedding_dim, layers, heads, max_length)

    def forward(self, images, tokens):
        return self.decoder(self.graph(self.encoder(images)), tokens)

    def generate(self, images, bos_id, eos_id, max_length, beam_size=1):
        return self.decoder.generate(self.graph(self.encoder(images)), bos_id, eos_id, max_length, beam_size)

import math

from .common import ResNetGrid, TransformerDecoder, nn, torch


class _GraphAttentionLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim, bias=False)
        self.source_score = nn.Linear(output_dim, 1, bias=False)
        self.target_score = nn.Linear(output_dim, 1, bias=False)

    def forward(self, features, adjacency):
        projected = self.projection(features)
        scores = torch.nn.functional.leaky_relu(
            self.source_score(projected) + self.target_score(projected).transpose(1, 2), negative_slope=0.2
        )
        scores = scores.masked_fill(~adjacency, float("-inf"))
        return torch.matmul(scores.softmax(-1), projected)


class GridGAT(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 512, layers: int = 2):
        super().__init__()
        if layers < 1:
            raise ValueError("GAT layers must be positive")
        dimensions = [feature_dim] + [hidden_dim] * layers
        self.layers = nn.ModuleList([_GraphAttentionLayer(dimensions[i], dimensions[i + 1]) for i in range(layers)])
        self.output_dim = hidden_dim

    @staticmethod
    def _adjacency(n: int, device):
        side = int(math.sqrt(n))
        if side * side != n:
            raise ValueError(f"CNN grid must be square, got {n} tokens")
        adjacency = torch.zeros((n, n), device=device, dtype=torch.bool)
        for row in range(side):
            for col in range(side):
                index = row * side + col
                adjacency[index, index] = True
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
                    rr, cc = row + dr, col + dc
                    if 0 <= rr < side and 0 <= cc < side:
                        adjacency[index, rr * side + cc] = True
        return adjacency

    def forward(self, features):
        adjacency = self._adjacency(features.size(1), features.device).unsqueeze(0)
        output = features
        for index, layer in enumerate(self.layers):
            output = layer(output, adjacency)
            if index + 1 < len(self.layers):
                output = torch.relu(output)
        return output


class GraphTransformer(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 512, layers: int = 4, heads: int = 8, max_length: int = 128, freeze_encoder: bool = True, resnet_path: str | None = None, graph_layers: int = 2, use_visual_encoder: bool = False):
        super().__init__()
        self.encoder = ResNetGrid(freeze_encoder, resnet_path)
        self.graph = GridGAT(self.encoder.output_dim, embedding_dim, graph_layers)
        self.decoder = TransformerDecoder(vocab_size, embedding_dim, embedding_dim, layers, heads, max_length, use_visual_encoder=use_visual_encoder, use_2d_position=False)

    def forward(self, images, tokens):
        return self.decoder(self.graph(self.encoder(images)), tokens)

    def generate(self, images, bos_id, eos_id, max_length, beam_size=1):
        return self.decoder.generate(self.graph(self.encoder(images)), bos_id, eos_id, max_length, beam_size)

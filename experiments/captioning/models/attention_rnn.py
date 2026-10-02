from .common import AdditiveAttention, ResNetGrid, nn, torch


class AttentionRNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 512, hidden_dim: int = 512, freeze_encoder: bool = True, resnet_path: str | None = None):
        super().__init__()
        self.encoder = ResNetGrid(freeze_encoder, resnet_path)
        self.visual_attention = nn.MultiheadAttention(self.encoder.output_dim, 8, batch_first=True)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.feature_projection = nn.Linear(self.encoder.output_dim, hidden_dim)
        self.attention = AdditiveAttention(hidden_dim, hidden_dim)
        self.cell = nn.GRUCell(embedding_dim + hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.initial = nn.Linear(hidden_dim, hidden_dim)

    def _features(self, images):
        features = self.encoder(images)
        features, _ = self.visual_attention(features, features, features)
        return self.feature_projection(features)

    def forward(self, images, tokens):
        features = self._features(images)
        hidden = self.initial(features.mean(1))
        outputs = []
        for step in range(tokens.size(1)):
            context, _ = self.attention(features, hidden)
            hidden = self.cell(torch.cat([self.embedding(tokens[:, step]), context], -1), hidden)
            outputs.append(self.output(hidden).unsqueeze(1))
        return torch.cat(outputs, 1)

    @torch.no_grad()
    def generate(self, images, bos_id, eos_id, max_length, beam_size=1):
        features = self._features(images)
        hidden = self.initial(features.mean(1))
        current = torch.full((images.size(0),), bos_id, device=images.device, dtype=torch.long)
        outputs = [current.unsqueeze(1)]
        for _ in range(max_length - 1):
            context, _ = self.attention(features, hidden)
            hidden = self.cell(torch.cat([self.embedding(current), context], -1), hidden)
            current = self.output(hidden).argmax(-1)
            outputs.append(current.unsqueeze(1))
            if (current == eos_id).all():
                break
        return torch.cat(outputs, 1)

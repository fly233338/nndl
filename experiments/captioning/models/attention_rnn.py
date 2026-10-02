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

    def _step(self, features, hidden, token):
        context, _ = self.attention(features, hidden)
        hidden = self.cell(torch.cat([self.embedding(token), context], -1), hidden)
        return self.output(hidden), hidden

    @torch.no_grad()
    def generate(self, images, bos_id, eos_id, max_length, beam_size=1):
        features = self._features(images)
        if beam_size <= 1:
            hidden = self.initial(features.mean(1))
            current = torch.full((images.size(0),), bos_id, device=images.device, dtype=torch.long)
            outputs = [current.unsqueeze(1)]
            finished = torch.zeros(images.size(0), dtype=torch.bool, device=images.device)
            for _ in range(max_length - 1):
                logits, hidden = self._step(features, hidden, current)
                current = logits.argmax(-1)
                current = torch.where(finished, torch.full_like(current, eos_id), current)
                outputs.append(current.unsqueeze(1))
                finished |= current.eq(eos_id)
                if finished.all():
                    break
            return torch.cat(outputs, 1)

        results = []
        for index in range(images.size(0)):
            sample_features = features[index:index + 1]
            initial = self.initial(sample_features.mean(1))
            beams = [(torch.tensor([bos_id], device=images.device), initial, 0.0, False)]
            for _ in range(max_length - 1):
                candidates = []
                for sequence, hidden, score, finished in beams:
                    if finished:
                        candidates.append((sequence, hidden, score, True))
                        continue
                    logits, next_hidden = self._step(sample_features, hidden, sequence[-1:])
                    values, indices = logits.log_softmax(-1).topk(min(beam_size, logits.size(-1)), dim=-1)
                    for value, token_id in zip(values[0], indices[0]):
                        token = token_id.view(1)
                        candidates.append((torch.cat([sequence, token]), next_hidden.clone(), score + float(value), bool(token_id == eos_id)))
                candidates.sort(key=lambda item: item[2], reverse=True)
                beams = candidates[:beam_size]
                if all(item[3] for item in beams):
                    break
            results.append(beams[0][0])

        width = max(sequence.numel() for sequence in results)
        output = torch.full((len(results), width), eos_id, device=images.device, dtype=torch.long)
        for index, sequence in enumerate(results):
            output[index, :sequence.numel()] = sequence
        return output

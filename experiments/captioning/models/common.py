import math

import torch

nn = torch.nn


def load_resnet(pretrained: bool = True, model_path: str | None = None):
    if model_path:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_path, local_files_only=True)
        return model, 2048
    from torchvision.models import ResNet50_Weights, resnet50
    model = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
    return nn.Sequential(*list(model.children())[:-2]), 2048


class _FrozenModule(nn.Module):
    def _set_frozen(self, module: nn.Module, freeze: bool) -> None:
        self._freeze = freeze
        if freeze:
            for parameter in module.parameters():
                parameter.requires_grad = False
            module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "_freeze", False):
            getattr(self, self._frozen_name).eval()
        return self


class ResNetGlobal(_FrozenModule):
    def __init__(self, freeze: bool = True, model_path: str | None = None):
        super().__init__()
        self.features, self.output_dim = load_resnet(model_path=model_path)
        self._frozen_name = "features"
        self.huggingface = bool(model_path)
        self._set_frozen(self.features, freeze)

    def forward(self, images):
        fmap = self.features(images)
        if self.huggingface:
            fmap = fmap.last_hidden_state
        return fmap.mean(dim=(-2, -1))


class ResNetGrid(_FrozenModule):
    def __init__(self, freeze: bool = True, model_path: str | None = None):
        super().__init__()
        self.features, self.output_dim = load_resnet(model_path=model_path)
        self._frozen_name = "features"
        self.huggingface = bool(model_path)
        self._set_frozen(self.features, freeze)

    def forward(self, images):
        fmap = self.features(images)
        if self.huggingface:
            fmap = fmap.last_hidden_state
        return fmap.flatten(2).transpose(1, 2)


class ViTPatches(_FrozenModule):
    def __init__(self, freeze: bool = True, model_path: str | None = None):
        super().__init__()
        if model_path:
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(model_path, local_files_only=True)
            self.output_dim = int(self.model.config.hidden_size)
            self.huggingface = True
        else:
            from torchvision.models import ViT_B_16_Weights, vit_b_16
            self.model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
            self.output_dim = self.model.hidden_dim
            self.huggingface = False
        self._frozen_name = "model"
        self._set_frozen(self.model, freeze)

    def forward(self, images):
        if self.huggingface:
            tokens = self.model(pixel_values=images).last_hidden_state
        else:
            x = self.model._process_input(images)
            n = x.shape[0]
            x = torch.cat([self.model.class_token.expand(n, -1, -1), x], dim=1)
            tokens = self.model.encoder(x)
        return tokens[:, 1:]


class AdditiveAttention(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.feature = nn.Linear(feature_dim, hidden_dim)
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, features, hidden):
        scores = self.score(torch.tanh(self.feature(features) + self.hidden(hidden).unsqueeze(1))).squeeze(-1)
        weights = scores.softmax(dim=1)
        return (features * weights.unsqueeze(-1)).sum(dim=1), weights


def _best_beams(candidates, beam_size):
    candidates.sort(key=lambda item: item[2], reverse=True)
    return candidates[:beam_size]


class GRUDecoder(nn.Module):
    def __init__(self, vocab_size: int, feature_dim: int, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.init = nn.Linear(feature_dim, hidden_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, tokens):
        hidden = self.init(features).unsqueeze(0)
        outputs, _ = self.gru(self.embedding(tokens), hidden)
        return self.output(outputs)

    @torch.no_grad()
    def generate(self, features, bos_id: int, eos_id: int, max_length: int, beam_size: int = 1):
        if beam_size <= 1:
            hidden = self.init(features).unsqueeze(0)
            current = torch.full((features.size(0), 1), bos_id, device=features.device, dtype=torch.long)
            outputs = [current]
            finished = torch.zeros(features.size(0), dtype=torch.bool, device=features.device)
            for _ in range(max_length - 1):
                logits, hidden = self.gru(self.embedding(current[:, -1:]), hidden)
                next_token = self.output(logits[:, -1]).argmax(-1, keepdim=True)
                next_token = torch.where(finished.unsqueeze(1), torch.full_like(next_token, eos_id), next_token)
                outputs.append(next_token)
                current = torch.cat([current, next_token], dim=1)
                finished |= next_token[:, 0].eq(eos_id)
                if finished.all():
                    break
            return torch.cat(outputs, dim=1)

        results = []
        for sample in features:
            initial_hidden = self.init(sample.unsqueeze(0)).unsqueeze(0)
            beams = [(torch.tensor([[bos_id]], device=features.device), initial_hidden, 0.0, False)]
            for _ in range(max_length - 1):
                candidates = []
                for sequence, hidden, score, finished in beams:
                    if finished:
                        candidates.append((sequence, hidden, score, True))
                        continue
                    logits, next_hidden = self.gru(self.embedding(sequence[:, -1:]), hidden)
                    log_probs = self.output(logits[:, -1]).log_softmax(-1)
                    values, indices = torch.topk(log_probs, min(beam_size, log_probs.size(-1)), dim=-1)
                    for value, index in zip(values[0], indices[0]):
                        token = index.view(1, 1)
                        candidates.append((torch.cat([sequence, token], dim=1), next_hidden.clone(), score + float(value), bool(index == eos_id)))
                beams = _best_beams(candidates, beam_size)
                if all(item[3] for item in beams):
                    break
            results.append(beams[0][0].squeeze(0))

        width = max(sequence.numel() for sequence in results)
        output = torch.full((len(results), width), eos_id, device=features.device, dtype=torch.long)
        for index, sequence in enumerate(results):
            output[index, :sequence.numel()] = sequence
        return output


def _grid_position(n: int, dim: int, device, dtype):
    side = int(math.sqrt(n))
    if side * side != n:
        raise ValueError(f"CNN grid must be square, got {n} tokens")
    coordinates = torch.arange(side, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    half = math.ceil(dim / 4)
    frequencies = torch.exp(torch.arange(half, device=device, dtype=dtype) * (-math.log(10000.0) / half))
    parts = [torch.sin(xx.reshape(-1, 1) * frequencies), torch.cos(xx.reshape(-1, 1) * frequencies), torch.sin(yy.reshape(-1, 1) * frequencies), torch.cos(yy.reshape(-1, 1) * frequencies)]
    return torch.cat(parts, dim=1)[:, :dim].unsqueeze(0)


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size: int, feature_dim: int, embedding_dim: int, layers: int, heads: int, max_length: int = 128, use_visual_encoder: bool = True, use_2d_position: bool = False, pad_id: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.position = nn.Embedding(max_length, embedding_dim)
        self.input_projection = nn.Linear(feature_dim, embedding_dim)
        self.use_visual_encoder = use_visual_encoder
        self.use_2d_position = use_2d_position
        self.pad_id = pad_id
        if use_visual_encoder:
            encoder_layer = nn.TransformerEncoderLayer(embedding_dim, heads, batch_first=True)
            self.encoder = nn.TransformerEncoder(encoder_layer, layers)
        else:
            self.encoder = None
        decoder_layer = nn.TransformerDecoderLayer(embedding_dim, heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, layers)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def _encode(self, features):
        projected = self.input_projection(features)
        if self.use_2d_position:
            projected = projected + _grid_position(projected.size(1), projected.size(2), projected.device, projected.dtype)
        return self.encoder(projected) if self.encoder is not None else projected

    def _decode(self, memory, tokens):
        positions = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0)
        embedded = self.embedding(tokens) + self.position(positions)
        mask = torch.ones((tokens.size(1), tokens.size(1)), device=tokens.device, dtype=torch.bool).triu(1)
        return self.output(self.decoder(embedded, memory, tgt_mask=mask, tgt_key_padding_mask=tokens.eq(self.pad_id)))

    def forward(self, features, tokens):
        return self._decode(self._encode(features), tokens)

    @torch.no_grad()
    def generate(self, features, bos_id: int, eos_id: int, max_length: int, beam_size: int = 1):
        if beam_size <= 1:
            tokens = torch.full((features.size(0), 1), bos_id, device=features.device, dtype=torch.long)
            finished = torch.zeros(features.size(0), dtype=torch.bool, device=features.device)
            for _ in range(max_length - 1):
                next_token = self.forward(features, tokens)[:, -1].argmax(-1, keepdim=True)
                next_token = torch.where(finished.unsqueeze(1), torch.full_like(next_token, eos_id), next_token)
                tokens = torch.cat([tokens, next_token], dim=1)
                finished |= next_token[:, 0].eq(eos_id)
                if finished.all():
                    break
            return tokens

        results = []
        for index in range(features.size(0)):
            sample_features = features[index:index + 1]
            beams = [(torch.tensor([[bos_id]], device=features.device), 0.0, False)]
            for _ in range(max_length - 1):
                candidates = []
                for sequence, score, finished in beams:
                    if finished:
                        candidates.append((sequence, score, True))
                        continue
                    log_probs = self.forward(sample_features, sequence)[:, -1].log_softmax(-1)
                    values, indices = torch.topk(log_probs, min(beam_size, log_probs.size(-1)), dim=-1)
                    for value, token_id in zip(values[0], indices[0]):
                        token = token_id.view(1, 1)
                        candidates.append((torch.cat([sequence, token], dim=1), score + float(value), bool(token_id == eos_id)))
                candidates.sort(key=lambda item: item[1], reverse=True)
                beams = candidates[:beam_size]
                if all(item[2] for item in beams):
                    break
            results.append(beams[0][0].squeeze(0))

        width = max(sequence.numel() for sequence in results)
        output = torch.full((len(results), width), eos_id, device=features.device, dtype=torch.long)
        for index, sequence in enumerate(results):
            output[index, :sequence.numel()] = sequence
        return output

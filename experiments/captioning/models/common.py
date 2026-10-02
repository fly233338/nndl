import torch
nn = torch.nn


def load_resnet(pretrained: bool = True, model_path: str | None = None):
    if model_path:
        from transformers import AutoModel

        return AutoModel.from_pretrained(model_path, local_files_only=True), 2048
    from torchvision.models import ResNet50_Weights, resnet50
    model = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
    features = nn.Sequential(*list(model.children())[:-2])
    return features, 2048


class ResNetGlobal(nn.Module):
    def __init__(self, freeze: bool = True, model_path: str | None = None):
        super().__init__()
        self.features, self.output_dim = load_resnet(model_path=model_path)
        self.huggingface = bool(model_path)
        if freeze:
            for parameter in self.features.parameters():
                parameter.requires_grad = False

    def forward(self, images):
        fmap = self.features(images)
        if self.huggingface:
            fmap = fmap.last_hidden_state
        return fmap.mean(dim=(-2, -1))


class ResNetGrid(nn.Module):
    def __init__(self, freeze: bool = True, model_path: str | None = None):
        super().__init__()
        self.features, self.output_dim = load_resnet(model_path=model_path)
        self.huggingface = bool(model_path)
        if freeze:
            for parameter in self.features.parameters():
                parameter.requires_grad = False

    def forward(self, images):
        fmap = self.features(images)
        if self.huggingface:
            fmap = fmap.last_hidden_state
        return fmap.flatten(2).transpose(1, 2)


class ViTPatches(nn.Module):
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
        if freeze:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

    def forward(self, images):
        if self.huggingface:
            return self.model(pixel_values=images).last_hidden_state
        x = self.model._process_input(images)
        n = x.shape[0]
        batch_class = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class, x], dim=1)
        return self.model.encoder(x)


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
        batch = features.size(0)
        hidden = self.init(features).unsqueeze(0)
        current = torch.full((batch, 1), bos_id, device=features.device, dtype=torch.long)
        outputs = [current]
        for _ in range(max_length - 1):
            logits, hidden = self.gru(self.embedding(current[:, -1:]), hidden)
            next_token = self.output(logits[:, -1]).argmax(-1, keepdim=True)
            outputs.append(next_token)
            current = torch.cat([current, next_token], dim=1)
            if (next_token == eos_id).all():
                break
        return torch.cat(outputs, dim=1)


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size: int, feature_dim: int, embedding_dim: int, layers: int, heads: int, max_length: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position = nn.Embedding(max_length, embedding_dim)
        self.input_projection = nn.Linear(feature_dim, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(embedding_dim, heads, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(embedding_dim, heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, layers)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, features, tokens):
        memory = self.encoder(self.input_projection(features))
        positions = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0)
        embedded = self.embedding(tokens) + self.position(positions)
        mask = nn.Transformer.generate_square_subsequent_mask(tokens.size(1), device=tokens.device)
        return self.output(self.decoder(embedded, memory, tgt_mask=mask))

    @torch.no_grad()
    def generate(self, features, bos_id: int, eos_id: int, max_length: int, beam_size: int = 1):
        tokens = torch.full((features.size(0), 1), bos_id, device=features.device, dtype=torch.long)
        for _ in range(max_length - 1):
            next_token = self.forward(features, tokens)[:, -1].argmax(-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
            if (next_token == eos_id).all():
                break
        return tokens

import re
from collections import Counter
from pathlib import Path


class Vocabulary:
    specials = ("<pad>", "<bos>", "<eos>", "<unk>")

    def __init__(self, token_to_id: dict[str, int] | None = None):
        self.token_to_id = token_to_id or {token: i for i, token in enumerate(self.specials)}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*|[^\w\s]", text.lower())

    @classmethod
    def build(cls, captions: list[str], min_freq: int = 1) -> "Vocabulary":
        counts = Counter(token for caption in captions for token in cls.tokenize(caption))
        vocab = cls()
        for token, count in sorted(counts.items()):
            if count >= min_freq and token not in vocab.token_to_id:
                vocab.token_to_id[token] = len(vocab.token_to_id)
        vocab.id_to_token = {i: token for token, i in vocab.token_to_id.items()}
        return vocab

    def encode(self, text: str, max_length: int) -> list[int]:
        ids = [self.token_to_id["<bos>"]]
        ids.extend(self.token_to_id.get(token, self.token_to_id["<unk>"]) for token in self.tokenize(text))
        ids.append(self.token_to_id["<eos>"])
        return ids[:max_length]

    def decode(self, ids: list[int]) -> str:
        tokens = []
        for idx in ids:
            token = self.id_to_token.get(int(idx), "<unk>")
            if token == "<eos>":
                break
            if token not in self.specials:
                tokens.append(token)
        text = " ".join(tokens)
        return re.sub(r"\s+([,.!?;:])", r"\1", text).strip()

    def save(self, path: str | Path) -> None:
        import json
        Path(path).write_text(json.dumps(self.token_to_id, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        import json
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

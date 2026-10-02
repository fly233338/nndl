import re
from collections import Counter
from pathlib import Path


_CJK = re.compile(r"[\u3400-\u9fff]")
_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*|[\u3400-\u9fff]|[^\w\s]")


class Vocabulary:
    specials = ("<pad>", "<bos>", "<eos>", "<unk>")

    def __init__(self, token_to_id: dict[str, int] | None = None):
        self.token_to_id = token_to_id or {token: i for i, token in enumerate(self.specials)}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return _TOKEN.findall(text.lower())

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
        if max_length < 2:
            raise ValueError("max_length must leave room for BOS and EOS")
        body = [self.token_to_id.get(token, self.token_to_id["<unk>"]) for token in self.tokenize(text)]
        body = body[: max_length - 2]
        return [self.token_to_id["<bos>"], *body, self.token_to_id["<eos>"]]

    def decode(self, ids: list[int]) -> str:
        tokens = []
        for idx in ids:
            token = self.id_to_token.get(int(idx), "<unk>")
            if token == "<eos>":
                break
            if token not in self.specials:
                tokens.append(token)
        text = ""
        previous = ""
        punctuation = set(",.!?;:%)]}，。！？；：、）】》")
        opening = set("([{【《“‘")
        for token in tokens:
            if not text:
                text = token
            elif _CJK.fullmatch(token) or token in punctuation:
                text += token
            elif _CJK.search(previous):
                text += " " + token
            elif previous in opening:
                text += token
            else:
                text += " " + token
            previous = token
        return text.strip()

    def save(self, path: str | Path) -> None:
        import json
        Path(path).write_text(json.dumps(self.token_to_id, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        import json
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

import json
from pathlib import Path


class JSONTokenizer:
    specials = ("<pad>", "<bos>", "<eos>", "<unk>")

    def __init__(self, token_to_id: dict[str, int] | None = None):
        self.token_to_id = token_to_id or {token: index for index, token in enumerate(self.specials)}
        self.id_to_token = {index: token for token, index in self.token_to_id.items()}

    @classmethod
    def build(cls, texts: list[str]):
        tokenizer = cls()
        for text in texts:
            for token in text:
                if token not in tokenizer.token_to_id:
                    tokenizer.token_to_id[token] = len(tokenizer.token_to_id)
        tokenizer.id_to_token = {index: token for token, index in tokenizer.token_to_id.items()}
        return tokenizer

    def encode(self, text: str, max_length: int) -> list[int]:
        if max_length < 2:
            raise ValueError("max_length must leave room for BOS and EOS")
        body = [self.token_to_id.get(token, self.token_to_id["<unk>"]) for token in text][: max_length - 2]
        return [self.token_to_id["<bos>"], *body, self.token_to_id["<eos>"]]

    def decode(self, ids: list[int]) -> str:
        output = []
        for index in ids:
            token = self.id_to_token.get(int(index), "<unk>")
            if token == "<eos>":
                break
            if token not in self.specials:
                output.append(token)
        return "".join(output)

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(self.token_to_id, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

from importlib.metadata import version


DEFAULT_METRICS = ("BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr-D", "SPICE")


def metric_metadata():
    return {"pycocoevalcap": version("pycocoevalcap"), "tokenizer": "PTBTokenizer", "scale": "raw (no x100); CIDEr-D uses scorer x10", "cider_sigma": 6.0}


def evaluate_caption_records(records: list[dict], metrics: tuple[str, ...] | list[str] | None = None) -> dict[str, float]:
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    selected = tuple(DEFAULT_METRICS if metrics is None else metrics)
    unknown = set(selected) - set(DEFAULT_METRICS)
    if unknown:
        raise ValueError(f"Unknown caption metrics: {sorted(unknown)}")
    if not records:
        raise ValueError("Cannot evaluate an empty prediction set")
    references = {index: [{"caption": caption} for caption in row["references"]] for index, row in enumerate(records)}
    predictions = {index: [{"caption": row["prediction"]}] for index, row in enumerate(records)}
    tokenizer = PTBTokenizer()
    references = tokenizer.tokenize(references)
    predictions = tokenizer.tokenize(predictions)
    result = {}
    if any(name.startswith("BLEU") for name in selected):
        from pycocoevalcap.bleu.bleu import Bleu
        scores, _ = Bleu(4).compute_score(references, predictions)
        for index, name in enumerate(("BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4")):
            if name in selected:
                result[name] = float(scores[index])
    if "METEOR" in selected:
        from pycocoevalcap.meteor.meteor import Meteor
        scorer = Meteor()
        try:
            result["METEOR"] = float(scorer.compute_score(references, predictions)[0])
        finally:
            del scorer
    if "ROUGE-L" in selected:
        from pycocoevalcap.rouge.rouge import Rouge
        result["ROUGE-L"] = float(Rouge().compute_score(references, predictions)[0])
    if "CIDEr-D" in selected:
        from pycocoevalcap.cider.cider import Cider
        result["CIDEr-D"] = float(Cider().compute_score(references, predictions)[0])
    if "SPICE" in selected:
        from pycocoevalcap.spice.spice import Spice
        result["SPICE"] = float(Spice().compute_score(references, predictions)[0])
    result["num_samples"] = len(records)
    return result

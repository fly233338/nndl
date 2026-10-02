from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice


def evaluate_caption_records(records: list[dict]) -> dict[str, float]:
    references = {index: [{"caption": caption} for caption in row["references"]] for index, row in enumerate(records)}
    predictions = {index: [{"caption": row["prediction"]}] for index, row in enumerate(records)}
    bleu_scores, _ = Bleu(4).compute_score(references, predictions)
    meteor_score, _ = Meteor().compute_score(references, predictions)
    rouge_score, _ = Rouge().compute_score(references, predictions)
    cider_score, _ = Cider().compute_score(references, predictions)
    spice_score, _ = Spice().compute_score(references, predictions)
    return {
        "BLEU-1": float(bleu_scores[0]),
        "BLEU-2": float(bleu_scores[1]),
        "BLEU-3": float(bleu_scores[2]),
        "BLEU-4": float(bleu_scores[3]),
        "METEOR": float(meteor_score),
        "ROUGE-L": float(rouge_score),
        "CIDEr-D": float(cider_score),
        "SPICE": float(spice_score),
        "num_samples": len(records),
    }

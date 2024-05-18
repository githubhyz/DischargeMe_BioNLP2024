#ref: https://github.com/Stanford-AIMI/discharge-me/tree/main/scoring

import numpy as np
import evaluate
from scoring.UMLSScorer import UMLSScorer
from scoring.align import AlignScorer
from scoring.bertscore import BertScore
from scoring.bleu import Bleu
from scoring.rouge import Rouge

metrics=["bleu", "rouge", "bertscore", "meteor", "align", "medcon"]

bleuScorer = Bleu()
rougeScorer = Rouge(["rouge1", "rouge2", "rougeL"])
bertScorer = BertScore()
meteorScorer = evaluate.load("meteor")
alignScorer = AlignScorer()
medconScorer = UMLSScorer(quickumls_fp="quickumls/")

def compute_overall_score(rows_ref, rows_gen, metrics=metrics):
    leaderboard = {}

    if "bleu" in metrics:
        temp = bleuScorer(refs=rows_ref, hyps=rows_gen)
        leaderboard["bleu"] = np.mean(temp)

    if "rouge" in metrics:
        temp = rougeScorer(refs=rows_ref, hyps=rows_gen)
        leaderboard["rouge1"] = np.mean(temp["rouge1"])
        leaderboard["rouge2"] = np.mean(temp["rouge2"])
        leaderboard["rougeL"] = np.mean(temp["rougeL"])

    if "bertscore" in metrics:
        temp = bertScorer(refs=rows_ref, hyps=rows_gen)
        leaderboard["bertscore"] = np.mean(temp)

    if "meteor" in metrics:
        temp = meteorScorer.compute(references=rows_ref, predictions=rows_gen)
        leaderboard["meteor"] = np.mean(temp["meteor"])

    if "align" in metrics:
        temp = alignScorer(refs=rows_ref, hyps=rows_gen)
        leaderboard["align"] = np.mean(temp)

    if "medcon" in metrics:
        temp = medconScorer(rows_ref, rows_gen)
        leaderboard["medcon"] = np.mean(temp)

    overall_score = np.mean(list(leaderboard.values()))
    leaderboard["overall"] = overall_score

    return leaderboard
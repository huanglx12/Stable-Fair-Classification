from fairness.algorithms.zafar.ZafarRegAlgorithm import ZafarRegAlgorithm
from fairness.algorithms.kamishima.KamishimaRegAlgorithm import KamishimaRegAlgorithm
from fairness.algorithms.ParamGridSearch import ParamGridSearch
from fairness.algorithms.GoelReg.GoelRegAlgorithm import GoelRegAlgorithmFairness

from fairness.metrics.DIAvgAll import DIAvgAll
from fairness.metrics.Accuracy import Accuracy
from fairness.metrics.MCC import MCC


ALGORITHMS = [
     GoelRegAlgorithmFairness(),                                    # GoelReg
     KamishimaRegAlgorithm(),                                       # KamishimaReg
     ZafarRegAlgorithm(),                                           # ZafarReg
#    ParamGridSearch(GoelRegAlgorithm(), Accuracy()),                # Grid search of GoelReg to find the best parameter
#    ParamGridSearch(KamishimaRegAlgorithm(), Accuracy()),
#    ParamGridSearch(KamishimaRegAlgorithm(), DIAvgAll()),
#    ParamGridSearch(ZafarRegAlgorithm(), Accuracy())
]

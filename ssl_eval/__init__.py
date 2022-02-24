from .offline_eval import OfflineEvaluator
from .online_eval import OnlineEvaluator


class Evaluator(OfflineEvaluator):

    def __init__(self, *args, **kwargs):
        super(Evaluator, self).__init__(*args, **kwargs)
        print(f"WARNING: Evalautor class is deprecated. Call OfflineEvalautor instead.")

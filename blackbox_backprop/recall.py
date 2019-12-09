from collections import deque

import torch

from .ranking import TrueRanker, rank_normalised


class RecallLoss(torch.nn.Module):
    def __init__(self, lambda_val, margin, weight_fn):
        """
        Torch module for computing recall-based loss as in "Blackbox differentiation of Ranking-based Metrics"
        :param lambda_val:  hyperparameter of black-box backprop
        :param margin: margin to be enforced between positives and negatives (alpha in the paper)
        :param weight_fn: callable torch.Tensor -> torch.Tensor (such as log(1+x) or log(1+log(1+x)) etc.)
        """
        super().__init__()
        self.sorter = TrueRanker()
        self.margin = margin
        self.lambda_val = lambda_val
        self.weight_fn = weight_fn

    def forward(self, score_sequences, gt_relevance_sequences):
        """
        :param score_sequences: [num_sequences, len_of_sequence] scores of images (floats in [-1,1])
        :param gt_relevance_sequences: [num_sequences, len_of_sequence] of booleans relevant/irrelevant
        """
        HIGH_CONSTANT = 2.0  # This is actually high enough as normalised ranks live in [0,1].
        TINY_CONSTANT = 1e-5
        length = score_sequences.shape[0]
        device = score_sequences.device

        deviations = (gt_relevance_sequences - 0.5).to(device)
        score_sequences = score_sequences - self.margin * deviations

        ranks_among_all = TrueRanker.apply(score_sequences, self.lambda_val)
        scores_among_positive = -ranks_among_all + HIGH_CONSTANT * gt_relevance_sequences
        scores_among_positive = scores_among_positive.to(device)
        ranks_among_positive = rank_normalised(scores_among_positive)
        ranks_among_positive.require_grad = False

        ranks_for_queries = (ranks_among_all - ranks_among_positive) * gt_relevance_sequences

        assert torch.all(ranks_for_queries > -TINY_CONSTANT)

        # denormalize ranks
        ranks_for_queries = ranks_for_queries * length
        recall = self.weight_fn(ranks_for_queries * gt_relevance_sequences).sum() / gt_relevance_sequences.sum()
        return recall


class BatchMemoryRecallLoss(torch.nn.Module):
    """
    A wrapper around a rank-based loss that allows batch memory.
    """
    def __init__(self, batch_memory, **kwargs):
        """
        :param batch_memory: How many batches should be in memory
        :param kwargs: arguments of the underlying loss
        """
        super().__init__()
        self.batch_memory = batch_memory
        self.loss = RecallLoss(**kwargs)

        self.batch_storage = deque()
        self.labels_storage = deque()

    def reset(self):
        self.batch_storage.clear()
        self.labels_storage.clear()


    def forward(self, score_sequences, gt_relevance_sequences):
        if self.batch_memory > 0:
            all_score_sequences = torch.cat((score_sequences,) + tuple(self.batch_storage), dim=0)
            all_relevance_sequences = torch.cat((gt_relevance_sequences,) + tuple(self.labels_storage), dim=0)
            result = self.loss(all_score_sequences, all_relevance_sequences)

            if len(self.batch_storage) == self.batch_memory:
                self.batch_storage.popleft()

            self.batch_storage.append(score_sequences.detach())

            if len(self.labels_storage) == self.batch_memory:
                self.labels_storage.popleft()

            self.labels_storage.append(gt_relevance_sequences.detach())
        else:
            result = self.loss(score_sequences, gt_relevance_sequences)
        return result

from functools import partial
import itertools

import blossom_v
import numpy as np
import torch

from .utils import maybe_parallelize


class MinCostPerfectMatchingSolver(torch.autograd.Function):
    """
    Torch module implementing Blossom V (Kolmogorov, 2009) algorithm to find the min-cost perfect matching
    between nodes on a graph given.
    """

    @staticmethod
    def forward(ctx, weights, lambda_val, num_vertices, edges):
        """
        :param ctx: context for backpropagation
        :param weights: torch.Tensor [batch_size, len(edges)] - suggested edge weights
        :param lambda_val (float): hyperparameter lambda
        :param num_vertices (int): number of vertices os the graph
        :param edges (list of pairs): list of graph edges
        :return solution torch.Tensor [batch_size, len(edges)] - indicator vectors of selected edges
        """
        ctx.weights = weights.detach().cpu().numpy()
        ctx.lambda_val = lambda_val
        ctx.solver = partial(min_cost_perfect_matching, edges=edges, num_vertices=num_vertices)
        ctx.perfect_matchings = np.array(maybe_parallelize(ctx.solver, list(ctx.weights)))

        return torch.from_numpy(ctx.perfect_matchings).float().to(weights.device)

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.shape == ctx.perfect_matchings.shape
        device = grad_output.device
        grad_output = grad_output.cpu().numpy()

        weights_prime = np.maximum(ctx.weights + ctx.lambda_val * grad_output, 0.0)
        better_matchings = np.array(maybe_parallelize(ctx.solver, list(weights_prime)))

        gradient = -(ctx.perfect_matchings - better_matchings) / ctx.lambda_val
        return torch.from_numpy(gradient).to(device), None


def min_cost_perfect_matching(edges, edge_weights, num_vertices):
    """
    Blossom V (Kolmogorov, 2009) algorithm to find the min-cost perfect matching between nodes on a graph
    given as a matrix of edge weights.
    :param edges (list of pairs): edges of the graph
    :param edge_weights (np.ndarray shape: [len(edges)]): vector where the i-th element is the edge weight of edge i
    :param num_vertices (int): total number of vertices in the graph
    :return: solution (np.ndarray shape: [len(edges)]) as an indicator vector of selected edges
    """

    edges = tuple(map(tuple, edges))  # Make hashable
    pm = blossom_v.PerfectMatching(num_vertices, len(edges))

    for (v1, v2), w in zip(edges, edge_weights):
        pm.AddEdge(int(v1), int(v2), float(w))

    pm.Solve()

    edge_to_index_dict = dict(zip(edges, itertools.count()))
    unique_matched_edges = [(v, pm.GetMatch(v)) for v in range(num_vertices) if v < pm.GetMatch(v)]
    indices = [edge_to_index_dict[edge] for edge in unique_matched_edges]
    solution = np.zeros(len(edges)).astype(np.float32)
    solution[indices] = 1
    return solution

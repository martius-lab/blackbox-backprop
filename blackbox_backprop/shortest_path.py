import heapq
import itertools
from functools import partial

import numpy as np
import torch

from .utils import maybe_parallelize


def neighbours_fn(x, y, x_max, y_max):
    """
    Returns all 8 neighbours of a given coordinate on a grid.
    """
    deltas_x = (-1, 0, 1)
    deltas_y = (-1, 0, 1)
    for (dx, dy) in itertools.product(deltas_x, deltas_y):
        x_new, y_new = x + dx, y + dy
        if 0 <= x_new < x_max and 0 <= y_new < y_max and (dx, dy) != (0, 0):
            yield x_new, y_new


def dijkstra(matrix):
    """
    Implementation of Dijkstra algorithm to find the (s,t)-shortest path between top-left and bottom-right nodes
    on a nxn grid graph (with 8-neighbourhood).
    NOTE: This is an vertex variant of the problem, i.e. nodes carry weights, not edges.
    :param matrix (np.ndarray [grid_dim, grid_dim]): Matrix of node-costs.
    :return: matrix (np.ndarray [grid_dim, grid_dim]), indicator matrix of nodes on the shortest path.
    """

    x_max, y_max = matrix.shape
    neighbors_func = partial(neighbours_fn, x_max=x_max, y_max=y_max)

    costs = np.full_like(matrix, 1.0e10)
    costs[0][0] = matrix[0][0]
    num_path = np.zeros_like(matrix)
    num_path[0][0] = 1
    priority_queue = [(matrix[0][0], (0, 0))]
    certain = set()
    transitions = dict()

    while priority_queue:
        cur_cost, (cur_x, cur_y) = heapq.heappop(priority_queue)
        if (cur_x, cur_y) in certain:
            pass

        for x, y in neighbors_func(cur_x, cur_y):
            if (x, y) not in certain:
                if matrix[x][y] + costs[cur_x][cur_y] < costs[x][y]:
                    costs[x][y] = matrix[x][y] + costs[cur_x][cur_y]
                    heapq.heappush(priority_queue, (costs[x][y], (x, y)))
                    transitions[(x, y)] = (cur_x, cur_y)
                    num_path[x, y] = num_path[cur_x, cur_y]
                elif matrix[x][y] + costs[cur_x][cur_y] == costs[x][y]:
                    num_path[x, y] += 1

        certain.add((cur_x, cur_y))
    # retrieve the path
    cur_x, cur_y = x_max - 1, y_max - 1
    on_path = np.zeros_like(matrix)
    on_path[-1][-1] = 1
    while (cur_x, cur_y) != (0, 0):
        cur_x, cur_y = transitions[(cur_x, cur_y)]
        on_path[cur_x, cur_y] = 1.0
    return on_path


class ShortestPath(torch.autograd.Function):
    """
    torch module calculating the solution of the shortest path problem from top-left to bottom-right
    on a given grid graph.
    """

    @staticmethod
    def forward(ctx, weights, lambda_val):
        """
        :param ctx: context for backpropagation
        :param weights (torch.Tensor of shape [batch_size, grid_dim, grid_dim]): vertex weights
        :param lambda_val: hyperparameter lambda
        :return: shortest paths (torch.Tensor of shape [batch_size, grid_dim, grid_dim]): indicator matrices
        of taken paths
        """
        ctx.weights = weights.detach().cpu().numpy()
        ctx.lambda_val = lambda_val
        ctx.suggested_tours = np.asarray(maybe_parallelize(dijkstra, arg_list=list(ctx.weights)))
        return torch.from_numpy(ctx.suggested_tours).float().to(weights.device)

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.shape == ctx.suggested_tours.shape
        grad_output_numpy = grad_output.detach().cpu().numpy()
        weights_prime = np.maximum(ctx.weights + ctx.lambda_val * grad_output_numpy, 0.0)
        better_paths = np.asarray(maybe_parallelize(dijkstra, arg_list=list(weights_prime)))
        gradient = -(ctx.suggested_tours - better_paths) / ctx.lambda_val
        return torch.from_numpy(gradient).to(grad_output.device), None

import numpy as np
import torch

try:
    from gurobipy import GRB, Model, quicksum
except ImportError:
    print("GurobiPy missing, TSP module not available")

from .utils import maybe_parallelize


def subtour(n, edges):
    """
    Given a list of edges, finds the shortest subtour.
    """
    visited = [False] * n
    cycles = []
    lengths = []
    selected = [[] for _ in range(n)]
    for x, y in edges:
        selected[x].append(y)
    while True:
        current = visited.index(False)
        thiscycle = [current]
        while True:
            visited[current] = True
            neighbors = [x for x in selected[current] if not visited[x]]
            if len(neighbors) == 0:
                break
            current = neighbors[0]
            thiscycle.append(current)
        cycles.append(thiscycle)
        lengths.append(len(thiscycle))
        if sum(lengths) == n:
            break
    return cycles[lengths.index(min(lengths))]


def subtourelim(n, model, where):
    """
    Callback - use lazy constraints to eliminate sub-tours.
    """
    if where == GRB.callback.MIPSOL:
        selected = []

        # make a list of edges selected in the solution
        for i in range(n):
            sol = model.cbGetSolution([model._vars[i, j] for j in range(n)])
            selected += [(i, j) for j in range(n) if sol[j] > 0.5]

        # find the shortest cycle in the selected edge list
        tour = subtour(n, selected)
        if len(tour) < n:
            # add a subtour elimination constraint
            expr = 0
            for i in range(len(tour)):
                for j in range(i + 1, len(tour)):
                    expr += model._vars[tour[i], tour[j]]
            model.cbLazy(expr <= len(tour) - 1)


def gurobi_tsp(distance_matrix):
    """
    Solves tsp problem.
    :param distance_matrix: symmetric matrix of distances, where the i,j element is the distance between object i and j
    :return: matrix containing {0, 1}, 1 for each transition that is included in the tsp solution
    """
    n = len(distance_matrix)
    m = Model()
    m.setParam("OutputFlag", False)
    m.setParam("Threads", 1)

    # Create variables
    vars = {}
    for i in range(n):
        for j in range(i + 1):
            vars[i, j] = m.addVar(
                obj=0.0 if i == j else distance_matrix[i][j], vtype=GRB.BINARY, name="e" + str(i) + "_" + str(j)
            )
            vars[j, i] = vars[i, j]
        m.update()

    # Add degree-2 constraint, and forbid loops
    for i in range(n):
        m.addConstr(quicksum(vars[i, j] for j in range(n)) == 2)
        vars[i, i].ub = 0
    m.update()

    # Optimize model
    m._vars = vars
    m.params.LazyConstraints = 1

    def subtour_fn(model, where):
        return subtourelim(n, model, where)

    m.optimize(subtour_fn)
    solution = m.getAttr("x", vars)
    selected = [(i, j) for i in range(n) for j in range(n) if solution[i, j] > 0.5]
    result = np.zeros_like(distance_matrix)
    for (i, j) in selected:
        result[i][j] = 1

    return result


class TspSolver(torch.autograd.Function):
    """
    Torch module calculating the solution of the travelling salesman problem on a given distance matrix
    using a Gurobi implementation of a cutting plane algorithm.
    """

    @staticmethod
    def forward(ctx, distance_matrices, lambda_val):
        """
        distance_matrices: torch.Tensor of shape [batch_size, num_vertices, num_vertices]
        return: torch.Tenspr of shape [batch_size, num_vertices, num_vertices] 0-1 indicator matrices of the solution
        """
        ctx.distance_matrices = distance_matrices.detach().cpu().numpy()
        ctx.lambda_val = lambda_val
        suggested_tours = np.asarray(maybe_parallelize(gurobi_tsp, arg_list=list(ctx.distance_matrices)))
        return torch.from_numpy(suggested_tours).float().to(distance_matrices.device)

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.shape == ctx.suggested_tours.shape
        grad_output_numpy = grad_output.detach().cpu().numpy()
        distances_prime = ctx.distance_matrices + ctx.lambda_val * grad_output_numpy
        better_tours = np.array(maybe_parallelize(gurobi_tsp, arg_list=list(distances_prime)))
        gradient = -(ctx.suggested_tours - better_tours) / ctx.lambda_val
        return torch.from_numpy(gradient.astype(np.float32)).to(grad_output.device), None

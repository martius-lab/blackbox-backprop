from visualization_utils import BlackboxSolverAbstract, gen_w_and_y_grad, gen_edges

try:
    from lpmp_py import gm_solver
    from lpmp_py import mgm_solver
except ImportError:
    print("lpmp_py missing. Install it separately for using (multi)graph matching solvers")

from blackbox_backprop.travelling_salesman import gurobi_tsp
from blackbox_backprop.shortest_path import dijkstra

import numpy as np
import itertools as it

class GraphMatchingBBSolver(BlackboxSolverAbstract):
    """
    Graph matching solver.
    """

    @staticmethod
    def solver(inputs, edges_left, edges_right, solver_params):
        unary_costs, quadratic_costs = inputs
        m, m_q = gm_solver(unary_costs, quadratic_costs, edges_left, edges_right, solver_params, verbose=False)
        return m, m_q

    @staticmethod
    def gen_input(num_nodes_l, num_edges_l, seed, w_normal_factor=1, w_normal_addend=0, w_shift_factor=1, 
                  w_shift_addend=0, y_factor=1, y_addend=0, directed_edges=False):
        w_slice_l, y_grad_l = gen_w_and_y_grad(
            seed=seed,
            # Choose a different normal_factor, normal_addend, shift_factor, shift_addend to specify a different cut
            params=[dict(shape=num_nodes_l,
                         w_slice_par=dict(mode='slice_random', normal_factor=w_normal_factor, normal_addend=w_normal_addend, 
                                          shift_factor=w_shift_factor, shift_addend=w_shift_addend),
                         y_grad_par=dict(mode='hamming_random', factor=y_factor, addend=y_addend)),  # unary costs
                    dict(shape=num_edges_l,
                         w_slice_par=dict(mode='const_random', normal_factor=w_normal_factor, normal_addend=w_normal_addend, 
                                          shift_factor=w_shift_factor, shift_addend=w_shift_addend),
                         y_grad_par=dict(mode='zero'))])  # quadratic costs

        edges_left, edges_right = [gen_edges(nn, ne, directed=directed_edges) for nn, ne in
                                   zip(num_nodes_l, num_edges_l)]
        solver_params = {
            'timeout': 100,
            'primalComputationInterval': 10,
            'maxIter': 1000000,
            'graphMatchingRounding': 'mcf',
            'graphMatchingFrankWolfeIterations': 50
        }
        solver_config = dict(edges_left=edges_left, edges_right=edges_right, solver_params=solver_params)
        return w_slice_l, y_grad_l, solver_config


class MultiGraphMatchingBBSolver(BlackboxSolverAbstract):
    """
    Multigraph matching solver.
    """

    @staticmethod
    def solver(inputs, edges, solver_params):
        l = len(inputs) // 2
        unary_costs_l, quadratic_costs_l = inputs[:l], inputs[l:]
        m_l, m_q_l = mgm_solver(unary_costs_l, quadratic_costs_l, edges, solver_params, verbose=False)
        return m_l + m_q_l

    @staticmethod
    def gen_input(num_nodes_l, num_edges_l, seed, w_normal_factor=1, w_normal_addend=0, w_shift_factor=1, w_shift_addend=0, 
                  y_factor=1, y_addend=0, directed_edges=False):
        unary_shapes = [(i, j) for i, j in it.combinations(num_nodes_l, 2)]
        quadratic_shapes = [(i, j) for i, j in it.combinations(num_edges_l, 2)]

        params_unary = [dict(shape=shape,
                             w_slice_par=dict(mode='slice_random', normal_factor=w_normal_factor, normal_addend=w_normal_addend, 
                                              shift_factor=w_shift_factor, shift_addend=w_shift_addend),
                             y_grad_par=dict(mode='hamming_random', factor=y_factor, addend=y_addend)) for shape in unary_shapes]
        params_quadratic = [dict(shape=shape,
                                 w_slice_par=dict(mode='const_random', normal_factor=w_normal_factor,
                                                  normal_addend=w_normal_addend, shift_factor=w_shift_factor, 
                                                  shift_addend=w_shift_addend),
                                 y_grad_par=dict(mode='zero')) for shape in quadratic_shapes]
        params = params_unary + params_quadratic

        w_slice_l, y_grad_l = gen_w_and_y_grad(seed=seed, params=params)

        edges = [gen_edges(nn, ne, directed=directed_edges) for nn, ne in zip(num_nodes_l, num_edges_l)]
        solver_params = {
            "maxIter": 20,
            "innerIteration": 10,
            "presolveIterations": 30,
            "primalCheckingTriplets": 100,
            "multigraphMatchingRoundingMethod": "MCF_PS",
            "tighten": "",
            "tightenIteration": 50,
            "tightenInterval": 20,
            "tightenConstraintsPercentage": 0.1,
            "tightenReparametrization": "uniform:0.5"
        }
        solver_config = dict(edges=edges, solver_params=solver_params)
        return w_slice_l, y_grad_l, solver_config


class RankingBBSolver(BlackboxSolverAbstract):
    """
    Ranking solver.
    """

    @staticmethod
    def ranks_normal(sequence):
        return (np.argsort(np.argsort(sequence)[::-1]) + 1) / float(len(sequence))

    @staticmethod
    def solver(inputs):
        sequence = inputs[0]
        s = RankingBBSolver.ranks_normal(sequence)
        return [s]

    @staticmethod
    def gen_input(sequence_length, seed, w_normal_factor=1, w_normal_addend=0, w_shift_factor=1, w_shift_addend=0, y_factor=1, 
                  y_addend=0):
        w_slice_l, y_grad_l = gen_w_and_y_grad(
            seed=seed,
            params=[dict(shape=[sequence_length],
                         w_slice_par=dict(mode='slice_random', normal_factor=w_normal_factor, normal_addend=w_normal_addend, 
                                              shift_factor=w_shift_factor, shift_addend=w_shift_addend),
                         y_grad_par=dict(mode='random', factor=y_factor, addend=y_addend))])

        solver_config = dict()
        return w_slice_l, y_grad_l, solver_config


class TSPBBSolver(BlackboxSolverAbstract):
    """
    Tsp solver.
    """

    @staticmethod
    def solver(inputs):
        matrix = inputs[0]
        m = gurobi_tsp(matrix)
        return [m]

    @staticmethod
    def gen_input(num_nodes, seed, w_normal_factor=1, w_normal_addend=0, w_shift_factor=1, w_shift_addend=0, y_factor=1, 
                  y_addend=0):
        w_slice_l, y_grad_l = gen_w_and_y_grad(
            seed=seed,
            params=[dict(shape=(num_nodes, num_nodes),
                         w_slice_par=dict(mode='slice_random', normal_factor=w_normal_factor, normal_addend=w_normal_addend, 
                                              shift_factor=w_shift_factor, shift_addend=w_shift_addend, sym=True),
                         y_grad_par=dict(mode='hamming_random', factor=y_factor, addend=y_addend, sym=True))])

        solver_config = dict()
        return w_slice_l, y_grad_l, solver_config


class ShortestPathBBSolver(BlackboxSolverAbstract):
    """
    Shortest path solver.
    """

    @staticmethod
    def solver(inputs):
        matrix = inputs[0]
        m = dijkstra(matrix)
        return [m]

    @staticmethod
    def gen_input(num_nodes, seed, w_normal_factor=1, w_normal_addend=0, w_shift_factor=1, w_shift_addend=0, y_factor=1, 
                  y_addend=0):
        w_slice_l, y_grad_l = gen_w_and_y_grad(
            seed=seed,
            params=[dict(shape=(num_nodes, num_nodes),
                         w_slice_par=dict(mode='slice_random', normal_factor=w_normal_factor, normal_addend=w_normal_addend, 
                                          shift_factor=w_shift_factor, shift_addend=w_shift_addend, pos=True),
                         y_grad_par=dict(mode='hamming_random', factor=y_factor, addend=y_addend))])

        solver_config = {}
        return w_slice_l, y_grad_l, solver_config




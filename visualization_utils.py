import datetime
import itertools as it
from abc import ABC, abstractmethod
from contextlib import suppress

import ipyvolume as ipv
import numpy as np

colors = [(0.923, 0.386, 0.209),
          (0.368, 0.507, 0.71)]

colors_alpha = [tuple(list(c) + [1.0]) for c in colors]


def darker(color, factor=1.0):
    r, g, b, a = color
    return (r * factor, g * factor, b * factor, a)


def interpolate(color1, color2, factor):
    exp = 4
    n = factor ** exp + (1 - factor) ** exp
    return tuple([(factor ** exp * c1 + (1 - factor) ** exp * c2) / n for c1, c2 in zip(color1, color2)])


def plot_mesh(value_mesh_list, w1_mesh, w2_mesh, save, show_box, show_axes):
    ipv.clear()
    figs = []
    for mesh in value_mesh_list:
        fig = ipv.pylab.figure()
        fig.camera.up = (0, 0, 1)
        fig.camera.position = (-2, 1, -0.5)
        fig.camera_control = "trackball"
        if not show_axes:
            ipv.style.box_off()
        else:
            ipv.xlabel("w1")
            ipv.ylabel("w2")
            ipv.zlabel("f_lambda")
        if not show_box:
            ipv.pylab.style.axes_off()

        ipv.pylab.zlim(mesh.min(), mesh.max())
        ptp = (mesh - mesh.min()).ptp()

        col = []
        for m in mesh:
            znorm = (m - m.min()) / (m.max() - m.min() + 1e-8)
            color = np.asarray([[interpolate(darker(colors_alpha[0], 1.5 * x + 0.75),
                                             darker(colors_alpha[1], 1.5 * (1 - x) + 0.75), x) for x in y] for y in
                                znorm])
            col.append(color)
        color = np.array(col)
        surf = ipv.plot_surface(w1_mesh, w2_mesh, mesh, color=color[..., :3])
        ipv.animation_control(surf, interval=400)
        figs.append(fig)
        ipv.show()
    if save:
        ipv.save(f'renders/{datetime.datetime.now().strftime("%m%d-%H%M")}.html', offline=True)
    return figs


def apply_shift_and_factor(val, sym=False, pos=False, addend=0.0, factor=1.0):
    val = val * factor + addend
    if sym:
        val = (val + val.T) / 2
    if pos:
        val = np.abs(val)
    return val


def hamming_grad_random(shape, **kwargs):
    y_true = np.random.randint(2, size=shape)
    return hamming_grad(y_true, **kwargs)


def hamming_grad(y_true, **kwargs):
    y_grad = -0.25 * (y_true - 0.5)
    return apply_shift_and_factor(y_grad, **kwargs)


def random_grad(shape, **kwargs):
    y_grad = np.random.rand(*shape)
    return apply_shift_and_factor(y_grad, **kwargs)


def zero_grad(shape, **kwargs):
    y_grad = np.zeros(shape=shape)
    return apply_shift_and_factor(y_grad, **kwargs)


grad_f_dict = dict(hamming_random=hamming_grad_random, hamming=hamming_grad, random=random_grad, zero=zero_grad)


def get_grad(mode, **kwargs):
    grad = grad_f_dict[mode](**kwargs)
    return grad


def w_slice(shift, normal, shift_addend, shift_factor, normal_addend, normal_factor, sym=False, pos=False):
    """
    n_shift and n_factor denote the shift and multiplication constant of the normal
    equivalent for the shift
    sym makes shift and normal symmetric
    """
    shift = shift * shift_factor + shift_addend
    normal = normal * normal_factor + normal_addend
    if sym:
        shift = (shift + shift.T) / 2
        normal = (normal + normal.transpose(0, 2, 1)) / 2
    if pos:
        shift = np.abs(shift)
        normal = np.abs(normal)
    return dict(shift=shift, normal=normal)


def w_slice_random(shape, **kwargs):
    shift = np.random.rand(*shape)
    normal = np.random.rand(*(2, *shape))
    return w_slice(shift, normal, **kwargs)


def w_const(w, **kwargs):
    normal = np.zeros(shape=(2, *w.shape))
    return w_slice(shift=w, normal=normal, **kwargs)


def w_const_random(shape, **kwargs):
    w = np.random.rand(*shape)
    return w_const(w, **kwargs)


w_f_dict = dict(slice_random=w_slice_random, slice=w_slice, const=w_const, const_random=w_const_random)


def get_w_slice(mode, **kwargs):
    w = w_f_dict[mode](**kwargs)
    return w


def gen_w_and_y_grad(seed, params):
    np.random.seed(seed)
    w_slice_l = []
    y_grad_l = []
    for p in params:
        w_slice_l.append(get_w_slice(shape=p['shape'], **p['w_slice_par']))
        y_grad_l.append(get_grad(shape=p['shape'], **p['y_grad_par']))
    return w_slice_l, y_grad_l


def gen_edges(num_nodes, num_edges, directed):
    all_e = [i for i in it.combinations(range(num_nodes), 2)]
    all_e_inv = [e[::-1] for e in all_e]

    if directed:
        all_e = all_e + all_e_inv
        ind = np.random.choice(len(all_e), num_edges, replace=False)
        return np.array([all_e[i] for i in ind])
    else:
        if num_edges % 2 == 1:
            raise ValueError(f'(Number of edges has to be divisible by 2 for an undirected graph, as each edge counts as two (one back and one forth)')
        ind = np.random.choice(len(all_e), num_edges // 2, replace=False)
        return np.array([all_e[i] for i in ind] + [all_e_inv[i] for i in ind])


class BlackboxSolverAbstract(ABC):
    """
    Abstract solver used for generating meshes and plotting.
    """

    def __init__(self, **kwargs):
        self.w_slice_l, self.y_grad_l, self.solver_config = self.gen_input(**kwargs)

    @staticmethod
    @abstractmethod
    def gen_input(**kwargs):
        pass

    @staticmethod
    @abstractmethod
    def solver(**solver_input):
        pass

    @staticmethod
    def cost(w, y):
        return np.sum(w * y)

    @staticmethod
    def f(y, y_grad):
        return np.sum(y * y_grad)

    def f_lambda(self, w_l, lambda_val):
        y_l = self.solver(w_l, **self.solver_config)
        w_prime_l = [w + lambda_val * y_grad for w, y_grad in zip(w_l, self.y_grad_l)]
        y_prime_l = self.solver(w_prime_l, **self.solver_config)
        c_val = sum(self.cost(w, y) for w, y in zip(w_l, y_l))
        c_val_prime = sum(self.cost(w, y_prime) for w, y_prime in zip(w_l, y_prime_l))
        f_val = sum(self.f(y_prime, y_grad) for y_prime, y_grad in zip(y_prime_l, self.y_grad_l))
        # Here we make use of the fact, that (c - c') = w * (y - y') ~ w * df/dw which is the gradient evaluated at w
        return f_val - (c_val - c_val_prime) / lambda_val, c_val_prime

    @staticmethod
    def d2slice_single(w1, w2, normal, shift):
        return np.sum(np.array([w * slice_n for w, slice_n in zip([w1, w2], normal)]), axis=0) + shift

    def d2slice(self, w1, w2):
        return [self.d2slice_single(w1, w2, **w_slice) for w_slice in self.w_slice_l]

    def gen_meshes(self, lambdas, bounds1, bounds2, partitions):
        w1_vals = np.linspace(*bounds1, partitions)
        w2_vals = np.linspace(*bounds2, partitions)
        w1_mesh, w2_mesh = np.meshgrid(w1_vals, w2_vals)

        def mesh_for_single(**kwargs):
            vals = zip(*[self.f_lambda(self.d2slice(w1, w2), **kwargs) for w1, w2 in it.product(w1_vals, w2_vals)])
            vals = [np.array(v).reshape(w1_mesh.shape) for v in vals]
            vals = [v - v.min() for v in vals]
            return vals

        meshes = [mesh_for_single(lambda_val=l) for l in lambdas]
        meshes = [np.array(vals) for vals in zip(*meshes)]
        return meshes, w1_mesh, w2_mesh

    def plot_flambda(self, lambdas, partitions, bounds1, bounds2, save=False, plot_cost=False, show_box=True, show_axes=True):
        meshes, w1_mesh, w2_mesh = self.gen_meshes(lambdas, bounds1, bounds2, partitions)
        if not plot_cost:
            meshes = [meshes[0]]
        self.figs = plot_mesh(meshes, w1_mesh, w2_mesh, save=save, show_box=show_box, show_axes=show_axes)

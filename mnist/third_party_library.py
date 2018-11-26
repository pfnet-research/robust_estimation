# Author: Mathieu Blondel
# License: BSD 3 clause
# https://gist.github.com/mblondel/6f3b7aaad90606b98f71
import chainer


def projection_simplex_sort(v, z=1):
    xp = chainer.backends.cuda.get_array_module(v)
    n_features = v.shape[0]
    u = xp.sort(v)[::-1]
    cssv = xp.cumsum(u) - z
    ind = xp.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = xp.maximum(v - theta, 0)
    return w

import inspect
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F

from chainer.dataset import convert
from third_party_library import projection_simplex_sort


class LRE(chainer.training.StandardUpdater):
    """
    """

    def __init__(self, iterator, optimizer, converter=convert.concat_examples,
                 device=None, loss_func=None, loss_scale=None, alpha=0.001):
        super(LRE, self).__init__(
            iterator, optimizer, converter, device, loss_func, loss_scale)
        self.alpha = alpha

    def update_core(self):
        it = self._iterators['main']
        batch = it.next()
        batchsize = len(batch)
        *x, t = self.converter(batch, self.device)

        batch_val = self._iterators['val'].next()
        *x_val, t_val = self.converter(batch_val, self.device)

        optimizer = self._optimizers['main']
        loss_func = self.loss_func or optimizer.target.lossfun
        model = optimizer.target
        model_tmp = model.copy()

        # Line 4 (Algorithm 1 in https://arxiv.org/abs/1803.09050)
        ys = model_tmp.predictor(*x)

        # Line 5
        weight = L.Linear(batchsize, 1, nobias=True, initialW=0)
        weight.to_gpu()

        # F.sigmoid_cross_entropyにはenable_double_backpropがないので
        if 'enable_double_backprop' in inspect.getargspec(loss_func).args:
            loss_f = weight(loss_func(ys, t, reduce='no',
                                      enable_double_backprop=True)[None])
        else:
            loss_f = weight(loss_func(ys, t, reduce='no')[None])

        # Line 6
        model_tmp.cleargrads()
        weight.cleargrads()
        loss_f.backward(retain_grad=True, enable_double_backprop=True)

        # Line 7
        for link in model_tmp.predictor.links(skipself=True):
            for name, param in link.namedparams():
                if name.count('/') == 1:
                    super(chainer.Link, link).__setattr__(
                        name[1:], param - self.alpha * param.grad_var)

        # Line 8
        ys_val = model_tmp.predictor(*x_val)

        # Line 9
        loss_g = loss_func(ys_val, t_val)

        # Line 10
        model_tmp.cleargrads()
        weight.cleargrads()

        # Line 11
        w = - chainer.grad([loss_g], [weight.W])[0].data
        w[w < 0] = 0
        if F.sum(w).data != 0:
            w /= F.sum(w).data
        weight.W.data[:] = w

        y = model.predictor(*x)
        loss_f2 = F.sum(weight(loss_func(y, t, reduce='no')[None]))
        model.cleargrads()
        loss_f2.backward()
        optimizer.update()

        # compatibility with chainer_chemistry.models.Classifier
        if isinstance(model.accfun, dict):
            metrics = {key: value(y, t) for key, value in model.accfun.items()}
            chainer.reporter.report(metrics, model)
            chainer.reporter.report({'loss': loss_f2}, model)
        else:
            chainer.reporter.report(
                {'loss': loss_f2, 'accuracy': model.accfun(y, t)}, model)


class Proposed(chainer.training.StandardUpdater):
    """
    """

    def __init__(self, iterator, optimizer, converter=convert.concat_examples,
                 device=None, loss_func=None, loss_scale=None, alpha=0.001, sampling_size=32):
        super(Proposed, self).__init__(iterator, optimizer,
                                       converter, device, loss_func, loss_scale)
        self.alpha = alpha
        self.sampling_size = sampling_size

    def update_core(self):
        it = self._iterators['main']
        batch = it.next()
        *x, t = self.converter(batch, self.device)

        batchsize = len(batch)
        optimizer = self._optimizers['main']
        loss_func = self.loss_func or optimizer.target.lossfun
        model = optimizer.target
        model_tmp = model.copy()

        # Line 4 (Algorithm 1 in https://arxiv.org/abs/1803.09050)
        ys = model_tmp.predictor(*x)

        # Line 5
        weight = L.Linear(batchsize, 1, nobias=True, initialW=0)
        weight.to_gpu()

        # F.sigmoid_cross_entropyにはenable_double_backpropがないので
        if 'enable_double_backprop' in inspect.getargspec(loss_func).args:
            loss_f = weight(loss_func(ys, t, reduce='no',
                                      enable_double_backprop=True)[None])
        else:
            loss_f = weight(loss_func(ys, t, reduce='no')[None])

        # Line 6
        model_tmp.cleargrads()
        weight.cleargrads()
        loss_f.backward(retain_grad=True, enable_double_backprop=True)

        # Line 7
        for link in model_tmp.predictor.links(skipself=True):
            for name, param in link.namedparams():
                if name.count('/') == 1:
                    super(chainer.Link, link).__setattr__(
                        name[1:], param - self.alpha * param.grad_var)

        xp = chainer.backends.cuda.get_array_module(t)
        val_ind = xp.where(t == 1)[0]

        # Line 8
        ys_val = model_tmp.predictor(*x)[val_ind]
        t_val = t[val_ind]
        # Line 9
        loss_g = loss_func(ys_val, t_val)
        # Line 10
        model_tmp.cleargrads()
        weight.cleargrads()
        # Line 11
        w_tmp = chainer.grad([loss_g], [weight.W])[0].data
        w_tmp = projection_simplex_sort(-w_tmp[0])
        val_ind = xp.random.choice(batchsize, size=self.sampling_size, p=w_tmp)
        y = model.predictor(*x)
        loss_f2 = loss_func(y[val_ind], t[val_ind])
        model.cleargrads()
        loss_f2.backward()
        optimizer.update()

        # compatibility with chainer_chemistry.models.Classifier
        if isinstance(model.accfun, dict):
            metrics = {key: value(y, t) for key, value in model.accfun.items()}
            chainer.reporter.report(metrics, model)
            chainer.reporter.report({'loss': loss_f2}, model)
        else:
            chainer.reporter.report(
                {'loss': loss_f2, 'accuracy': model.accfun(y, t)}, model)


class LossImportanceSampling(chainer.training.StandardUpdater):
    """

    """

    def __init__(self, iterator, optimizer, converter=convert.concat_examples,
                 device=None, loss_func=None, loss_scale=None, sampling_size=32):
        super(LossImportanceSampling, self).__init__(
            iterator, optimizer, converter, device, loss_func, loss_scale)
        self.sampling_size = sampling_size

    def update_core(self):
        optimizer = self._optimizers['main']
        loss_func = self.loss_func or optimizer.target.lossfun
        model = optimizer.target

        it = self._iterators['main']
        batch = it.next()
        batchsize = len(batch)

        *x, t = self.converter(batch, self.device)

        y = model.predictor(*x)
        loss = loss_func(y, t, reduce='no')
        loss_all = F.mean(loss)
        prob = loss.data[:, 0] + 1e-10
        prob /= prob.sum()

        index = np.random.choice(
            batchsize, size=self.sampling_size, replace=True, p=prob)
        weight = L.Linear(batchsize, 1, nobias=True, initialW=0)
        weight.to_gpu()
        weight.W.data[:, index] = 1 / (batchsize * prob[index])

        loss_im = F.sum(weight(loss[None]))
        model.cleargrads()
        weight.cleargrads()
        loss_im.backward()
        optimizer.update()

        # compatibility with chainer_chemistry.models.Classifier
        if isinstance(model.accfun, dict):
            metrics = {key: value(y, t) for key, value in model.accfun.items()}
            chainer.reporter.report(metrics, model)
            chainer.reporter.report({'loss': F.sum(loss_all)}, model)
        else:
            chainer.report(
                {'loss': F.sum(loss), 'accuracy': model.accfun(y, t)}, model)

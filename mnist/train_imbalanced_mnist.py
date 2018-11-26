import argparse
import os

import numpy as np
import chainer
from chainer import functions as F
from chainer import iterators
from chainer import optimizers
from chainer import training
from chainer.training import extensions as E
from chainer_chemistry.models.prediction import Classifier
from chainer_chemistry.training.extensions import ROCAUCEvaluator  # NOQA

from net import LeNet
from updaters import Proposed, LRE


def get_imbalanced_data(n_train=1000, seed=111, dataset='mnist'):
    np.random.seed(seed)
    if dataset == 'mnist':
        train, test = chainer.datasets.get_mnist(ndim=3)
    else:
        train, test = chainer.datasets.cifar.get_cifar10()

    x, t = chainer.dataset.concat_examples(train)

    if not isinstance(n_train, dict):
        d = {}
        for i in np.unique(t):
            d[i] = n_train
        n_train = d

    train_images, train_labels = None, None

    for cls in n_train.keys():
        n = n_train[cls]
        indices = np.where(t == cls)[0]
        train_ind = np.random.permutation(indices)[:n]

        train_images = x[train_ind] if train_images is None else \
            np.concatenate((train_images, x[train_ind]), axis=0)
        train_labels = t[train_ind] if train_labels is None else \
            np.concatenate((train_labels, t[train_ind]), axis=0)

    x, t = chainer.dataset.concat_examples(test)
    test_images = x[np.isin(t, list(n_train.keys()))]
    test_labels = t[np.isin(t, list(n_train.keys()))]

    train = chainer.datasets.tuple_dataset.TupleDataset(
        train_images, train_labels)
    test = chainer.datasets.tuple_dataset.TupleDataset(
        test_images, test_labels)

    return train, test


def get_binary_imbalanced_data(n_train={4: 5 * 5 - 5, 9: 995 * 5 - 5},
                               n_train_val={4: 5, 9: 5}, dataset='mnist'):
    if dataset == 'mnist':
        train, test = chainer.datasets.get_mnist(ndim=3)
    else:
        train, test = chainer.datasets.cifar.get_cifar10()

    x, t = chainer.dataset.concat_examples(train)
    x_test, t_test = chainer.dataset.concat_examples(test)

    if not isinstance(n_train, dict) or not isinstance(n_train_val, dict):
        raise TypeError

    if len(np.unique(n_train.keys())) >= 2:
        raise NotImplementedError

    train_images, train_labels = None, None
    train_val_images, train_val_labels = None, None
    test_images, test_labels = None, None

    for i, cls in enumerate(n_train.keys()):
        n1 = n_train[cls]
        n2 = n_train_val[cls]
        indices = np.where(t == cls)[0]
        train_ind, train_val_ind, _ = np.split(
            np.random.permutation(indices), np.cumsum([n1, n2]))

        train_images = x[train_ind] if train_images is None else \
            np.concatenate((train_images, x[train_ind]), axis=0)
        train_label = np.full(len(train_ind), i)
        train_labels = train_label if train_labels is None else \
            np.concatenate((train_labels, train_label), axis=0)

        train_val_images = x[train_val_ind] if train_val_images is None \
            else np.concatenate((train_val_images, x[train_val_ind]), axis=0)
        train_val_label = np.full(len(train_val_ind), i)
        train_val_labels = train_val_label if train_val_labels is None \
            else np.concatenate((train_val_labels, train_val_label), axis=0)

        test_ind = np.where(t_test == cls)[0]
        test_images = x_test[test_ind] if test_images is None else \
            np.concatenate((test_images, x_test[test_ind]), axis=0)
        test_label = np.full(len(test_ind), i)
        test_labels = test_label if test_labels is None else np.concatenate(
            (test_labels, test_label), axis=0)

    train = chainer.datasets.tuple_dataset.TupleDataset(
        train_images, train_labels)
    train_val = chainer.datasets.tuple_dataset.TupleDataset(
        train_val_images, train_val_labels)
    test = chainer.datasets.tuple_dataset.TupleDataset(
        test_images, test_labels)

    return train, train_val, test


def main():
    parser = argparse.ArgumentParser(
        description='Imbalanced MNIST classification')
    parser.add_argument('--eval-mode', type=int, default=1,
                        help='Evaluation mode.'
                        '0: only binary_accuracy is calculated.'
                        '1: binary_accuracy and ROC-AUC score is calculated')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='batch size')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID to use. Negative value indicates '
                        'not to use GPU and to run the code in CPU.')
    parser.add_argument('--out', '-o', type=str, default='result',
                        help='path to output directory')
    parser.add_argument('--epoch', '-e', type=int, default=10,
                        help='number of epochs')
    parser.add_argument('--resume', '-r', type=str, default='',
                        help='path to a trainer snapshot')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--protocol', type=int, default=2,
                        help='protocol version for pickle')
    parser.add_argument('--model-filename', type=str, default='classifier.pkl',
                        help='file name for pickled model')
    parser.add_argument('--updater-type', type=str, default='standard')
    parser.add_argument('--sampling-size', type=int, default=32)
    parser.add_argument('--optimizer-type', type=str, default='Adam')
    parser.add_argument('--alpha', type=str, default='0.001')

    args = parser.parse_args()
    # Dataset preparation
    train, train_val, val = get_binary_imbalanced_data()

    train_iter = iterators.SerialIterator(train, args.batchsize)
    val_iter = iterators.SerialIterator(val, args.batchsize,
                                        repeat=False, shuffle=False)

    model = LeNet(n_class=1, binary=True)
    classifier = Classifier(model,
                            lossfun=F.sigmoid_cross_entropy,
                            metrics_fun=F.binary_accuracy,
                            device=args.gpu)

    if args.optimizer_type == 'Adam':
        optimizer = optimizers.Adam()
    else:
        optimizer = optimizers.SGD(lr=1e-3)
    optimizer.setup(classifier)

    updater_type = args.updater_type
    if updater_type == 'standard':
        updater = training.StandardUpdater(
            train_iter, optimizer, device=args.gpu)
    elif updater_type == 'proposed':
        updater = Proposed(
            train_iter, optimizer, device=args.gpu,
            sampling_size=args.sampling_size)
    elif updater_type == 'LRE':
        x, t = chainer.dataset.concat_examples(train)

        train_val_iter = iterators.SerialIterator(train_val, len(train_val))
        updater = LRE(
            {'main': train_iter, 'val': train_val_iter}, optimizer,
            device=args.gpu, alpha=args.alpha)

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(E.Evaluator(val_iter, classifier,
                               device=args.gpu))
    trainer.extend(E.LogReport())

    eval_mode = args.eval_mode
    if eval_mode == 0:
        trainer.extend(E.PrintReport([
            'epoch', 'main/loss', 'main/accuracy', 'validation/main/loss',
            'validation/main/accuracy', 'elapsed_time']))
    elif eval_mode == 1:
        train_eval_iter = iterators.SerialIterator(train, args.batchsize,
                                                   repeat=False, shuffle=False)
        trainer.extend(ROCAUCEvaluator(
            train_eval_iter, classifier, eval_func=model,
            device=args.gpu, name='train',
            pos_labels=1, ignore_labels=-1, raise_value_error=False))
        # extension name='validation' is already used by `Evaluator`,
        # instead extension name `val` is used.
        trainer.extend(ROCAUCEvaluator(
            val_iter, classifier, eval_func=model,
            device=args.gpu, name='val',
            pos_labels=1, ignore_labels=-1))
        trainer.extend(E.PrintReport([
            'epoch', 'main/loss', 'main/accuracy', 'train/main/roc_auc',
            'validation/main/loss', 'validation/main/accuracy',
            'val/main/roc_auc', 'elapsed_time']))
    else:
        raise ValueError('Invalid accfun_mode {}'.format(eval_mode))
    trainer.extend(E.ProgressBar(update_interval=10))
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(E.snapshot(), trigger=(frequency, 'epoch'))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()
    classifier.save_pickle(os.path.join(args.out, args.model_filename),
                           protocol=args.protocol)


if __name__ == '__main__':
    main()

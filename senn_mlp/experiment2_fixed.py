from functools import partial
from itertools import islice
from typing import Any, Callable, Sequence, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math

import jax

jax.config.update("jax_enable_x64", False)
from jax import lax, random, numpy as jnp
from jax.numpy.linalg import norm as jnorm
from jax.tree_util import tree_map as jtm, tree_reduce as jtr, Partial
import flax
from flax import linen as nn
from flax.training import checkpoints
from tensorflow import summary
from tensorflow import convert_to_tensor

import nets
from nets import DDense, Rational1D, DConv, Layer, Layers
from nets import tree_length, pad_target
from optim import SimpleGradient, KrylovNG, CGNG

from data import smallnist, smallfnist, cfg_tranches, tranch_cat
from jaxutils import key_iter
import experiment_utils

from langevin import mala_step, mala_steps

import os
import confuse
import argparse
from datetime import datetime

from rtpt.rtpt import RTPT


from tqdm import tqdm


def tree_dot(a, b):
    return jtr(lambda c, d: c + d, jtm(lambda c, d: jnp.sum(c * d), a, b))


def insert_params(state, params):
    return state.copy({"params": params})


def validation_loss(model, loss_function, dataset, labels, params):
    Y = jax.vmap(lambda x: model.apply(params, x))(dataset)

    Ym = Y
    L = jax.vmap(loss_function)(labels, Y)
    correct = Ym.argmax(axis=-1) == labels
    pseudo_loss = jnp.log10(jnp.sum(correct) + 1) - jnp.log10(jnp.sum(~correct) + 1)
    return L.mean(), correct.mean(), pseudo_loss


def print_tree_type(tree):
    print(jtm(lambda x: x.shape, tree))


def print_tree_mags(tree):
    print(jtm(lambda arr: (arr**2).mean(), tree))


@dataclass
class ModelTemplate:
    capacities: Sequence[int]
    contents: Sequence[Optional[int]]
    rational: bool = False

    def layer(self, features, final=False):
        if self.rational:
            nonlin = partial(Rational1D, residual=True, init_identity=False)
        else:
            nonlin = lambda: jax.nn.silu
        return Layer(features, [], DDense, [] if final else [nonlin])

    def build(self):
        hidden_layers = [self.layer(f) for f in self.capacities[:-1]]
        final_layer = self.layer(self.capacities[-1], final=True)
        return Layers(hidden_layers + [final_layer])

    def enabled_layers(self):
        return list([i for i, d in enumerate(self.contents) if d is not None])

    def disabled_layers(self):
        return list([i for i, d in enumerate(self.contents) if d is None])

    def in_out_indices(self, layer_index):
        conarr = np.array(self.contents)
        split_at = layer_index + 1
        (preceding_enabled,) = np.nonzero(conarr[:split_at][:-1] != None)
        in_index = 0 if len(preceding_enabled) == 0 else preceding_enabled[-1] + 1

        (subsequent_enabled,) = np.nonzero(conarr[split_at:] != None)
        out_index = split_at + subsequent_enabled[0]
        return in_index, out_index


class Task(ABC):

    @abstractmethod
    def get_data(cfg, test=False):
        pass
        return dataset, labels

    @abstractmethod
    def loss_function(label, output):
        pass


class Regression(Task):

    @staticmethod
    def loss_function(label, output):
        return ((output - label) ** 2).mean()


class Classification(Task):

    @staticmethod
    def loss_function(label, output):
        return jax.nn.logsumexp(output) - output[label]


class SameFamilyRegression(Regression):

    def __init__(self, cfg, key):
        self.out_size = cfg["task"]["out_size"].get()
        self.in_size = cfg["task"]["in_size"].get(self.out_size)
        capacities = cfg["task"]["hidden"].get() + [self.out_size]
        self.template = ModelTemplate(capacities, capacities, rational=True)
        self.model = self.template.build()
        self.state = self.model.init(key, jnp.zeros((self.in_size,)))
        self.target_func = jax.jit(Partial(self.model.apply, self.state))

        self.N = cfg["data"]["N"].get()
        self.TN = cfg["data"]["TN"].get(self.N)

    def get_data(self, cfg, test=False, key=None):
        assert key is not None
        N = self.TN if test else self.N
        xs = jax.random.normal(key, (N, self.in_size))
        ys = jax.vmap(self.target_func)(xs)
        return xs, ys


class ImgVecClass(Classification):

    def __init__(self, cfg):
        self.cfg = cfg
        self.tranches = cfg_tranches(cfg["data"], cfg["task"]["resolution"].get())

    def get_data(self, _, test=False, index=0):
        dataset, labels = tranch_cat(self.tranches, index, train=not test)
        return jax.vmap(jnp.ravel)(dataset), labels


class SklGenClass(Classification):

    def __init__(self, cfg):
        self.cfg = cfg
        self.train_seed = cfg["data"]["train_seed"].get()
        self.test_seed = cfg["data"]["test_seed"].get()

    def get_data(self, _, test=False, index=0):
        data_cfg = self.cfg["data"]
        assert (
            data_cfg["name"].get(None) == "make_moons"
        ), "only make_moons is supported"
        from sklearn.datasets import make_moons

        data, labels = make_moons(
            n_samples=data_cfg["TN"].get() if test else data_cfg["N"].get(),
            noise=data_cfg["noise"].get(),
            random_state=(
                data_cfg["test_seed"].get() if test else data_cfg["train_seed"].get()
            ),
        )
        return jnp.array(data), jnp.array(labels)


class Solver:
    """probably should be further factored"""

    def __init__(self, cfg, template, task, key, example):
        self.cfg = cfg["opt"]
        self.template = template
        self.task = task
        self.model = self.template.build()
        self.state = self.model.init(key, example)
        print(f"contents: {self.template.contents}")

        def nonincreasing(seq):
            return all(x >= y for x, y in zip(seq, seq[1:]))

        assert nonincreasing(
            [c for c in self.template.contents if c is not None]
        ), "model.restrict_params is broken for increasing feature sizes"
        self.state = self.model.apply(
            self.state, self.template.contents, method=self.model.restrict_params
        )
        self.optimizer, self.opt_state = self.make_opt(self.state)
        self.lr = cfg["opt"]["lr"].get()
        self.recompile()
        self.last_natlen = None
        self.weight_decay = cfg["opt"]["weight_decay"].get(0.0)

    def item_loss(self, state, datum, label):
        y = self.model.apply(state, datum)
        loss = self.task.loss_function(label, y)
        return loss

    def item_correct(self, state, datum, label):
        y = self.model.apply(state, datum)
        return jnp.argmax(y) == label

    def _batch_loss(self, state, data, labels):
        """NOTE: This will not account for a changing model -
        if model changes this must be re-jitted"""
        return jax.vmap(partial(self.item_loss, state))(data, labels).mean()

    def _batch_acc(self, state, data, labels):
        """NOTE: This will not account for a changing model -
        if model changes this must be re-jitted"""
        return jax.vmap(partial(self.item_correct, state))(data, labels).mean()

    def make_opt(self, state):
        opt = CGNG(self.cfg, state["params"])
        opt_state = opt.init(flax.core.frozen_dict.freeze({}))
        return opt, opt_state

    def restrict_grad(self, grad):
        variables = {"params": grad}
        return self.model.apply(
            variables, self.template.contents, method=self.model.restrict_grad
        )["params"]

    def recompile(self):

        func = lambda params, x: self.model.apply(
            self.state.copy({"params": params}), x
        )
        self.observe = jax.jit(
            partial(
                self.optimizer.observe,
                func,
                self.task.loss_function,
                self.restrict_grad,
            )
        )
        self.apply_model = jax.jit(self.model.apply)
        self.batch_loss = jax.jit(self._batch_loss)
        self.batch_acc = jax.jit(self._batch_acc)

    def update_params(self, mul, tan, weight_decay=0.0):

        tan = jtm(
            lambda t, s: 1 / (1 + weight_decay) * t
            + weight_decay / (1 + weight_decay) * s,
            tan,
            self.state["params"],
        )
        tan = self.restrict_grad(tan)

        new = jtm(lambda t, s: s + -mul * self.lr * t, tan, self.state["params"])
        self.state = self.state.copy({"params": new})

    def train_batch(self, batch, observe_only=False, loud=False):
        data, labels = batch
        loss = self.batch_loss(self.state, data, labels)
        if loud:
            print(f"loss: {loss:.3E}")
        summary.scalar("loss", loss)
        summary.scalar(
            "features", sum([c for c in self.template.contents if c is not None])
        )
        for i, f in enumerate(self.template.contents):
            summary.scalar(f"features_{i}", f if f is not None else 0)
        self.opt_state = self.observe(
            data, labels, self.state["params"], self.opt_state
        )

        grad = self.optimizer.SG.read(self.opt_state)
        ngrad = self.optimizer.read(self.opt_state)
        nat_len = tree_dot(ngrad, grad)
        self.last_natlen = nat_len
        summary.scalar("baseline", nat_len)
        summary.scalar("normed_baseline", nat_len / loss)

        summary.scalar("param_Fnorm", self.optimizer.param_Fnorm.read(self.opt_state))
        summary.scalar(
            "param_l2norm", tree_dot(self.state["params"], self.state["params"])
        )

        if not observe_only:

            self.update_params(1.0, ngrad, weight_decay=self.weight_decay)
        if loud:
            return loss

    def eval_feature_proposal(self, batch, fstate):
        assert self.cfg["tau"].get() is None
        assert (
            False
        ), "nat_len calculation is inconsistent with that of the training update"

    def test_batch(self, batch):

        data, labels = batch
        loss = self.batch_loss(self.state, data, labels)

        summary.scalar("validation loss", loss)

    def test_acc(self, batch):
        data, labels = batch
        acc = self.batch_acc(self.state, data, labels)
        summary.scalar("validation accuracy", acc)

    def train_acc(self, batch):
        data, labels = batch
        acc = self.batch_acc(self.state, data, labels)
        summary.scalar("training accuracy", acc)


def kfac_observe(model, loss_function, state, tangent, pair):

    datum, label = pair
    y, Jt, aux = jax.jvp(
        fun=lambda state: model.apply(state, datum, mutable="intermediates"),
        primals=(state,),
        tangents=(tangent,),
        has_aux=True,
    )
    dloss = jax.grad(partial(loss_function, label))(y)
    loss_sqnorm = lax.pmean(jnp.sum(dloss**2), "batch")

    _, backward = jax.vjp(
        lambda state: model.apply(state, datum),
        state,
    )

    (grad,) = backward(dloss / jnp.sqrt(jnp.sum(dloss**2)))

    Jt_rescale = lax.psum(jnp.sum(Jt * dloss), "batch") / (
        lax.psum(jnp.sum(Jt * Jt), "batch") + 1e-10
    )

    lres = dloss - Jt * Jt_rescale

    (state_residual,) = backward(lres)

    acts = model.apply(aux, method=model.extract_activations)
    out_grads = model.apply(grad, method=model.extract_out_grads)
    out_ress = model.apply(state_residual, method=model.extract_out_grads)

    As = [lax.pmean(a[:, None] * a[None, :], "batch") for (a,) in acts]
    Gs = [lax.pmean(g[:, None] * g[None, :], "batch") for g in out_grads]

    return As, Gs, acts, out_ress, out_grads, loss_sqnorm


def kfac_single_eval(fmodel, act, Ginv, res, feature):
    fval = fmodel.apply(feature, act)
    A = lax.pmean(fval[..., :, None] * fval[..., None, :], "batch")
    Ainv = jnp.linalg.pinv(A)
    corr = lax.pmean(fval[:, None] * res[None, :], "batch")

    normed_corr = Ainv @ corr @ Ginv
    rscore = jnp.sum(corr * normed_corr)
    return rscore


def kfac_single_sgd(fmodel, act, Ginv, res, feature, eps, atikh=0.0):
    fval, backward = jax.vjp(lambda theta: fmodel.apply(theta, act), feature)
    A = lax.pmean(fval[..., :, None] * fval[..., None, :], "batch")
    Ainv = jnp.linalg.pinv(tikhonov(A, atikh))
    corr = lax.pmean(fval[:, None] * res[None, :], "batch")

    normed_corr = Ainv @ corr @ Ginv
    rscore = jnp.sum(corr * normed_corr)

    resres = res - corr.T @ Ainv @ fval
    fgrad = normed_corr @ resres
    (theta_grad,) = jtm(lambda arr: lax.pmean(arr, "batch"), backward(fgrad))

    def ldet(feat):
        L = feat["params"]["linear"]["kernel"]
        sign, logabs = jnp.linalg.slogdet(L.T @ L)
        return logabs

    REGULARISE = 1e-2
    lnmag, lnmag_grad = jax.value_and_grad(ldet)(feature)

    new_feature = jtm(
        lambda theta, g, lgrad: theta + eps * g - REGULARISE * lnmag * lgrad,
        feature,
        theta_grad,
        lnmag_grad,
    )
    return rscore, new_feature


def get_prior_var(feature):

    return jtm(lambda f: jnp.ones_like(f), feature)


def kfac_mala_burst(
    fmodel,
    act,
    Ginv,
    res,
    feature,
    key,
    lr,
    temp=1e0,
    atikh=0.0,
    steps=10,
    score_norm=1e0,
):
    priorvar = get_prior_var(feature)

    def score_cotan(fval):
        A = lax.pmean(fval[..., :, None] * fval[..., None, :], "batch")
        Ainv = jnp.linalg.pinv(tikhonov(A, atikh))
        corr = lax.pmean(fval[:, None] * res[None, :], "batch") / jnp.sqrt(score_norm)

        normed_corr = Ainv @ corr @ Ginv
        rscore = jnp.sum(corr * normed_corr)

        resres = res - corr.T @ Ainv @ fval
        fgrad = normed_corr @ resres
        return rscore, fgrad

    def loss_grad(feature):
        fval, backward = jax.vjp(lambda theta: fmodel.apply(theta, act), feature)
        rscore, cotangent = score_cotan(fval)
        (theta_grad,) = jtm(lambda arr: lax.pmean(arr, "batch"), backward(cotangent))
        return -rscore, theta_grad

    feature, accept_rate = mala_steps(
        loss_grad, priorvar, feature, key, lr, steps, temp=temp
    )
    rscore, _ = score_cotan(fmodel.apply(feature, act))
    return rscore * score_norm, feature, accept_rate


def kfac_direct_mala(
    model,
    loss_function,
    fmodel,
    state,
    tangent,
    full_grad,
    in_index,
    out_index,
    feature,
    pair,
    lr,
    steps,
    key,
    temp=1e0,
):
    As, Gs, acts, resids, grads, loss_sqnorm = kfac_observe(
        model, loss_function, state, tangent, pair
    )
    A, G, (act_in,), (act_out,) = (
        As[out_index],
        Gs[out_index],
        acts[in_index],
        acts[out_index],
    )
    res, grad = resids[out_index], grads[out_index]
    Ginv = jnp.linalg.pinv(tikhonov(G, 1e-1 * meandiag(G)))
    atikh = 0e-1 * meandiag(A)
    res = layer_residual(A, res, act_out)
    lin_grad = full_grad["params"][f"layers_{out_index}"]["linear"]["kernel"]
    layer_score = layer_baseline(A, G, lin_grad)

    @flax.struct.dataclass
    class State:
        lr: float
        rscore: float
        feature: Any

    init_state = State(lr, 0.0, feature)

    def fun(state, key):
        new_rscore, new_feature, accept_rate = kfac_mala_burst(
            fmodel,
            act_in,
            Ginv,
            res,
            state.feature,
            key,
            state.lr,
            temp=temp,
            score_norm=layer_score,
        )
        TARGET = 0.6
        changed_lr = jnp.where(accept_rate > TARGET, state.lr * 1.3, state.lr / 1.3)
        new_lr = jnp.where(jnp.abs(accept_rate - TARGET) > 0.3, changed_lr, state.lr)
        return State(new_lr, new_rscore, new_feature), accept_rate

    final_state, accepts = jax.lax.scan(fun, init_state, jax.random.split(key, steps))
    return final_state.rscore, final_state.feature, layer_score, loss_sqnorm


def layer_residual(A, res, act):
    Ainv = jnp.linalg.pinv(A)
    Cra = lax.pmean(res[:, None] * act[None, :], "batch")
    lres = res - Cra @ Ainv @ act
    return lres


def kfac_direct_eval(
    model,
    loss_function,
    fmodel,
    state,
    tangent,
    full_grad,
    in_index,
    out_index,
    feature,
    pair,
):
    As, Gs, acts, resids, _, loss_sqnorm = kfac_observe(
        model, loss_function, state, tangent, pair
    )
    A, G, (act_in,), (act_out,), res = (
        As[out_index],
        Gs[out_index],
        acts[in_index],
        acts[out_index],
        resids[out_index],
    )
    Ginv = jnp.linalg.pinv(G)
    res = layer_residual(A, res, act_out)
    rscore = kfac_single_eval(fmodel, act_in, Ginv, res, feature)
    lin_grad = full_grad["params"][f"layers_{out_index}"]["linear"]["kernel"]
    layer_score = layer_baseline(A, G, lin_grad)
    return rscore, layer_score, loss_sqnorm


def meandiag(matrix):
    assert len(matrix.shape) == 2
    N = matrix.shape[0]
    return jnp.trace(matrix) / N


def tikhonov(matrix, strength):
    assert len(matrix.shape) == 2
    N = matrix.shape[0]
    diag = jnp.eye(N) * strength
    return matrix + diag


def layer_baseline(A, G, lin_grad):
    """lin_grad: best estimate of true gradient on layer linear kernel"""

    corr = lin_grad
    Ainv = jnp.linalg.pinv(A)
    Ginv = jnp.linalg.pinv(G)
    normed_corr = Ainv @ corr @ Ginv
    score = jnp.sum(corr * normed_corr)
    return score


def kfac_direct_sgd(
    model,
    loss_function,
    fmodel,
    state,
    tangent,
    full_grad,
    in_index,
    out_index,
    feature,
    pair,
    eps,
    steps,
):
    As, Gs, acts, resids, grads, loss_sqnorm = kfac_observe(
        model, loss_function, state, tangent, pair
    )
    A, G, (act_in,), (act_out,) = (
        As[out_index],
        Gs[out_index],
        acts[in_index],
        acts[out_index],
    )
    res, grad = resids[out_index], grads[out_index]
    Ginv = jnp.linalg.pinv(tikhonov(G, 1e-1 * meandiag(G)))
    atikh = 0e-1 * meandiag(A)
    res = layer_residual(A, res, act_out)
    lin_grad = full_grad["params"][f"layers_{out_index}"]["linear"]["kernel"]
    layer_score = layer_baseline(A, G, lin_grad)

    @flax.struct.dataclass
    class State:
        eps: float
        rscore: float
        feature: Any

    init_state = State(eps, 0.0, feature)

    def body_fun(_, state):
        new_rscore, new_feature = kfac_single_sgd(
            fmodel, act_in, Ginv, res, state.feature, state.eps, atikh
        )

        rscore_increased = new_rscore / state.rscore > 1.0
        return State(
            jnp.where(rscore_increased, eps, eps / 3.0), new_rscore, new_feature
        )

    final_state = jax.lax.fori_loop(0, steps, body_fun, init_state)
    return final_state.rscore, final_state.feature, layer_score, loss_sqnorm


class Proposer:

    def __init__(self, template, model, key_iter):
        self.template = template
        self.model = model
        self.key_iter = key_iter

    def new_feature(self, null, input_size, output_size=1, key=None):

        if key is None:
            key = next(self.key_iter)
        active = int(sum(~null))
        total = len(null)

        input_shape = (3, 3, active)
        layer = self.template.layer(output_size)
        feature = layer.init(key, jnp.zeros(input_shape))
        assert input_size >= active
        feature = layer.apply(feature, input_size - active, method=layer.pad_inputs)
        return feature

    def embed_feature(self, state, feature, layer_index):
        lift = partial(nn.apply(Layers.lift, self.model), state, layer_index)
        assert not lift(Layer.dormant)
        null = lift(Layer.null)
        assert null.any()
        idx = null.argmax()
        state = state.unfreeze()
        name = f"layers_{layer_index}"
        state["params"][name] = lift(Layer.insert_feature, feature.unfreeze(), idx)[
            "params"
        ]
        return flax.core.frozen_dict.freeze(state)

    def embed_layer(self, state, feature, layer_index):
        unpadded_basis = feature["params"]["linear"]["kernel"]
        unpadded_shift = feature["params"]["linear"]["bias"]

        pad_amount = self.template.capacities[layer_index] - unpadded_shift.shape[-1]
        assert pad_amount >= 0
        padded_basis = nets.pad_axis(unpadded_basis, pad_amount, axis=-1)
        padded_shift = nets.pad_axis(unpadded_shift, pad_amount, axis=-1)

        new_state = self.model.apply(
            state,
            dims=self.template.contents,
            index=layer_index,
            basis=padded_basis,
            shift=padded_shift,
            method=self.model.activate_layer,
        )
        return new_state

    def get_input_size(self, state, layer_index):
        return self.model.apply(
            state, index=layer_index, func=Layer.input_size, method=self.model.lift
        )

    def get_output_size(self, state, layer_index):
        return self.model.apply(
            state, index=layer_index, func=Layer.output_size, method=self.model.lift
        )

    def get_input_null(self, state, layer_index):

        if layer_index == 0:

            input_size = self.get_input_size(state, 0)
            return jnp.zeros((input_size,), dtype=jnp.bool_)
        else:
            return self.model.apply(
                state, index=layer_index - 1, func=Layer.null, method=self.model.lift
            )

    def generate_feature(
        self, state, layer_index, key=None, kill_lin=False, in_index=None, output_size=1
    ):
        if in_index is None:
            in_index = layer_index
        feature = self.new_feature(
            null=self.get_input_null(state, in_index),
            input_size=self.get_input_size(state, layer_index),
            output_size=output_size,
            key=key,
        )
        if kill_lin:
            feature = feature.copy(
                {
                    "params": feature["params"].copy(
                        {
                            "equivariant_0": feature["params"]["equivariant_0"].copy(
                                {"w_lin": jnp.array([0.0])}
                            )
                        }
                    )
                }
            )

        return feature

    def propose_feature(self, state, layer_index):
        raise NotImplementedError
        if layer_index == 0:
            input_size = self.model.apply(
                state, index=0, func=Layer.input_size, method=self.model.lift
            )
            input_null = jnp.zeros((input_size,), dtype=jnp.bool_)
        else:
            input_null = self.model.apply(
                state, index=layer_index - 1, func=Layer.null, method=self.model.lift
            )
        input_null = self.get_input_null(state, layer_index)
        feature = self.new_feature(input_null)
        fstate = self.embed_feature(state, feature, layer_index)
        return feature, fstate

    def verify_state(self, state):
        nulls = self.model.apply(state, method=self.model.nulls)
        for i, (c, n) in enumerate(zip(self.template.contents, nulls)):
            if c is not None:
                sum_not_null = jnp.sum(~n)
                assert (
                    c == sum_not_null
                ), f"content-null mismatch at layer {i} with content {c} and sum(~null) {sum_not_null}"


@dataclass
class Sampler:
    key_iter: Any
    dataset: Any
    labels: Any
    batch_size: Optional[int]

    def num_batches(self):
        if self.batch_size is None:
            return 1
        DLEN = self.dataset.shape[0]
        num_batches = DLEN // self.batch_size
        return num_batches

    def batches(self):
        if self.batch_size is None:
            yield self.dataset, self.labels
            return
        DLEN = self.dataset.shape[0]
        pindices = jax.random.permutation(next(self.key_iter), DLEN)
        num_batches = DLEN // self.batch_size
        for batch_num in range(num_batches):
            bindices = pindices[batch_num * self.batch_size :][: self.batch_size]
            bdataset = self.dataset[bindices, ...]
            blabels = self.labels[bindices, ...]
            yield bdataset, blabels


def process_task(cfg):
    task_type = cfg["task"]["type"].get("regression")
    if task_type == "regression":
        seed = cfg["task"]["seed"].get(cfg["meta"]["seed"].get())
        key = key_iter(seed)
        task = SameFamilyRegression(cfg, next(key))
        train = task.get_data(cfg, test=False, key=next(key))
        test = task.get_data(cfg, test=True, key=next(key))
        out_size = task.out_size
        return task, train, test, out_size
    elif task_type == "classification":
        if cfg["task"]["sklearn"].get(False):
            task = SklGenClass(cfg)
            train = task.get_data(cfg, test=False)
            test = task.get_data(cfg, test=True)
            out_size = test[1].max() + 1
            return task, train, test, out_size
        else:
            task = ImgVecClass(cfg)
            train = task.get_data(cfg, test=False)
            test = task.get_data(cfg, test=True)
            out_size = test[1].max() + 1
            return task, train, test, out_size

    else:
        raise ValueError(f"unrecognised task type: {task_type}")


@flax.struct.dataclass
class TrainState:
    epoch: int
    contents: Sequence[Optional[int]]
    solver_state: Any


def make_feature(p, s, c, l, v):
    kernel = 1 / (1e-6 + s)
    bias = -p * kernel
    return flax.core.frozen_dict.freeze(
        {
            "params": {
                "equivariant_0": {
                    "w_const": jnp.array([c]),
                    "w_lin": jnp.array([l]),
                    "w_vec": jnp.array([v]),
                },
                "linear": {
                    "bias": jnp.array([bias]),
                    "kernel": jnp.array([[kernel]]),
                },
            }
        }
    )


def locate_linear(kernel, bias):
    position = -bias / kernel
    scale = 1 / (1e-6 + kernel)
    return (position, scale)


def locate_feature(feature):
    kernel = feature["params"]["linear"]["kernel"][0, 0]
    bias = feature["params"]["linear"]["bias"][0]
    return locate_linear(kernel, bias)


def locate_neurons(state, layer=0, wrt=0):
    linear = state["params"][f"layers_{layer}"]["linear"]
    return locate_linear(linear["kernel"][wrt], linear["bias"])


def main():
    cfg = experiment_utils.get_cfg("experiment2")
    writer = experiment_utils.set_writer(cfg)

    key = key_iter(cfg["meta"]["seed"].get())

    task, (dataset, labels), (testset, testlabels), out_size = process_task(cfg)

    train_sampler = Sampler(
        key_iter=key_iter(cfg["meta"]["seed"].get()),
        dataset=dataset,
        labels=labels,
        batch_size=cfg["opt"]["batch_size"].get(),
    )

    (addition_batch,) = islice(train_sampler.batches(), 1)
    addition_validation_batch = (dataset, labels)

    example = dataset[0]

    ################################################################################################
    # cfg["net"]["contents"] = [11]  # Set your desired initial sizes here
    # cfg["net"]["capacities"] = [128]  # Set your desired maximum sizes here

    # capacities = tuple(cfg["net"]["capacities"].get() + [int(out_size)])
    # initial_contents = list(cfg["net"]["contents"].get() + [int(out_size)])
    ################################################################################################
    capacities = tuple(cfg["net"]["capacities"].get() + [int(out_size)])
    initial_contents = list(cfg["net"]["contents"].get())
    ################################################################################################

    # capacities = tuple(cfg["net"]["capacities"].get() + [int(out_size)])
    # initial_contents = list(cfg["net"]["contents"].get() + [int(out_size)])
    use_rationals = cfg["net"]["rational"].get(False)
    template = ModelTemplate(capacities, initial_contents, rational=use_rationals)
    solver = Solver(cfg, template, task, next(key), example)
    proposer = Proposer(
        solver.template, solver.model, key_iter(cfg["meta"]["propseed"].get())
    )

    initial_train_state = TrainState(0, initial_contents, solver.state)
    if (
        cfg["checkpointing"]["enable"].get(False)
        and cfg["checkpointing"]["restore"].get()
    ):
        ckpt_dir = (
            f"{cfg['checkpointing']['directory'].get()}/{cfg['meta']['name'].get()}"
        )
        initial_train_state = checkpoints.restore_checkpoint(
            ckpt_dir=ckpt_dir, target=initial_train_state
        )
        if initial_train_state.contents[-1] != initial_train_state.contents[-1]:
            print("output size changed")
            exit()
        solver.state = initial_train_state.solver_state
        template.contents = list(initial_train_state.contents)
    print(initial_train_state)

    def baseline_eval():
        return solver.last_natlen

    print(f"enabled indices: {template.enabled_layers()}")
    print(f"disabled indices: {template.disabled_layers()}")

    print("Initialising...")
    print_tree_type(solver.state["params"])

    max_epochs = cfg["opt"]["max_epochs"].get()
    rtpt = experiment_utils.get_rtpt(
        f"XNNs: {cfg['meta']['name'].get('untitled')}", max_iter=max_epochs
    )

    @dataclass
    class Proposal:
        feature: Any
        scale: float
        location: float
        theta: float

    def get_evaluator(layer_index, size=None):
        in_index, out_index = solver.template.in_out_indices(layer_index)

        return jax.experimental.maps.xmap(
            lambda state, ngrad, grad, feature, pair, eps, key, temp: kfac_direct_mala(
                solver.model,
                solver.task.loss_function,
                proposer.template.layer(1 if size is None else size),
                state,
                ngrad,
                grad,
                layer_index,
                out_index,
                feature,
                pair,
                eps,
                cfg["evo"]["steps"].get(),
                key,
                temp=temp,
            ),
            (
                [...],
                [...],
                [...],
                ["features", ...],
                ["batch", ...],
                ["features", ...],
                ["features", ...],
                [...],
            ),
            (["features", ...], ["features", ...], [...], [...]),
        )

    def get_validator(layer_index, size=None):
        in_index, out_index = solver.template.in_out_indices(layer_index)
        return jax.experimental.maps.xmap(
            lambda state, ngrad, grad, feature, pair: kfac_direct_eval(
                solver.model,
                solver.task.loss_function,
                proposer.template.layer(1 if size is None else size),
                state,
                ngrad,
                grad,
                layer_index,
                out_index,
                feature,
                pair,
            ),
            ([...], [...], [...], [...], ["batch", ...]),
            ([...], [...], [...]),
        )

    @dataclass
    class WidthModification:
        ratio: float
        layer_index: int
        new_state: Any

        def apply(self):
            assert solver.template.contents[self.layer_index] is not None
            prev_loss = solver.train_batch(
                (dataset, labels), observe_only=True, loud=True
            )
            solver.state = self.new_state
            new_size = solver.template.contents[self.layer_index] + 1
            solver.template.contents[self.layer_index] = new_size
            refresh_evaluators(layer_only=True)
            print(f"new size for layer {self.layer_index}: {new_size}")
            new_loss = solver.train_batch(
                (dataset, labels), observe_only=True, loud=True
            )
            assert new_loss / prev_loss < 1.001, "adding width made loss worse"

    @dataclass
    class DepthModification:
        ratio: float
        layer_index: int
        new_state: Any

        def apply(self):

            assert solver.template.contents[self.layer_index] is None
            prev_loss = solver.train_batch(
                (dataset, labels), observe_only=True, loud=True
            )
            old_state = solver.state
            solver.state = self.new_state

            if self.layer_index > 0:
                new_size = solver.template.contents[self.layer_index - 1]
            else:
                new_size = proposer.get_input_size(solver.state, layer_index=0)
            solver.template.contents[self.layer_index] = new_size
            refresh_evaluators()
            print(f"Created new layer at {self.layer_index} with size: {new_size}")
            new_loss = solver.train_batch(
                (dataset, labels), observe_only=True, loud=True
            )
            if new_loss / prev_loss >= 1.2:
                print(f"old state: {old_state}")
                print(f"new state: {self.new_state}")
                assert new_loss / prev_loss < 1.2, "adding layer made loss worse"

    save_index = 0
    just_added = False

    def consider_adding_width(evaluator, validator, layer_index, final=False):
        nonlocal save_index
        nonlocal just_added
        landscape_results = dict(dataset=dataset, datalabels=labels)
        neuron_positions, neuron_scales = locate_neurons(solver.state)
        landscape_results.update(
            dict(neuron_positions=neuron_positions, neuron_scales=neuron_scales)
        )

        baseline = baseline_eval()
        num_props = cfg["evo"]["proposals_per_layer"].get(10)

        in_index, out_index = solver.template.in_out_indices(layer_index)
        print(f"in and out for {layer_index}: {in_index}, {out_index}")
        featstack = jax.vmap(
            lambda key: proposer.generate_feature(
                solver.state, layer_index, key, in_index=in_index
            )
        )(jax.random.split(next(proposer.key_iter), num_props))

        init_positions, init_scales = jax.vmap(locate_feature)(featstack)
        landscape_results.update(
            dict(init_positions=init_positions, init_scales=init_scales)
        )

        tangent = insert_params(solver.state, solver.optimizer.read(solver.opt_state))
        tangent = jtm(jnp.zeros_like, tangent)
        full_grad = insert_params(
            solver.state, solver.optimizer.raw_grad(solver.opt_state)
        )

        vis_x0 = jnp.linspace(-3.0, 3.0, 100)
        vis_x1 = jnp.linspace(-3.0, 3.0, 100)
        vis_y = jax.experimental.maps.xmap(
            lambda p, x0, x1: solver.model.apply(p, jnp.array([x0, x1])),
            ([...], ["x0"], ["x1"]),
            (["x0", "x1", ...]),
        )(solver.state, vis_x0, vis_x1)
        landscape_results.update(dict(x0=vis_x0, x1=vis_x1, pred=vis_y))

        if cfg["evo"]["pure_kfac"].get(False):
            tangent = jtm(jnp.zeros_like, tangent)

        @flax.struct.dataclass
        class OptState:
            rmetrics: Any
            epss: Any
            featstack: Any

            def get_feature(self, i):
                return jtm(lambda arr: arr[i], self.featstack)

        init_eps = cfg["evo"]["initial_lr"].get()
        opt_state = OptState(
            jnp.zeros((num_props,)), jnp.ones((num_props,)) * init_eps, featstack
        )
        temp = cfg["evo"]["proposal_temperature"].get()

        def step(state):
            rmetrics, featstack, layer_score, loss_sqnorm = evaluator(
                solver.state,
                tangent,
                full_grad,
                state.featstack,
                addition_batch,
                state.epss,
                jax.random.split(next(proposer.key_iter), num_props),
                temp,
            )
            inc = rmetrics > state.rmetrics

            return (
                OptState(
                    rmetrics, jnp.where(inc, state.epss, state.epss / 3.0), featstack
                ),
                layer_score,
                loss_sqnorm,
            )

        for i in range(1):
            opt_state, layer_score, loss_sqnorm = step(opt_state)

            best = jax.random.categorical(
                next(proposer.key_iter),
                logits=opt_state.rmetrics / (temp * layer_score),
                shape=(),
            )

            best_raw = jnp.sum(
                jax.nn.softmax(opt_state.rmetrics / (temp * layer_score))
                * opt_state.rmetrics
            )
            best_ratio = 1.0 + best_raw / baseline

        final_positions, final_scales = jax.vmap(locate_feature)(opt_state.featstack)
        landscape_results.update(
            dict(final_positions=final_positions, final_scales=final_scales)
        )
        landscape_results.update(dict(selected=best, rmetrics=opt_state.rmetrics))

        best_local_ratio = best_raw / layer_score
        summary.scalar(f"local baseline {layer_index}", layer_score)
        summary.scalar(f"best local ratio {layer_index}", best_local_ratio)
        summary.scalar(f"best proposal {layer_index}", best_ratio)
        summary.scalar(f"best raw rmetric {layer_index}", best_raw)
        summary.scalar(f"normed rmetric {layer_index}", best_raw / loss_sqnorm)
        summary.scalar(f"normed baseline", baseline / loss_sqnorm)
        summary.scalar(f"loss sqnorm", loss_sqnorm)

        vmetric, vlayer_score, _ = validator(
            solver.state,
            tangent,
            full_grad,
            opt_state.get_feature(best),
            addition_validation_batch,
        )

        print(
            f"layer {layer_index} best_ratio: {best_raw/layer_score:.8f} ({best_raw:.8f}/{layer_score:.8f})"
        )

        summary.scalar(f"local validation baseline {layer_index}", vlayer_score)
        summary.scalar(
            f"best local validation ratio {layer_index}", vmetric / vlayer_score
        )
        summary.scalar(f"best raw validation metric {layer_index}", vmetric)

        going_to_add = (
            best_local_ratio > cfg["evo"]["thresh"].get()
            and best_raw / loss_sqnorm > cfg["evo"]["abs_thresh"].get()
        )

        if (going_to_add or just_added or is_final) and layer_index == 0:
            landscape_results.update(dict(just_added=just_added))
            just_added = False
            print("Saving feature landscape...")
            np.savez(f"./results/landscape{save_index:04d}", **landscape_results)
            save_index += 1
            if save_index >= 25:
                exit()

        if going_to_add:
            new_state = proposer.embed_feature(
                solver.state, opt_state.get_feature(best), layer_index
            )
            return WidthModification(
                ratio=best_ratio, layer_index=layer_index, new_state=new_state
            )
        else:
            return None

    def consider_inserting_layer(evaluator, validator, layer_index):

        assert solver.template.contents[layer_index] is None
        baseline = baseline_eval()
        num_props = cfg["evo"]["layer_proposals_per_layer"].get()

        in_index, out_index = solver.template.in_out_indices(layer_index)
        new_layer_size = jnp.sum(~proposer.get_input_null(solver.state, in_index))
        print(f"new layer size: {new_layer_size}")
        featstack = jax.vmap(
            lambda key: proposer.generate_feature(
                state=solver.state,
                layer_index=layer_index,
                key=key,
                in_index=in_index,
                output_size=new_layer_size,
            )
        )(jax.random.split(next(proposer.key_iter), num_props))

        def get_feature(i):
            return jtm(lambda arr: arr[i], featstack)

        tangent = insert_params(solver.state, solver.optimizer.read(solver.opt_state))
        full_grad = insert_params(
            solver.state, solver.optimizer.raw_grad(solver.opt_state)
        )
        if cfg["evo"]["pure_kfac"].get(False):
            tangent = jtm(jnp.zeros_like, tangent)

        @flax.struct.dataclass
        class OptState:
            rmetrics: Any
            epss: Any
            featstack: Any

        init_eps = 3e-1
        opt_state = OptState(
            jnp.zeros((num_props,)), jnp.ones((num_props,)) * init_eps, featstack
        )
        temp = cfg["evo"]["proposal_temperature"].get()

        def step(state):
            rmetrics, featstack, layer_score, loss_sqnorm = evaluator(
                solver.state,
                tangent,
                full_grad,
                state.featstack,
                addition_batch,
                state.epss,
                jax.random.split(next(proposer.key_iter), num_props),
                temp,
            )
            inc = rmetrics > state.rmetrics
            return (
                OptState(
                    rmetrics, jnp.where(inc, state.epss, state.epss / 3.0), featstack
                ),
                layer_score,
                loss_sqnorm,
            )

        for i in range(1):
            opt_state, layer_score, loss_sqnorm = step(opt_state)

            best = jax.random.categorical(
                next(proposer.key_iter),
                logits=opt_state.rmetrics / (temp * layer_score),
                shape=(),
            )

            best_raw = jnp.sum(
                jax.nn.softmax(opt_state.rmetrics / (temp * layer_score))
                * opt_state.rmetrics
            )
            best_ratio = 1.0 + best_raw / baseline

        best_local_ratio = best_raw / layer_score
        summary.scalar(f"local baseline {layer_index}", layer_score)
        summary.scalar(f"best layer local ratio {layer_index}", best_local_ratio)
        summary.scalar(f"best layer proposal {layer_index}", best_ratio)
        summary.scalar(f"best layer raw metric {layer_index}", best_raw)
        summary.scalar(f"normed layer rmetric {layer_index}", best_raw / loss_sqnorm)
        summary.scalar(f"normed layer baseline", baseline / loss_sqnorm)
        summary.scalar(f"layer loss sqnorm", loss_sqnorm)

        def make_invertible(feat):
            L = feat["params"]["linear"]["kernel"]

            u, s, vh = jnp.linalg.svd(L, full_matrices=False)

            s = jnp.clip(s, 1e-3 * jnp.mean(s), None)

            L = u @ jnp.diag(s) @ vh

            return feat.copy(
                {
                    "params": feat["params"].copy(
                        {"linear": feat["params"]["linear"].copy({"kernel": L})}
                    )
                }
            )

        print("making invertible...")
        best_feature = make_invertible(get_feature(best))

        vmetric, vlayer_score, _ = validator(
            solver.state, tangent, full_grad, best_feature, addition_validation_batch
        )

        summary.scalar(f"local validation baseline {layer_index}", vlayer_score)
        summary.scalar(
            f"best layer local validation ratio {layer_index}", vmetric / vlayer_score
        )
        summary.scalar(f"best layer raw validation metric {layer_index}", vmetric)

        cost_mul = cfg["evo"]["layer_cost_mul"].get()
        if cfg["evo"]["size_costing"].get():
            cost_mul *= new_layer_size
        adjusted_metric = best_raw / cost_mul
        print(
            f"layer {layer_index} adjusted ratio: {adjusted_metric/vlayer_score:.5f} ({adjusted_metric:.5f}/{vlayer_score:.5f})"
        )

        if (
            adjusted_metric / vlayer_score > cfg["evo"]["thresh"].get()
            and adjusted_metric / loss_sqnorm > cfg["evo"]["layer_abs_thresh"].get()
        ):

            new_state = proposer.embed_layer(solver.state, best_feature, layer_index)

            adjusted_ratio = 1.0 + adjusted_metric / baseline
            return DepthModification(
                ratio=adjusted_ratio, layer_index=layer_index, new_state=new_state
            )
        else:
            return None

    def diagnose():
        print("---- DIAGNOSIS ----:")
        solver_grad = insert_params(
            solver.state, solver.optimizer.SG.read(solver.opt_state)
        )
        print(f"solver_grad: {solver_grad}")
        solver_ngrad = insert_params(
            solver.state, solver.optimizer.read(solver.opt_state)
        )
        print(f"solver_ngrad: {solver_ngrad}")

        def eval_model(state):
            return jax.vmap(partial(solver.model.apply, state))(dataset)

        Y, backward = jax.vjp(eval_model, solver.state)

        def item_grad(l, y):
            return jax.grad(partial(solver.task.loss_function, l))(y)

        raw_dloss = jax.vmap(item_grad)(labels, Y)
        print(f"raw_dloss: {raw_dloss}")
        raw_grad = backward(raw_dloss)
        print(f"raw_grad: {raw_grad}")

        def forward(tan_state):
            Y, out = jax.jvp(eval_model, (solver.state,), (tan_state,))
            return out

        Jngrad = forward(solver_ngrad)
        print(f"Jngrad: {Jngrad}")
        out_res = raw_dloss - Jngrad
        print(f"out_res: {out_res}")
        raw_res = backward(out_res)
        print(f"raw_res: {raw_res}")
        Fngrad = backward(Jngrad)
        print(f"Fngrad: {Fngrad}")

    evaluators = [get_evaluator(i) for i in range(len(solver.template.contents[:-1]))]
    validators = [get_validator(i) for i in range(len(solver.template.contents[:-1]))]

    def refresh_evaluators(layer_only=False):
        for layer_index, con in enumerate(solver.template.contents[:-1]):
            if con is None:
                in_index, out_index = solver.template.in_out_indices(layer_index)
                new_layer_size = jnp.sum(
                    ~proposer.get_input_null(solver.state, in_index)
                )
                evaluators[layer_index] = get_evaluator(
                    layer_index, size=new_layer_size
                )
                validators[layer_index] = get_validator(
                    layer_index, size=new_layer_size
                )
            elif not layer_only:
                evaluators[layer_index] = get_evaluator(layer_index)
                validators[layer_index] = get_validator(layer_index)

    refresh_evaluators()

    ################################################################################################
    print("Initial network architecture:")
    for i, layer_size in enumerate(template.contents):
        if layer_size is not None:
            print(f"Layer {i}: {layer_size} neurons")
        else:
            print(f"Layer {i}: Not active")

    initial_params = sum(
        jax.tree_util.tree_leaves(
            jax.tree_map(lambda x: x.size, solver.state["params"])
        )
    )
    print(f"Initial number of parameters: {initial_params}")
    print("\n" + "=" * 50 + "\n")  # This adds a separator for clarity
    ################################################################################################

    layer_cooldown = 0
    initial_epoch = initial_train_state.epoch
    for epoch in range(initial_epoch, max_epochs):
        is_final = (epoch == max_epochs - 1) or (epoch == 0)

        for bidx, batch in enumerate(train_sampler.batches()):

            writer.set_as_default(step=bidx + train_sampler.num_batches() * epoch)
            bdata, blabels = batch
            solver.train_batch(batch)

        solver.test_batch((testset, testlabels))
        solver.test_acc((testset, testlabels))
        solver.train_acc((dataset, labels))
        solver.train_batch((dataset, labels), observe_only=True)

        proposer.verify_state(solver.state)

        if epoch % cfg["evo"]["cooldown"].get(20) == 0 or is_final:
            pass  # Disabled network evolution

        if (
            cfg["checkpointing"]["enable"].get(False)
            and epoch % cfg["checkpointing"]["cooldown"].get() == 0
        ):

            ckpt_dir = (
                f"{cfg['checkpointing']['directory'].get()}/{cfg['meta']['name'].get()}"
            )

            checkpoint_data = TrainState(
                epoch + 1, tuple(template.contents), solver.state
            )
            checkpoints.save_checkpoint(
                ckpt_dir=ckpt_dir, target=checkpoint_data, step=epoch
            )
            pass

        rtpt.step()

    ################################################################################################
    print("Final network architecture:")
    for i, layer_size in enumerate(template.contents):
        if layer_size is not None:
            print(f"Layer {i}: {layer_size} neurons")
        else:
            print(f"Layer {i}: Not active")

    final_params = sum(
        jax.tree_util.tree_leaves(
            jax.tree_map(lambda x: x.size, solver.state["params"])
        )
    )
    print(f"Final number of parameters: {final_params}")

    print(f"\nChange in number of parameters: {final_params - initial_params}")
    print(f"Relative change: {(final_params - initial_params) / initial_params:.2%}")
    ################################################################################################

    exit()


if __name__ == "__main__":
    main()


import numpy
from dataclasses import dataclass
from numpy import ones as _ones, zeros as _zeros
from myia.dtype import Array, Tuple, pytype_to_myiatype
from myia.abstract import InferenceError
from myia.composite import grad
from myia.pipeline import standard_pipeline
from myia.prim.py_implementations import array_reduce, scalar_add

from .test_compile import parse_compare
from .test_grad import grad_test
from .test_infer import infer_std, af64_of, af32_of
from .common import MA, MB, MC, MD


MA = MA * 0.1
MB = MB * 0.1
MC = MC * 0.1
MD = MD * 0.1


#############
# Utilities #
#############


def ones(*shp, dtype='float64'):
    return _ones(shp, dtype=dtype)


def zeros(*shp, dtype='float64'):
    return _zeros(shp, dtype=dtype)


#########
# Model #
#########


@dataclass(frozen=True)
class TanhLayer:
    W: Array
    b: Array

    def apply(self, input):
        return numpy.tanh(input @ self.W + self.b)


@dataclass(frozen=True)
class RNNLayer:
    W: Array
    R: Array
    b: Array
    h0: Array

    def step(self, x, h_tm1):
        return numpy.tanh((x @ self.W) + (h_tm1 @ self.R) + self.b)

    def apply(self, x):
        h = self.h0
        for e in x:
            h = self.step(e, h)
        # Maybe collect and return the full list of outputs?
        return h


@dataclass(frozen=True)
class Model:
    layers: Tuple

    def apply(self, x):
        for layer in self.layers:
            x = layer.apply(x)
        return x


#########
# Tests #
#########


def make_model(dtype='float64'):
    return Model(
        layers=(
            TanhLayer(MA(6, 9, dtype=dtype), zeros(1, 9, dtype=dtype)),
            TanhLayer(MB(9, 10, dtype=dtype), zeros(1, 10, dtype=dtype)),
            TanhLayer(MC(10, 8, dtype=dtype), zeros(1, 8, dtype=dtype)),
        )
    )


def make_rnn():
    return Model(
        layers=(
            RNNLayer(MA(6, 10), MB(10, 10), zeros(1, 10), zeros(2, 10)),
            TanhLayer(MC(10, 8), zeros(1, 8)),
        )
    )


def cost(model, x, y):
    yy = model.apply(x)
    diff = (yy - y)
    return (array_reduce(scalar_add, diff ** 2, ())).item()


@infer_std(
    (make_model(), MC(3, 6), af64_of(3, 8)),
    (make_model('float32'), MC(3, 6), InferenceError),
    (make_model('float32'), MC(3, 6, dtype='float32'), af32_of(3, 8)),
    (make_model(), MC(3, 9), InferenceError),
)
def test_forward_infer(model, x):
    return model.apply(x)


@parse_compare((make_model(), MC(3, 6)), array=True)
def test_forward_specialize(model, x):
    return model.apply(x)


@parse_compare((make_model(), MC(3, 6)), array=True, profile=True)
def test_forward_profile(model, x):
    return model.apply(x)


@infer_std(
    (make_model(), MC(3, 6), MC(3, 8), make_model()),
    (make_model(), MC(3, 6), MC(3, 9), InferenceError),
    (make_model('float32'), MC(3, 6), MC(3, 8), InferenceError),
    (make_model('float32'),
     MC(3, 6, dtype='float32'),
     MC(3, 8, dtype='float32'),
     make_model('float32')),
    (make_rnn(), [MA(2, 6), MB(2, 6), MC(2, 6)], MC(2, 8), make_rnn()),
)
def test_backward_infer(model, x, y):
    return grad(cost)(model, x, y)


@grad_test((make_model(), MC(3, 6), MD(3, 8)),
           pipeline=standard_pipeline,
           rel_error=1e-1)
def test_backward_specialize(model, x, y):
    return cost(model, x, y)


@grad_test((make_rnn(), [MA(2, 6), MB(2, 6), MC(2, 6)], MC(2, 8)),
           pipeline=standard_pipeline,
           rel_error=1e-1)
def test_backward_specialize_rnn(model, x, y):
    return cost(model, x, y)

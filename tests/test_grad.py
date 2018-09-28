
import pytest
import numpy as np

from myia.api import standard_debug_pipeline
from myia.composite import grad
from myia.debug.finite_diff import gen_variants, GradTester
from myia.dtype import JTagged as JT, SensitivityMap as SM
from myia.infer import InferenceError
from myia.prim import ops as P
from myia.prim.py_implementations import scalar_cast, J, Jinv, \
    array_reduce, scalar_add, typeof
from myia.prim.grad_implementations import augmented_graphs
from myia.prim.py_implementations import py_implementations as pyi

from .common import B, T, L, F, i16, i32, i64, u64, f16, f32, f64, \
    li32, li64, lf64, ai16, ai32, ai64, af16, af32, af64, Nil, \
    Point, Point_t, Point3D, Point3D_t, Thing_f, Thing_ftup, mysum
from .test_infer import infer, infer_std, af16_of, af32_of


@infer_std(
    type=[
        (f32, JT[f32]),
        (T[f32, f64], JT[T[f32, f64]])
    ]
)
def test_J(x):
    return J(x)


@infer_std(
    type=[
        (JT[f32], f32),
        (JT[T[f32, f64]], T[f32, f64]),
        (f32, InferenceError)
    ]
)
def test_Jinv(x):
    return Jinv(x)


@infer_std(
    type=[
        (f32, f32, InferenceError),
        (JT[f32], JT[f32], T[JT[f32], T[SM, f32, f32]])
    ]
)
def test_J_fn(x, y):
    def f(a, b):
        return a * b

    jf = J(f)
    res, bprop = jf(x, y)
    return res, bprop(1)


@infer_std(
    type=[
        (i64, i64, i64),
        (f32, f32, f32),
        (f64, f64, f64),
        (f32, f64, InferenceError),
        (af32_of(), af32_of(), af32),
    ],
    shape=[
        (af32_of(), af32_of(), ()),
        (af32_of(2, 5), af32_of(2, 5), InferenceError),
    ]
)
def test_grad_simple(x, y):
    def f(x, y):
        return x * y

    return grad(f)(x, y)


@infer_std(
    type=[
        (i64, i64),
        (f32, f32),
        (f64, f64),
    ]
)
def test_grad_cast(x):
    def f(x):
        return scalar_cast(x, f16)

    return grad(f)(x)


@infer_std(
    type=[
        (af16_of(2, 5), af16_of(2, 5), af16),
    ],
    shape=[
        (af16_of(2, 5), af16_of(2, 5), (2, 5)),
    ]
)
def test_grad_reduce(xs, ys):
    def f(xs, ys):
        return array_reduce(scalar_add, xs * ys, ())

    return grad(f)(xs, ys)


def test_GradTester():

    def f(x, y):
        return x / y

    def df(x, y, dz):
        return dz / y, -dz * x / (y * y)

    arglist = (
        (7.3, 4.2),
        (np.array([[4.3, 2.0], [5.1, 7.7], [3.4, 8.2]]),
         np.array([[1.2, 5.0], [3.3, 2.7], [6.9, 7.2]])),
    )

    for args in arglist:
        gtest = GradTester(
            fn=f,
            gfn=df,
            args=args,
            argnames=['x', 'y'],
            outnames=['out']
        )

        gtest.assert_match()


def test_GradTester_outtup():

    def f(x, y):
        return x * y, x / y

    def df(x, y, dz):
        dz1, dz2 = dz
        return (dz1 * y + dz2 / y,
                dz1 * x + -dz2 * x / (y * y))

    gtest = GradTester(
        fn=f,
        gfn=df,
        args=(7.3, 4.2),
        argnames=['x', 'y'],
        outnames=None
    )

    gtest.assert_match()


prim_tests = {
    P.scalar_add: [(-7.1, 4.3)],
    P.scalar_sub: [(-7.1, 4.3)],
    P.scalar_mul: [(-7.1, 4.3)],
    P.scalar_div: [(-7.1, 4.3)],
    P.scalar_pow: [(7.1, 4.3), (5.3, -1.2)],
    P.scalar_uadd: [(-7.1,)],
    P.scalar_usub: [(-7.1,)],
    # P.scalar_gt: [(-7.1, 4.3)],
    # P.scalar_lt: [(-7.1, 4.3)],
    # P.scalar_ge: [(-7.1, 4.3)],
    # P.scalar_le: [(-7.1, 4.3)],
}


@pytest.mark.parametrize('prim,cases', prim_tests.items())
def test_prim_grads(prim, cases):
    primg = augmented_graphs[prim]
    g = grad.make_gf(primg, primg.parameters,
                     dbg=primg.debug, sens_param=True, get_all=True)
    pip = standard_debug_pipeline \
        .select('resolve', 'infer', 'specialize', 'opt', 'export') \
        .make()
    types = [{'type': typeof(arg)} for arg in cases[0]]
    res = pip(graph=g, argspec=[*types, types[0]])

    gtest = GradTester(
        fn=pyi[prim],
        gfn=res['output'],
        args=cases[0],
        argnames=[None]*len(cases[0]),
        outnames=None
    )

    gtest.assert_match()

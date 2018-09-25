
from myia.composite import grad
from myia.dtype import JTagged as JT, SensitivityMap as SM
from myia.infer import InferenceError
from myia.prim.py_implementations import scalar_cast, J, Jinv, \
    array_reduce, scalar_add

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

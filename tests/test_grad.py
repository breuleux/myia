
import pytest
import numpy as np
from types import FunctionType

from myia.api import standard_debug_pipeline, Optimizer
from myia.composite import grad
from myia.debug.finite_diff import GradTester
from myia.dtype import JTagged as JT, EnvType as Env
from myia.grad import J as realJ
from myia.infer import InferenceError
from myia.opt import lib as optlib, CSE
from myia.pipeline import pipeline_function
from myia.prim import ops as P, Primitive
from myia.prim.py_implementations import scalar_cast, J, Jinv, \
    array_reduce, scalar_add, typeof
from myia.prim.py_implementations import py_implementations as pyi

from .common import T, i64, f16, f32, f64, af16, af32
from .test_infer import infer_std, af16_of, af32_of


step_optgrad = Optimizer.partial(
    phases=dict(
        main=[
            optlib.simplify_always_true,
            optlib.simplify_always_false,
            optlib.inline_core,
            optlib.simplify_partial,
            optlib.replace_applicator,
            optlib.bubble_op_tuple_unary,
            optlib.elim_identity,
        ],
        grad=[
            optlib.expand_J,
        ],
        renormalize='renormalize',
        elimj=[
            optlib.elim_j,
            optlib.elim_jinv,
            optlib.elim_jct,
            optlib.elim_j_jinv,
            optlib.elim_jinv_j,
        ],
        cse=CSE.partial(report_changes=False),
    )
)


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
        (JT[f32], JT[f32], T[JT[f32], T[Env, f32, f32]])
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


@pipeline_function
def grad_wrap(self, graph):
    if isinstance(graph, Primitive):
        jg = realJ(graph, self.resources)
        g = grad.make_gf(jg, jg.parameters,
                         dbg=jg.debug, sens_param=True, get_all=True)
    else:
        g = grad.make_gf(graph, graph.parameters,
                         dbg=graph.debug, sens_param=True, get_all=True,
                         apply_j=True)
    return g


def _grad_test(fn, obj, args, sens_type=f64):
    in_types = [{'type': typeof(arg)} for arg in args]
    sens_type = {'type': sens_type}
    steps = ['resolve', 'infer', 'specialize', 'prepare',
             'validate', 'export']
    if isinstance(obj, FunctionType):
        res = standard_debug_pipeline \
            .select('parse', *steps) \
            .insert_before('infer', wrap=grad_wrap) \
            .insert_after('prepare', opt=step_optgrad) \
            .run(input=obj, argspec=[*in_types, sens_type])
    else:
        res = standard_debug_pipeline \
            .select(*steps) \
            .insert_before(wrap=grad_wrap) \
            .insert_after('prepare', opt=step_optgrad) \
            .run(graph=obj, argspec=[*in_types, sens_type])

    gtest = GradTester(
        fn=fn,
        gfn=res['output'],
        args=args,
        argnames=[f'in{i}' for i in range(len(args))],
        outnames=None
    )
    gtest.assert_match()


@pytest.mark.parametrize('prim,cases', prim_tests.items())
def test_prim_grads(prim, cases):
    for case in cases:
        _grad_test(pyi[prim], prim, case)


def grad_test(*tests):
    """Decorate a function to parse and run it against pure Python.

    Returns a unit test that will parse the function, and then for
    each `inputs` tuple in `tests` it will check that the pure Python,
    undecorated function returns that same output.

    Arguments:
        tests: One or more inputs tuple.

    """

    def decorate(fn):
        def test(args):
            if not isinstance(args, tuple):
                args = (args,)

            _grad_test(fn, fn, args)

        m = pytest.mark.parametrize('args', list(tests))(test)
        m.__orig__ = fn
        return m
    return decorate


@grad_test((13.0, 14.0))
def test_null(x, y):
    """Test null gradient."""
    return 10.0 + 28.0 / 43.0


@grad_test((1.0, 4.0), (5.0, -13.0))
def test_grad_add(x, y):
    return x + y


@grad_test((3.0, 4.0))
def test_grad_expr(x, y):
    return x**3.0 * y**4.0


@grad_test((3.0,))
def test_constant(x):
    """Test the use of a literal in the expression."""
    return 18.0 * x


@grad_test((3.0,))
def test_dup_args_in_call(x):
    """The naive gradient update rule fails when a function's arguments
    contain the same variable more than once."""
    return x * x


@grad_test((3.0,))
def test_quadruple_args_in_call(x):
    """Test that duplicated arguments still cause no problem even if
    there are four of them."""
    def g(a, b, c, d):
        return a * b * c * d
    return g(x, x, x, x)


@grad_test((3.0, 5.0))
def test_tuples(x, y):
    tup = x + y, x * y
    z = tup[0] + tup[1]
    return z


@grad_test((4.0, 5.0))
def test_hof(a, b):
    """Test higher order functions."""
    def f(g, x):
        return g(x) * g(x + 10.0)

    def g(x):
        return x * b

    return f(g, a) + f(g, b)


@grad_test((4.0, 5.0))
def test_simple_closure(a, b):
    """Test some trivial closures."""
    def f():
        return a + 1.0

    def g():
        return b + 2.0
    return f() * g()


@grad_test((4.0,))
def test_closure(a):
    """This is the closure test in the paper."""
    def x1(b):

        def x4(c):
            return b
        return x4
    x2 = x1(a)
    x3 = x2(1.0)
    return x3


@grad_test((4.0, 5.0), (6.4, -7.8))
def test_if(a, b):
    # This is max, but what this is really testing is the most basic
    # if statement, so I prefer to name the test 'test_if'
    if a > b:
        return a
    else:
        return b


@grad_test((4.0, 5.0), (6.4, -7.8))
def test_if2(a, b):
    # This is max, but what this is really testing is the most basic
    # if statement, so I prefer to name the test 'test_if'
    if a > b:
        return a * a
    else:
        return b + b


@grad_test(4.1,)
def test_fact(x):
    def fact(n):
        if n <= 1:
            return 1
        else:
            return n * fact(n - 1)
    return fact(x)


@grad_test((4.0,))
def test_while(x):
    rval = x
    while rval < 100:
        rval = rval * rval
    return rval


@grad_test((4.0, 5.0, 2.0),)
def test_while_2(x, y, z):
    rval = 0
    # Cannot compare to 0 or finite diff is unstable
    while x > -0.1:
        rval = rval + y
        x = x - z
    return rval


@grad_test(2.0,)
def test_pow10(x):
    v = x
    j = 0
    while j < 3:
        i = 0
        while i < 3:
            v = v * x
            i = i + 1
        j = j + 1
    return v

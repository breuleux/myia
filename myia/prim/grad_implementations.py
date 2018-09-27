"""Implementations of the primitives' gradients.

Each primitive is associated to an augmented function, which returns a pair of
the (augmented) original primitive's output and a backpropagator function.
"""

from math import log
from types import FunctionType

from ..api import standard_pipeline
from ..composite import zeros_like
from ..dtype import newenv
from ..info import NamedDebugInfo, About
from ..ir import Constant, Graph, manage, clone
from ..utils import Registry

from . import ops as primops
from .ops import Primitive
from .py_implementations import \
    Jinv, J


parse = standard_pipeline \
    .select('parse') \
    .make_transformer('input', 'graph')


def bprop_to_augm(prim, fn):
    """Given a function for the bprop, make the augmented function."""
    info = NamedDebugInfo(prim=prim, name=prim.name)

    bprop = parse(fn)
    bprop.debug.name = None
    bprop.debug.about = About(info, 'grad_bprop')  # type: ignore
    bprop.output = bprop.apply(
        primops.make_tuple,
        newenv,
        *bprop.output.inputs[1:]
    )

    *args, out_param, dout = bprop.parameters

    with About(info, 'grad_fprop'):
        outer = Graph()
        # outer.transforms['primal'] = prim
        outer.output = Constant(None)

    mng = manage(bprop, outer)

    transf_args = []
    for p in args:
        with About(p.debug, 'grad_fprop'):
            outer_p = outer.add_parameter()
        with About(p.debug, 'equiv'):
            transf_p = outer.apply(primops.Jinv, outer_p)
        mng.replace(p, transf_p)
        transf_args.append(transf_p)

    with About(out_param.debug, 'equiv'):
        out_value = outer.apply(prim, *transf_args)

    mng.replace(out_param, out_value)

    with About(out_param.debug, 'grad_sens'):
        new_dout = bprop.add_parameter()
        mng.replace(dout, new_dout)
        # We remove all parameters except new_dout
        bprop.parameters = [new_dout]

    result = outer.apply(primops.J, out_value)
    outer.output = outer.apply(
        primops.make_tuple,
        result,
        bprop
    )
    return clone(outer)


augmented_graphs = Registry()
register = augmented_graphs.register


def register_bprop(prim):
    """Register an augmented function for prim, given a backpropagator."""
    def deco(fn):
        fn2 = bprop_to_augm(prim, fn)
        return register(prim)(fn2)
    return deco


def register_augm(prim):
    """Register an augmented function for prim."""
    from ..debug.label import short_labeler, short_relation_symbols as syms

    def deco(fn):
        fn2 = parse(fn)
        for g in manage(fn2, weak=True).graphs:
            name = short_labeler.name(g)
            name = name.replace('__fprop__', syms['grad_fprop'])
            g.debug.name = name.replace('__bprop__', syms['grad_bprop'])
        # fn2.transforms['primal'] = prim
        return register(prim)(fn2)
    return deco


@register_bprop(primops.scalar_add)
def bprop_scalar_add(x, y, out, dout):
    """Backpropagator for primitive `scalar_add`."""
    return (dout, dout)


@register_bprop(primops.scalar_sub)
def bprop_scalar_sub(x, y, out, dout):
    """Backpropagator for primitive `scalar_sub`."""
    return (dout, -dout)


@register_bprop(primops.scalar_mul)
def bprop_scalar_mul(x, y, out, dout):
    """Backpropagator for primitive `scalar_mul`."""
    return (dout * y, dout * x)


@register_bprop(primops.scalar_div)
def bprop_scalar_div(x, y, out, dout):
    """Backpropagator for primitive `scalar_div`."""
    return (dout / y, -dout * out / y)


@register_bprop(primops.scalar_pow)
def bprop_scalar_pow(x, y, out, dout):
    """Backpropagator for primitive `scalar_pow`."""
    return (dout * (y * x ** (y - 1)),
            dout * log(x) * out)


@register_bprop(primops.scalar_uadd)
def bprop_scalar_uadd(x, out, dout):
    """Backpropagator for primitive `scalar_uadd`."""
    return (dout,)


@register_bprop(primops.scalar_usub)
def bprop_scalar_usub(x, out, dout):
    """Backpropagator for primitive `scalar_usub`."""
    return (-dout,)


@register_bprop(primops.scalar_gt)
def bprop_scalar_gt(x, y, out, dout):
    """Backpropagator for primitive `scalar_gt`."""
    return (zeros_like(x), zeros_like(y))


@register_bprop(primops.scalar_lt)
def bprop_scalar_lt(x, y, out, dout):
    """Backpropagator for primitive `scalar_lt`."""
    return (zeros_like(x), zeros_like(y))


@register_bprop(primops.scalar_ge)
def bprop_scalar_ge(x, y, out, dout):
    """Backpropagator for primitive `scalar_ge`."""
    return (zeros_like(x), zeros_like(y))


@register_bprop(primops.scalar_le)
def bprop_scalar_le(x, y, out, dout):
    """Backpropagator for primitive `scalar_le`."""
    return (zeros_like(x), zeros_like(y))


@register_bprop(primops.tuple_getitem)
def bprop_tuple_getitem(data, idx, out, dout):
    """Backpropagator for primitive `tuple_getitem`."""
    return (tuple_setitem(zeros_like(data), idx, dout),
            zeros_like(idx))


@register_bprop(primops.J)
def bprop_J(x, out, dout):
    """Backpropagator for primitive `J`."""
    return (Jinv(dout),)


@register_bprop(primops.Jinv)
def bprop_Jinv(x, out, dout):
    """Backpropagator for primitive `Jinv`."""
    return (J(dout),)


# @register_augm(primops.if_)
# def __fprop__if_(c, tb, fb):
#     """Backpropagator for primitive `if`."""
#     if Jinv(c):
#         res = tb()
#     else:
#         res = fb()

#     rval, branch_bprop = res

#     def __bprop__if_(dout):
#         zc = zeros_like(c)
#         value = branch_bprop(dout)[0]
#         if Jinv(c):
#             return (), zc, value, zeros_like(Jinv(fb))
#         else:
#             return (), zc, zeros_like(Jinv(tb)), value

#     return rval, __bprop__if_


class MakeTupleGradient:
    def specialize_from_types(self, types):
        g = Graph()

        params = [g.add_parameter() for t in types]
        jinv_params = [g.apply(primops.Jinv, p) for p in params]
        tup = g.apply(primops.make_tuple, *jinv_params)
        out = g.apply(primops.J, tup)

        b = Graph()
        dout = b.add_parameter()
        grads = [g.apply(primops.tuple_getitem, dout, i)
                 for i, p in enumerate(params)]
        b.output = g.apply(primops.make_tuple, newenv, *grads)

        g.output = g.apply(primops.make_tuple, out, b)

        return g


register(primops.make_tuple)(MakeTupleGradient())

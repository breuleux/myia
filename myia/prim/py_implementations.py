"""Implementations for the debug VM."""

from copy import copy
from typing import Callable
import numpy as np

from .. import dtype as types
from ..utils import Registry

from . import ops as primops


py_implementations: Registry[primops.Primitive, Callable] = Registry()
vm_implementations: Registry[primops.Primitive, Callable] = Registry()
py_register = py_implementations.register
vm_register = vm_implementations.register


def register(prim):
    """Register an implementation for this primitive.

    The same implementation will be used for both the VM and for the pure
    Python version.
    """
    def deco(fn):
        vm_register(prim)(lambda vm, *args: fn(*args))
        return py_register(prim)(fn)
    return deco


@register(primops.add)
def add(x, y):
    """Implement `add`."""
    return x + y


@register(primops.sub)
def sub(x, y):
    """Implement `sub`."""
    return x - y


@register(primops.mul)
def mul(x, y):
    """Implement `mul`."""
    return x * y


@register(primops.div)
def div(x, y):
    """Implement `div`."""
    return x / y


@register(primops.mod)
def mod(x, y):
    """Implement `mod`."""
    return x % y


@register(primops.pow)
def pow(x, y):
    """Implement `pow`."""
    return x ** y


@register(primops.uadd)
def uadd(x):
    """Implement `iadd`."""
    return x


@register(primops.usub)
def usub(x):
    """Implement `isub`."""
    return -x


@register(primops.eq)
def eq(x, y):
    """Implement `eq`."""
    return x == y


@register(primops.lt)
def lt(x, y):
    """Implement `lt`."""
    return x < y


@register(primops.gt)
def gt(x, y):
    """Implement `gt`."""
    return x > y


@register(primops.ne)
def ne(x, y):
    """Implement `ne`."""
    return x != y


@register(primops.le)
def le(x, y):
    """Implement `le`."""
    return x <= y


@register(primops.ge)
def ge(x, y):
    """Implement `ge`."""
    return x >= y


@register(primops.not_)
def not_(x):
    """Implement `not_`."""
    return not x


@register(primops.typeof)
def typeof(x):
    """Implement typeof."""
    if isinstance(x, types.Type) or isinstance(x, type):
        return types.Type
    elif isinstance(x, bool):
        return types.Bool()
    elif isinstance(x, int):
        return types.Int(64)
    elif isinstance(x, float):
        return types.Float(64)
    elif isinstance(x, tuple):
        return types.Tuple(map(typeof, x))
    elif isinstance(x, list) and len(x) > 0:
        type0, *rest = map(typeof, x)
        if any(t != type0 for t in rest):
            raise TypeError(f'All list elements should have same type')
        return types.List(type0)
    else:
        return types.External(type(x))


def hastype_helper(t, model):
    """Check that type t is represented by model."""
    if t == model:
        return True
    elif isinstance(model, type) and issubclass(model, types.Type):
        return isinstance(t, model)
    else:
        return False


@register(primops.hastype)
def hastype(x, t):
    """Implement `hastype`."""
    return hastype_helper(typeof(x), t)


@register(primops.cons_tuple)
def cons_tuple(head, tail):
    """Implement `cons_tuple`."""
    return (head,) + tail


@register(primops.head)
def head(tup):
    """Implement `head`."""
    return tup[0]


@register(primops.tail)
def tail(tup):
    """Implement `tail`."""
    return tup[1:]


@py_register(primops.getitem)
def getitem(data, item):
    """Implement `getitem`."""
    return data[item]


@vm_register(primops.getitem)
def _vm_getitem(vm, data, item):
    """Implement `getitem`."""
    return vm.convert(data[item])


@register(primops.setitem)
def setitem(data, item, value):
    """Implement `setitem`."""
    if isinstance(data, tuple):
        return tuple(value if i == item else x
                     for i, x in enumerate(data))
    else:
        data2 = copy(data)
        data2[item] = value
        return data2


@vm_register(primops.getattr)
def _vm_getattr(vm, data, attr):
    """Implement `getattr`."""
    return vm.convert(getattr(data, attr))


py_setattr = setattr


@register(primops.setattr)
def setattr(data, attr, value):
    """Implement `setattr`."""
    data2 = copy(data)
    py_setattr(data2, attr, value)
    return data2


@register(primops.shape)
def shape(array):
    """Implement `shape`."""
    return array.shape


@py_register(primops.map_array)
def map_array(fn, array):
    """Implement `map_array`."""
    def f(ary):
        it = np.nditer([ary, None])
        for x, y in it:
            y[...] = fn(x)
        return it.operands[1]
    return np.apply_along_axis(f, 0, array)


@vm_register(primops.map_array)
def _map_array_vm(vm, fn, array):
    def fn_(x):
        return vm.call(fn, (x,))
    return map_array(fn_, array)


@py_register(primops.scan_array)
def scan_array(fn, init, array, axis):
    """Implement `scan_array`."""
    # This is inclusive scan because it's easier to implement
    # We will have to discuss what semantics we want later
    def f(ary):
        val = init
        it = np.nditer([ary, None])
        for x, y in it:
            val = fn(val, x)
            y[...] = val
        return it.operands[1]
    return np.apply_along_axis(f, axis, array)


@vm_register(primops.scan_array)
def _scan_array_vm(vm, fn, init, array, axis):
    def fn_(a, b):
        return vm.call(fn, [a, b])
    return scan_array(fn_, init, array, axis)


@py_register(primops.reduce_array)
def reduce_array(fn, init, array, axis):
    """Implement `reduce_array`."""
    def f(ary):
        val = init
        it = np.nditer([ary])
        for x in it:
            val = fn(val, x)
        return val
    return np.apply_along_axis(f, axis, array)


@vm_register(primops.reduce_array)
def _reduce_array_vm(vm, fn, init, array, axis):
    def fn_(a, b):
        return vm.call(fn, [a, b])
    return reduce_array(fn_, init, array, axis)


@register(primops.distribute)
def distribute(v, shape):
    """Implement `distribute`."""
    return np.broadcast_to(v, shape)


@register(primops.reshape)
def reshape(v, shape):
    """Implement `reshape`."""
    return np.reshape(v, shape)


@register(primops.dot)
def dot(a, b):
    """Implement `dot`."""
    return np.dot(a, b)


@register(primops.return_)
def return_(x):
    """Implement `return_`."""
    return x


@py_register(primops.maplist)
def maplist(f, xs):
    """Implement `maplist` in pure Python."""
    return list(map(f, xs))


@vm_register(primops.maplist)
def _maplist_vm(vm, f, xs):
    """Implement `maplist` for Myia's VM."""
    def f_(*args):
        return vm.call(f, args)
    return list(map(f_, xs))


@vm_register(primops.resolve)
def _resolve_vm(vm, data, item):
    """Implement `resolve` for the VM."""
    # There is no Python implementation for this one.
    value = data[item]
    return vm.convert(value)


@py_register(primops.partial)
def partial(f, *args):
    """Implement `partial`."""
    def res(*others):
        return f(*(args + others))
    return res


@register(primops.iter)
def _iter(xs):
    """Implement `iter`."""
    return (0, xs)


@register(primops.hasnext)
def _hasnext(it):
    """Implement `hasnext`."""
    n, data = it
    return n < len(data)


@register(primops.next)
def _next(it):
    """Implement `next`."""
    n, data = it
    return (data[n], (n + 1, data))

"""Miscellaneous utilities."""

import builtins
import inspect
import itertools
import sys
import numpy as np
from typing import Any, Dict, List, TypeVar
from colorama import AnsiToWin32
from dataclasses import dataclass


builtins_d = vars(builtins)


T1 = TypeVar('T1')
T2 = TypeVar('T2')


class Named:
    """A named object.

    This class can be used to construct objects with a name that will be used
    for the string representation.

    """

    def __init__(self, name):
        """Construct a named object.

        Args:
            name: The name of this object.

        """
        self.name = name

    def __repr__(self):
        """Return the object's name."""
        return self.name


UNKNOWN = Named('UNKNOWN')


class Registry(Dict[T1, T2]):
    """Associates primitives to implementations."""

    def __init__(self) -> None:
        """Initialize a Registry."""
        super().__init__()

    def register(self, prim):
        """Register a primitive."""
        def deco(fn):
            """Decorate the function."""
            self[prim] = fn
            return fn
        return deco


def repr_(obj: Any, **kwargs: Any):
    """Return unique string representation of object with additional info.

    The usual representation is `<module.Class object at address>`. This
    function returns `<module.Class(key=value) object at address>` instead, to
    make objects easier to identify by their attributes.

    Args:
        **kwargs: The attributes and their values that will be printed as part
            of the string representation.

    """
    name = f'{obj.__module__}.{obj.__class__.__name__}'
    info = ', '.join(f'{key}={value}' for key, value in kwargs.items())
    address = str(hex(id(obj)))
    return f'<{name}({info}) object at {address}>'


def list_str(lst: List):
    """Return string representation of a list.

    Unlike the default string representation, this calls `str` instead of
    `repr` on each element.

    """
    elements = ', '.join(str(elem) for elem in lst)
    return f'[{elements}]'


def flatten(lst):
    """Flatten a list of lists (one level)."""
    return itertools.chain.from_iterable(lst)


class TypeMap(dict):
    """Map types to handlers or values.

    Mapping a type to a value also (lazily) maps all of its subclasses to the
    same value, unless they have a mapping of their own.

    TypeMap should ideally not be updated after it is used, because updates may
    make some cached associations invalid.
    """

    def register(self, *obj_ts):
        """Decorator to register a handler to the given types."""
        def deco(handler):
            for obj_t in obj_ts:
                self[obj_t] = handler
            return handler
        return deco

    def __missing__(self, obj_t):
        """Get the handler for the given type."""
        handler = None
        to_set = []

        for cls in type.mro(obj_t):
            handler = super().get(cls, None)
            if handler is not None:
                for cls2 in to_set:
                    self[cls2] = handler
                break
            to_set.append(cls)

        if handler is not None:
            return handler
        else:
            raise KeyError(obj_t)


class Overload:
    """Overloaded function.

    A function can be added with the `register` method. One of its parameters
    should be annotated with a type, but only one, and every registered
    function should annotate the same parameter.

    Arguments:
        bind_to: Binds the first argument to the given object.
    """

    def __init__(self,
                 *,
                 bind_to=None,
                 wrapper=None,
                 mixins=[],
                 _parent=None):
        """Initialize an Overload."""
        if bind_to is True:
            bind_to = self
        elif bind_to is False:
            bind_to = None
        self.__self__ = bind_to
        self._parent = _parent
        self._wrapper = wrapper
        self.state = None
        if _parent:
            assert _parent.which is not None
            self.map = _parent.map
            self._uncached_map = _parent._uncached_map
            self.which = _parent.which
            self._wrapper = _parent._wrapper
            return
        _map = {}
        self.which = None
        for mixin in mixins:
            if self.which is None:
                self.which = mixin.which
            else:
                assert mixin.which == self.which
            _map.update(mixin._uncached_map)
        self.map = TypeMap(_map)
        self._uncached_map = _map

    def wrapper(self, wrapper):
        """Set a wrapper function."""
        if self._wrapper is not None:
            raise TypeError(f'wrapper for {self} is already set')
        self._wrapper = wrapper
        return self

    def register(self, fn):
        """Register a function."""
        if self._parent:  # pragma: no cover
            raise Exception('Cannot register a function on derived Overload')
        ann = fn.__annotations__
        if len(ann) != 1:
            raise Exception('Only one parameter may be annotated.')
        argnames = inspect.getfullargspec(fn).args
        for i, name in enumerate(argnames):
            t = ann.get(name, None)
            if t is not None:
                if isinstance(t, str):
                    t = eval(t)
                if self.which is None:
                    self.which = i
                elif self.which != i:
                    raise Exception(
                        'Annotation must always be on same parameter'
                    )
                break

        ts = t if isinstance(t, tuple) else (t,)

        for t in ts:
            self.map[t] = fn
            self._uncached_map[t] = fn

        return self

    def variant(self, fn=None):
        """Create a variant of this Overload.

        New functions can be registered to the variant without affecting the
        original.
        """
        fself = self.__self__
        bootstrap = True if fself is self else fself
        ov = Overload(
            bind_to=bootstrap,
            wrapper=self._wrapper,
            mixins=[self]
        )
        if fn is not None:
            ov.register(fn)
        return ov

    def __get__(self, obj, cls):
        return Overload(bind_to=obj, _parent=self)

    def __getitem__(self, t):
        if self.__self__:
            return self.map[t].__get__(self.__self__)
        else:
            return self.map[t]

    def __call__(self, *args, **kwargs):
        """Call the overloaded function."""
        fself = self.__self__
        if fself == 'stateful':
            ov = Overload(bind_to=True, _parent=self)
            return ov(*args)

        if fself is not None:
            args = (fself,) + args

        main = args[self.which]

        try:
            method = self.map[type(main)]
        except KeyError as err:
            method = None

        if self._wrapper is None:
            if method is None:
                raise TypeError(f'No overloaded method for {type(main)}')
            else:
                return method(*args, **kwargs)
        else:
            return self._wrapper(method, *args, **kwargs)


def _find_overload(fn, bootstrap):
    mod = __import__(fn.__module__, fromlist='_')
    dispatch = getattr(mod, fn.__name__, None)
    if dispatch is None:
        dispatch = Overload(bind_to=bootstrap)
    else:  # pragma: no cover
        assert bootstrap is False
    if not isinstance(dispatch, Overload):  # pragma: no cover
        raise TypeError('@overload requires Overload instance')
    return dispatch


def overload(fn=None, *, bootstrap=False):
    """Overload a function.

    Overloading is based on the function name.

    The decorated function should have one parameter annotated with a type.
    Any parameter can be annotated, but only one, and every overloading of a
    function should annotate the same parameter.

    The decorator optionally takes keyword arguments, *only* on the first
    use.

    Arguments:
        bootstrap: Whether to bootstrap this function so that it receives
            itself as its first argument. Useful for recursive functions.
    """
    if fn is None:
        def deco(fn):
            return overload(fn, bootstrap=bootstrap)
        return deco
    dispatch = _find_overload(fn, bootstrap)
    return dispatch.register(fn)


def overload_wrapper(wrapper=None, *, bootstrap=False):
    """Overload a function using the decorated function as a wrapper.

    The wrapper is the entry point for the function and receives as its
    first argument the method to dispatch to, and then the arguments to
    give to that method.
    """
    if wrapper is None:
        def deco(wrapper):
            return overload_wrapper(wrapper, bootstrap=bootstrap)
        return deco
    dispatch = _find_overload(wrapper, bootstrap)
    return dispatch.wrapper(wrapper)


overload.wrapper = overload_wrapper


def require_same(fns, objs):
    """Check that all objects have the same properties.

    Arguments:
        fns: A collection of functions. Each function must return the same
            result when applied to each object. For example, the functions
            may be `[type, len]`.
        objs: Objects that must be invariant with respect to the given
            functions.
    """
    o, *rest = objs
    for fn in fns:
        for obj in rest:
            if fn(o) != fn(obj):
                raise TypeError("Objects do not have the same properties:"
                                f" `{o}` and `{obj}` are not conformant.")


@overload(bootstrap=True)
def smap(self, arg: object, *rest):
    """Map a function recursively over all scalars in a structure."""
    raise NotImplementedError()


@overload  # noqa: F811
def smap(self, arg: (list, tuple), *rest):
    seqs = [arg, *rest]
    require_same([type, len], seqs)
    return type(arg)(self(*[s[i] for s in seqs])
                     for i in range(len(arg)))


@overload  # noqa: F811
def smap(self, arg: np.ndarray, *rest):
    return np.vectorize(self)(arg, *rest)


class Event:
    """Simple Event class.

    >>> e = Event('tickle')
    >>> e.register(lambda event, n: print("hi" * n))
    <function>
    >>> e(5)
    hihihihihi

    Attributes:
        name: The name of the event.
        owner: The object defining this event (defaults to None).
        history: A function that returns a list of previous events, or
            None if no history exists.

    """

    def __init__(self, name, owner=None, history=None):
        """Initialize an Event."""
        self._handlers = []
        self.name = name
        self.owner = owner
        self.history = history

    def register(self, handler, run_history=False):
        """Register a handler for this event.

        This returns the handler so that register can be used as a decorator.

        Arguments:
            handler: A function. The handler's first parameter will always
                be the event.
            run_history: Whether to call the handler for all previous events
                or not.
        """
        self._handlers.append(handler)
        if run_history and self.history:
            for entry in self.history():
                if isinstance(entry, tuple):
                    handler(self, *entry)
                else:
                    handler(self, entry)
        return handler

    def register_with_history(self, handler):
        """Register a handler for this event and run it on the history."""
        return self.register(handler, True)

    def remove(self, handler):
        """Remove this handler."""
        self._handlers.remove(handler)

    def __iter__(self):
        """Iterate over all handlers in the order they were added."""
        return iter(self._handlers)

    def __call__(self, *args, **kwargs):
        """Fire an event with the given arguments and keyword arguments."""
        for f in self:
            f(self, *args, **kwargs)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f'Event({self.name})'


class Events:
    """A group of named events.

    >>> events = Events(tickle=None, yawn=None)
    >>> events.tickle.register(lambda event, n: print("hi" * n))
    <function>
    >>> events.yawn.register(lambda event, n: print("a" * n))
    <function>
    >>> events.tickle(5)
    hihihihihi
    >>> events.yawn(20)
    aaaaaaaaaaaaaaaaaaaa

    Args:
        owner: The object that owns all the events. It will be passed
            as the first argument on each event.
        events: Keyword arguments mapping an event name to a history,
            or None if there is no history.
    """

    def __init__(self, owner=None, **events):
        """Initialize Events."""
        self.owner = owner
        for event_name, history in events.items():
            ev = Event(event_name, owner=owner, history=history)
            setattr(self, event_name, ev)


class NS:
    """Simple namespace, acts a bit like a dict with dot access on keys.

    This is different from Namespace below.

    This namespace preserves key order in the representation, unlike
    types.SimpleNamespace.
    """

    def __init__(self, **kwargs):
        """Initialize NS."""
        self.__dict__.update(kwargs)

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, item, value):
        self.__dict__[item] = value

    def __repr__(self):
        args = [f'{k}={v}' for k, v in self.__dict__.items()]
        return f'NS({", ".join(args)})'


class Namespace:
    """Namespace in which to resolve variables."""

    def __init__(self, label, *dicts):
        """Initialize a namespace.

        Args:
            label: The namespace's name.
            dicts: A list of dictionaries containing the namespace's data,
                which are consulted sequentially.
        """
        self.label = label
        self.dicts = dicts

    def __contains__(self, name):
        for d in self.dicts:
            if name in d:
                return True
        else:
            return False

    def __getitem__(self, name):
        for d in self.dicts:
            if name in d:
                return d[name]
        else:
            raise NameError(name)

    def __repr__(self):
        return f':{self.label}'  # pragma: no cover


class ModuleNamespace(Namespace):
    """Namespace that represents a Python module."""

    def __init__(self, name):
        """Initialize a ModuleNamespace.

        Args:
            name: Qualified name for the module. Must be possible
                to `__import__` it.
        """
        mod = vars(__import__(name, fromlist=['_']))
        super().__init__(name, mod, builtins_d)


class ClosureNamespace(Namespace):
    """Namespace that represents a function's closure."""

    def __init__(self, fn):
        """Initialize a ClosureNamespace.

        Args:
            fn: The function.

        """
        lbl = f'{fn.__module__}..<{fn.__name__}>'
        names = fn.__code__.co_freevars
        cells = fn.__closure__
        ns = dict(zip(names, cells or ()))
        super().__init__(lbl, ns)

    def __getitem__(self, name):
        d, = self.dicts
        try:
            return d[name].cell_contents
        except ValueError:
            raise UnboundLocalError(name)


stderr = AnsiToWin32(sys.stderr).stream


def eprint(*things):
    """Print to stderr."""
    print(*things, file=stderr)


def is_dataclass_type(cls):
    """Returns whether cls is a dataclass."""
    return isinstance(cls, type) and hasattr(cls, '__dataclass_fields__')


def as_frozen(x):
    """Return an immutable representation for x."""
    if isinstance(x, dict):
        return tuple(sorted((k, as_frozen(v)) for k, v in x.items()))
    elif isinstance(x, (list, tuple)):
        return tuple(as_frozen(y) for y in x)
    else:
        return x


class ErrorPool:
    """Accumulates a list of errors.

    Arguments:
        exc_class: The exception to raise if any error occurred.
    """

    def __init__(self, *, exc_class=Exception):
        """Initialize the ErrorPool."""
        self.errors = []
        self.exc_class = exc_class

    def add(self, error):
        """Add an exception to the pool."""
        assert isinstance(error, Exception)
        self.errors.append(error)

    def trigger(self, stringify=lambda err: f'* {err.args[0]}'):
        """Raise an exception if an error occurred."""
        if self.errors:
            msg = "\n".join(stringify(e) for e in self.errors)
            exc = self.exc_class(msg)
            exc.errors = self.errors
            raise exc


@dataclass(frozen=True)
class SymbolicKeyInstance:
    """Stores information that corresponds to a node in the graph."""

    node: "ANFNode"
    abstract: object


@smap.variant
def _add(self, x: object, y):
    return x + y


class EnvInstance:
    """Environment mapping keys to values.

    Keys are SymbolicKeyInstances, which represent nodes in the graph along
    with inferred properties.
    """

    def __init__(self, _contents={}):
        """Initialize a EnvType."""
        self._contents = dict(_contents)

    def get(self, key, default):
        """Get the sensitivity list for the given key."""
        return self._contents.get(key, default)

    def set(self, key, value):
        """Set a value for the given key."""
        rval = EnvInstance(self._contents)
        rval._contents[key] = value
        return rval

    def add(self, other):
        """Add two EnvInstances."""
        rval = EnvInstance(self._contents)
        for k, v in other._contents.items():
            v0 = rval._contents.get(k)
            if v0 is not None:
                rval._contents[k] = _add(v0, v)
            else:
                rval._contents[k] = v
        return rval

    def __len__(self):
        return len(self._contents)


newenv = EnvInstance()

"""Debug/Testing Virtual Machine.

This VM will directly execute a graph so it should be suitable for
testing or debugging.  Don't expect stellar performance from this
implementation.
"""

from collections import defaultdict
from typing import Iterable, Mapping, Any, List

from .ir import Graph, Apply, Parameter, ANFNode
from .ir.utils import is_constant_graph, is_constant, is_apply, is_parameter
from .prim import Primitive
from .prim.ops import if_, return_, partial
from .utils import TypeMap


def is_partial(node):
    """Returns True if node is an application of partial."""
    return (is_apply(node) and
            is_constant(node.inputs[0]) and
            node.inputs[0].value == partial)


class VMFrame:
    """An execution frame.

    This holds the state for an application of a graph.  The todo list
    must contain free variables of graphs encountered before the
    graph themselves.

    You can index a frame with a node to get its value in the context
    of this frame (if it has already been evaluated).

    Attributes:
        values: Mapping of node to their values in this application
        todo: list of nodes remaining to execute
        closure: values for the closure if the current application is a closure

    """

    def __init__(self, nodes: Iterable[ANFNode], values: Mapping[ANFNode, Any],
                 *, closure: Mapping[ANFNode, Any] = None) -> None:
        """Initialize a frame."""
        self.values = dict(values)
        self.todo = list(nodes)
        self.todo.reverse()
        self.closure = closure

    def __getitem__(self, node: ANFNode):
        if is_constant_graph(node):
            node = node.value
        if node in self.values:
            return self.values[node]
        elif self.closure is not None and node in self.closure:
            return self.closure[node]
        elif is_constant(node):
            # Should be a constant
            return node.value
        if isinstance(node, Graph):
            return node
        else:
            raise ValueError(node)  # pragma: no cover


class Closure:
    """Representation of a closure."""

    def __init__(self, graph: Graph, values: Mapping[ANFNode, Any]) -> None:
        """Build a closure."""
        self.graph = graph
        self.values = values
        self.vm: 'VM' = None

    def __call__(self, *args):
        """Evaluates the closure."""
        return self.vm.evaluate(self.graph, args, closure=self.values)


class Partial:
    """Representation of a partial application."""

    def __init__(self, fn, args, vm):
        """Build a partial."""
        self.fn = fn
        if args is not None:
            self.args = tuple(args)
        else:
            self.args = None
        self.vm = vm

    def __call__(self, *args):
        """Evaluates the partial."""
        return self.vm.call(self.fn, self.args + args)


class Prealloc:
    """Marker to preallocate a recursive structure."""

    def __init__(self, node):
        """Create a Prealloc for `node`."""
        self.node = node


class VM:
    """Virtual Machine interface."""

    class _Call(Exception):
        """Indicate a call to a new frame."""

        def __init__(self, frame):
            self.frame = frame

    class _Return(Exception):
        """Indicates a return with its value."""

        def __init__(self, value):
            self.value = value

    def __init__(self, convert, manager, py_implementations, implementations):
        """Initialize the VM."""
        self.convert = convert
        self.manager = manager
        self._exporters = TypeMap({
            tuple: self._export_sequence,
            list: self._export_sequence,
            Closure: self._export_Closure,
            Graph: self._export_Graph,
            Primitive: self._export_Primitive,
            object: self._export_object,
        })
        self.implementations = implementations
        self.py_implementations = py_implementations
        self._vars = defaultdict(set)

    def _compute_fvs(self, graph):
        rval = set()
        for fv in graph.free_variables_total:
            if isinstance(fv, Graph):
                rval.add(fv)
            else:
                rval.add(fv)
        return rval

    def _acquire_graph(self, graph):
        if graph in self._vars:
            return
        self.manager.add_graph(graph)
        for g in graph.manager.graphs:
            self._vars[g] = self._compute_fvs(g)

    def _export_sequence(self, seq):
        return type(seq)(self.export(x) for x in seq)

    def _export_Primitive(self, prim):
        return self.py_implementations[prim]

    def _export_Closure(self, clos):
        clos.vm = self
        return clos

    def _export_Graph(self, g):
        """Return an object that executes `g` when called on arguments."""
        c = Closure(g, None)
        c.vm = self
        return c

    def _export_object(self, obj):
        return obj

    def export(self, value):
        """Convert a value from the VM into a corresponding Python object."""
        return self._exporters[type(value)](value)

    def evaluate(self, graph: Graph, _args: Iterable[Any], *,
                 closure: Mapping[ANFNode, Any] = None) -> Any:
        """Run a graph.

        This will evaluate the passed-in graph and return the
        resulting value.
        """
        args = self.convert(tuple(_args))

        self._acquire_graph(graph)

        if len(args) != len(graph.parameters):
            raise RuntimeError("Call with wrong number of arguments")

        top_frame = VMFrame(self._toposort(graph.return_),
                            dict(zip(graph.parameters, args)),
                            closure=closure)
        frames = [top_frame]

        while frames:
            try:
                frame = frames[-1]
                todo = frame.todo
                while todo:
                    self._handle_node(todo[-1], frame)
                    todo.pop()
            except self._Call as c:
                # The last element of todo is always a return
                if len(todo) == 2:
                    frames[-1] = c.frame
                else:
                    frames.append(c.frame)
            except self._Return as r:
                frames.pop()
                if frames:
                    frames[-1].values[frames[-1].todo[-1]] = r.value
                    frames[-1].todo.pop()
                else:
                    return self.export(r.value)

    def _toposort(self, root: ANFNode, closure=None) -> List[ANFNode]:
        done = set()
        prepend = set()
        todo = [root]
        rank = {}
        res = []
        pres = []

        if closure is not None:
            done.update(closure.keys())

        while todo:
            node = todo[-1]
            if node in done:
                todo.pop()
                continue
            if node in rank and rank[node] != len(todo):
                pos = len(todo) - 1
                n = node
                while not self._is_break_point(n) and pos >= rank[node]:
                    pos -= 1
                    n = todo[pos]
                if pos < rank[node]:
                    raise ValueError('cycle')
                prepend.add(n)
                pres.append(Prealloc(n))
                del todo[pos:]
                continue
            rank[node] = len(todo)
            cont = False
            for i in self._succ_vm(node):
                if i not in done and i not in prepend:
                    todo.append(i)
                    cont = True
            if cont:
                continue
            done.add(node)
            res.append(node)
            todo.pop()

        res = pres + res
        return res

    def _is_break_point(self, node):
        """Return if a cycle can be broken at this node."""
        return (is_partial(node) or
                # It's a closure
                (isinstance(node, Graph) and
                 len(self._vars[node]) != 0))

    def _succ_vm(self, node: ANFNode) -> Iterable[ANFNode]:
        """Follow node.incoming and free variables."""
        if isinstance(node, Graph):
            yield from self._vars[node]
        else:
            for i in node.inputs:
                if i.graph == node.graph and not is_parameter(i):
                    yield i
                if is_constant_graph(i):
                    yield i.value

    def call(self, fn, args):
        """Call the `fn` object.

        `fn` can be anything that would be valid as the first element
        of an apply.
        """
        if isinstance(fn, Primitive):
            return self.implementations[fn](self, *args)

        elif isinstance(fn, Graph):
            return self.evaluate(fn, args)

        elif isinstance(fn, Closure):
            return self.evaluate(fn.graph, args, closure=fn.values)

        else:
            raise AssertionError(f"Can't call {fn}")

    def _call(self, graph: Graph, args: List[Any]):
        clos = None
        if isinstance(graph, Closure):
            clos = graph.values
            graph = graph.graph

        assert isinstance(graph, Graph)

        if len(args) != len(graph.parameters):
            raise RuntimeError("Call with wrong number of arguments")

        raise self._Call(VMFrame(self._toposort(graph.return_, closure=clos),
                                 dict(zip(graph.parameters, args)),
                                 closure=clos))

    def _make_closure(self, graph: Graph, frame: VMFrame, closure=None):
        clos = dict()
        for v in self._vars[graph]:
            clos[v] = frame[v]
        if closure is None:
            closure = Closure(graph, None)
        assert closure.values is None
        closure.values = clos
        return closure

    def _dispatch_call(self, node, frame, fn, args):
        if isinstance(fn, Primitive):
            if fn == if_:
                cond, tb, fb = args
                fn = tb if cond else fb
                self._dispatch_call(node, frame, fn, [])
            elif fn == return_:
                raise self._Return(args[0])
            elif fn == partial:
                partial_fn, *partial_args = args
                res = frame.values.get(node, Partial(None, None, self))
                assert res.fn is None
                res.fn = partial_fn
                assert res.args is None
                res.args = tuple(partial_args)
                frame.values[node] = res
            else:
                frame.values[node] = self.implementations[fn](self, *args)
        elif isinstance(fn, Partial):
            self._dispatch_call(node, frame, fn.fn, fn.args + tuple(args))
        elif isinstance(fn, (Graph, Closure)):
            self._call(fn, args)
        else:
            raise AssertionError(f'Invalid fn to call: {fn}')

    def _handle_node(self, node: ANFNode, frame: VMFrame):
        if isinstance(node, Graph):
            if frame.closure is not None and node in frame.closure:
                return

            # We only visit constant graphs
            if len(self._vars[node]) != 0:
                frame.values[node] = self._make_closure(
                    node, frame, closure=frame.values.get(node, None))
            # We don't need to do anything special for non-closures

        elif isinstance(node, Parameter):
            pass

        elif isinstance(node, Apply):
            fn, *args = (frame[inp] for inp in node.inputs)
            self._dispatch_call(node, frame, fn, args)

        elif isinstance(node, Prealloc):
            if frame.closure is not None and node.node in frame.closure:
                return
            if isinstance(node.node, Graph):
                frame.values[node.node] = Closure(node.node, None)
            elif is_partial(node.node):
                frame.values[node.node] = Partial(None, None, self)
            else:
                raise RuntimeError("Unsupported Prealloc type")

        else:
            raise AssertionError("Unknown node type")  # pragma: no cover

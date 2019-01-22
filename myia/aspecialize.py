"""Specialize graphs according to the types of their arguments."""

import numpy
from itertools import count

from .abstract import GraphXInferrer
from .abstract.base import GraphAndContext, concretize_abstract, \
    AbstractTuple, AbstractList, AbstractArray, AbstractScalar, \
    AbstractFunction, PartialApplication, TYPE, VALUE, SHAPE, REF, \
    AbstractError, AbstractValue
from .dtype import Function, TypeMeta
from .infer import ANYTHING, Context, concretize_type, \
    GraphInferrer, PartialInferrer, Inferrer, Unspecializable, \
    DEAD, INACCESSIBLE, POLY
from .ir import GraphCloner, Constant, Graph, MetaGraph
from .prim import ops as P, Primitive
from .utils import Overload, overload, Namespace, SymbolicKeyInstance, \
    EnvInstance


_count = count(1)


def _const(v, t):
    ct = Constant(v)
    ct.type = t
    return ct


def _const2(v, t):
    ct = Constant(v)
    ct.abstract = t
    return ct


async def concretize_cache(cache):
    for (track, k), v in list(cache.items()):
        kc = await concretize_abstract(k)
        cache[(track, kc)] = v


async def concretize_cache2(cache):
    for k, v in list(cache.items()):
        kc = await concretize_abstract(k)
        cache[kc] = v


class TypeSpecializer:
    """Specialize a graph using inferred type information."""

    def __init__(self, engine):
        """Initialize a TypeSpecializer."""
        self.engine = engine
        self.seen = set()
        assert len(self.engine.tracks) == 1
        self.track = self.engine.tracks['abstract']
        self.mng = self.engine.mng
        self.specializations = {Context.empty(): None}

    def run(self, graph, context):
        """Run the specializer on the given graph in the given context."""
        ginf = self.track.get_inferrer_for(
            GraphAndContext(graph, context),
            None
        )

        argrefs = [self.engine.ref(p, context)
                   for p in graph.parameters]

        return self.engine.run_coroutine(
            self._specialize_helper(ginf, argrefs)
        )

    async def _specialize_helper(self, ginf, argrefs):
        await concretize_cache(self.engine.cache.cache)
        await concretize_cache2(self.engine.reference_map)
        g = ginf._graph
        argvals = [await self.engine.get_inferred('abstract', ref)
                   for ref in argrefs]
        ctx = ginf.make_context(self.track, argvals)
        return await self._specialize(g, ctx, argrefs)

    async def _specialize(self, g, ctx, argrefs):
        ctxkey = ctx  # TODO: Reify ctx to collapse multiple ctx into one
        if ctxkey in self.specializations:
            return self.specializations[ctxkey].new_graph

        gspec = _GraphSpecializer(self, g, await concretize_abstract(ctx))
        g2 = gspec.new_graph
        self.specializations[ctxkey] = gspec
        await gspec.run()
        return g2


_legal = (int, float, numpy.number, numpy.ndarray,
          str, Namespace, SymbolicKeyInstance, TypeMeta)


def _visible(g, node):
    if isinstance(node, Graph):
        g2 = node.parent
    else:
        g2 = node.graph
    while g and g2 is not g:
        g = g.parent
    return g2 is g


@overload
def _is_concrete_value(v: (tuple, list)):
    return all(_is_concrete_value(x) for x in v)


@overload  # noqa: F811
def _is_concrete_value(v: EnvInstance):
    return all(_is_concrete_value(x) for x in v._contents.values())


@overload  # noqa: F811
def _is_concrete_value(v: _legal):
    return True


@overload  # noqa: F811
def _is_concrete_value(v: object):
    return False


class _GraphSpecializer:
    """Helper class for TypeSpecializer."""

    def __init__(self, specializer, graph, context):
        # buche(context)
        self.parent = specializer.specializations[context.parent]
        self.specializer = specializer
        self.engine = specializer.engine
        self.graph = graph
        self.context = context
        self.cl = GraphCloner(
            self.graph,
            total=False,
            clone_children=False,
            graph_relation=next(_count)
        )
        self.cl.run()
        self.repl = self.cl.repl
        self.new_graph = self.repl[self.graph]
        self.todo = [self.graph.return_] + list(self.graph.parameters)
        self.marked = set()

    def get(self, node):
        g = node.graph
        sp = self
        while g is not None and g is not sp.graph:
            sp = sp.parent
        return sp.repl.get(node, node)

    async def run(self):
        await self.first_pass()
        await self.second_pass()
        # while self.todo:
        #     node = self.todo.pop()
        #     if node.graph is None:
        #         continue
        #     if node.graph is not self.graph:
        #         self.parent.todo.append(node)
        #         await self.parent.run()
        #         continue
        #     if node in self.marked:
        #         continue
        #     self.marked.add(node)
        #     await self.process_node(node)

    def ref(self, node):
        return self.engine.ref(node, self.context)

    # #########
    # # Build #
    # #########

    # build = Overload()

    # @build.register
    # async def build(self, a: AbstractFunction, argvals=None):
    #     fns = a.values[VALUE]
    #     if len(fns) != 1:
    #         raise Unspecializable('xx1')
    #     fn, = fns
    #     return await self.build_inferrer(a, fn, argvals)

    # @build.register
    # async def build(self, a: AbstractScalar, argvals=None):
    #     v = a.values[VALUE]
    #     if v is ANYTHING:
    #         raise Unspecializable('xx0')
    #     else:
    #         return _const2(v, a)

    # @build.register
    # async def build(self, a: AbstractTuple, argvals=None):
    #     raise Unspecializable('xx3')

    # @build.register
    # async def build(self, a: object, argvals=None):
    #     raise Unspecializable('xx4')

    build_inferrer = Overload()

    @build_inferrer.register
    async def build_inferrer(self, a, fn: Primitive, argvals):
        return _const2(fn, a)

    @build_inferrer.register
    async def build_inferrer(self, a, fn: Graph, argvals):
        return await self.build_inferrer(
            a, GraphAndContext(fn, Context.empty()), argvals
        )

    @build_inferrer.register
    async def build_inferrer(self, a, fn: GraphAndContext, argvals):
        if not _visible(self.graph, fn.graph):
            raise Unspecializable('xx5')
        inf = self.specializer.track.get_inferrer_for(fn, None)
        return await self._build_inferrer_x(a, inf, argvals)

    # @build_inferrer.register
    # async def build_inferrer(self, a, fn: PartialApplication, argvals):
    #     all_argvals = None if argvals is None else [*fn.args, *argvals]
    #     suba = AbstractFunction(fn.fn)
    #     sub_build = await self.build_inferrer(suba, fn.fn, all_argvals)
    #     ptl_args = [await self.build2(val) for val in fn.args]
    #     ptl = _const2(P.partial, AbstractFunction(P.partial))
    #     res = self.new_graph.apply(
    #         ptl,
    #         sub_build,
    #         *ptl_args
    #     )
    #     res.abstract = a
    #     return res

    @build_inferrer.register
    async def build_inferrer(self, a, fn: MetaGraph, argvals):
        assert argvals is not None
        inf = self.specializer.track.get_inferrer_for(fn, argvals)
        return await self._build_inferrer_x(a, inf, argvals)

    @build_inferrer.register
    async def build_inferrer(self, a, fn: object, argvals):
        raise Exception(f'Unrecognized: {fn}')

    async def _find_unique_argvals(self, inf, argvals):
        if argvals is None:
            argvals = ()

        argvals = tuple(argvals)

        if argvals in inf.cache:
            return argvals

        n = len(argvals)
        choices = set()
        await concretize_cache2(inf.cache)
        for k, v in inf.cache.items():
            k = tuple([await concretize_abstract(x) for x in k])
            if k[:n] == argvals:
                choices.add(k)

        if len(choices) == 1:
            for choice in choices:
                # Return the only element
                return choice
        elif len(choices) == 0:
            raise Unspecializable(DEAD)
        else:
            raise Unspecializable(POLY)

        # if argvals is None:
        #     if len(inf.cache) == 1:
        #         argvals, = inf.cache.keys()
        #         return argvals
        #     elif len(inf.cache) == 0:
        #         # print(a, inf.cache)
        #         raise Unspecializable('xx6')
        #     else:
        #         raise Unspecializable('xx7')
        # else:
        #     return argvals

    async def _build_inferrer_x(self, a, inf, argvals):
        argvals = await self._find_unique_argvals(inf, argvals)
        ctx = inf.make_context(self.specializer.track, argvals)
        v = await self.specializer._specialize(inf._graph, ctx, None)
        return _const2(v, a)

    ###########
    # Process #
    ###########

    async def process_node(self, node):
        ref = self.ref(node)
        new_node = self.get(node)
        if new_node.graph is not self.new_graph:
            raise AssertionError('Error in specializer [A]')

        new_node.abstract = await concretize_abstract(await ref['abstract'])

        if node.is_apply():
            new_inputs = new_node.inputs
            irefs = list(map(self.ref, node.inputs))
            ivals = [await concretize_abstract(await iref['abstract'])
                     for iref in irefs]
            for i, ival in enumerate(ivals):
                iref = irefs[i]
                argvals = ivals[1:] if i == 0 else None
                while iref in self.specializer.engine.reference_map:
                    iref = self.specializer.engine.reference_map[iref]
                    self.cl.clone_disconnected(iref.node)
                try:
                    repl = await self.build(ival, argvals)
                except Unspecializable as e:
                    if new_inputs[i].is_constant_graph():
                        # Graphs that cannot be specialized are replaced
                        # by a constant with the associated Problem type.
                        # We can't keep references to unspecialized graphs.
                        repl = _const2(e.problem.kind,
                                       AbstractError(e.problem))
                    else:
                        self.todo.append(iref.node)
                        repl = self.get(iref.node)
                        repl.abstract = await concretize_abstract(ival)
                if repl is not new_inputs[i]:
                    # await self.fill_inferred(repl, iref)
                    new_inputs[i] = repl
                else:
                    self.todo.append(node.inputs[i])

    #############
    # New stuff #
    #############

    async def first_pass(self):
        while self.todo:
            node = self.todo.pop()
            if node.graph is None:
                continue
            if node.graph is not self.graph:
                self.parent.todo.append(node)
                await self.parent.run()
                continue
            if node in self.marked:
                continue
            self.marked.add(node)
            await self.process_node2(node)

    async def second_pass(self):
        # Specialize constant graphs
        from .graph_utils import dfs
        from .ir.utils import succ_incoming
        for node in dfs(self.new_graph.return_, succ=succ_incoming):
            if node.is_apply():
                await self.process_apply(node)

    build2 = Overload()

    @build2.register
    async def build2(self, ref, a: AbstractFunction):
        fns = a.values[VALUE]
        if len(fns) == 1:
            fn, = fns
            if isinstance(fn, (Primitive, Graph, MetaGraph)):
                g = fn
            elif isinstance(fn, GraphAndContext):
                g = fn.graph
            else:
                return None
            if not isinstance(g, Graph) \
                    or g.parent is None \
                    or (ref.node.is_constant_graph()
                        and _visible(self.graph, g)):
                return _const2(g, a)
            else:
                return None
        else:
            return None

    @build2.register
    async def build2(self, ref, a: AbstractScalar):
        v = a.values[VALUE]
        if v is ANYTHING:
            return None
        else:
            return _const2(v, a)

    @build2.register
    async def build2(self, ref, a: AbstractValue):
        # Default case
        return None

    async def process_node2(self, node):
        ref = self.ref(node)
        new_node = self.get(node)
        if new_node.graph is not self.new_graph:
            raise AssertionError('Error in specializer [A]')

        new_node.abstract = await concretize_abstract(await ref['abstract'])

        if node.is_apply():
            new_inputs = new_node.inputs
            irefs = list(map(self.ref, node.inputs))
            ivals = [await concretize_abstract(await iref['abstract'])
                     for iref in irefs]
            for i, ival in enumerate(ivals):
                iref = irefs[i]
                repl = await self.build2(iref, ival)
                if repl is None:
                    while iref in self.specializer.engine.reference_map:
                        iref = self.specializer.engine.reference_map[iref]
                        self.cl.clone_disconnected(iref.node)
                    self.todo.append(iref.node)
                    repl = self.get(iref.node)
                    repl.abstract = await concretize_abstract(ival)
                if repl is not new_inputs[i]:
                    new_inputs[i] = repl

    async def process_apply(self, new_node):
        if new_node in self.specializer.seen:
            return
        self.specializer.seen.add(new_node)

        async def _helper(i, node, a, argvals):
            if node.is_constant_graph() or node.is_constant(MetaGraph):
                fns = a.values[VALUE]
                try:
                    if len(fns) != 1:
                        raise Unspecializable('xxx')
                    fn, = fns
                    repl = await self.build_inferrer(a, fn, argvals)
                except Unspecializable as e:
                    # repl = _const2(e.problem.kind, AbstractError(e.problem.kind))
                    repl = _const2(e.problem, AbstractError(e.problem))
                new_inputs[i] = repl

        new_inputs = new_node.inputs
        fn, *args = new_inputs
        while fn.is_apply(P.partial):
            args = fn.inputs[2:] + args
            fn = fn.inputs[1]
            new_inputs = [fn, *args]
        fnval, *argvals = [x.abstract for x in new_inputs]
        await _helper(0, fn, fnval, argvals)
        for i, val in enumerate(argvals):
            await _helper(i + 1, args[i], val, None)

        new_node.inputs = new_inputs

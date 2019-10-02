"""Implementation of the 'user_switch' operation."""

from functools import reduce
from itertools import product

from .. import lib
from ..lib import (
    ANYTHING,
    CloneRemapper,
    GraphCloner,
    MyiaTypeError,
    force_pending,
    macro,
    union_simplify,
)
from ..xtype import Bool
from . import primitives as P


class _CastRemapper(CloneRemapper):

    def __init__(self,
                 graphs,
                 inlines,
                 manager,
                 relation,
                 graph_relation,
                 clone_constants,
                 graph_repl,
                 fv_replacements):
        """Initialize the GraphCloner."""
        super().__init__(
            graphs=graphs,
            inlines=inlines,
            manager=manager,
            relation=relation,
            graph_repl=graph_repl,
            graph_relation=graph_relation,
            clone_constants=clone_constants,
        )
        self.fv_replacements = fv_replacements

    def gen_fv(self, g, ng, fv):
        """Remap the free variables we want to remap."""
        if fv in self.fv_replacements:
            new = self.fv_replacements[fv]
            self.remap_node((g, fv), g, fv, ng, new, link=False)


async def make_trials(engine, ref, repl):

    def getrepl(node, opt):
        key = (node, opt)
        if key not in repl:
            repl[key] = g.apply(P.unsafe_static_cast, node, opt)
        return repl[key]

    node = ref.node
    g = node.graph
    typ = await ref.get()

    if isinstance(typ, lib.AbstractUnion):
        return {
            frozenset({(node, opt)}): getrepl(node, opt)
            for opt in typ.options
        }

    elif ref.node.is_apply():
        arg_results = [
            (await make_trials(
                engine, engine.ref(arg, ref.context), repl
            )).items()
            for arg in ref.node.inputs
        ]
        res = {}
        for entry in product(*arg_results):
            s = set()
            for s2, n in entry:
                s |= s2
            nodes = [n for _, n in entry]
            if nodes == ref.node.inputs:
                new_node = node
            else:
                new_node = g.apply(*nodes)
            res[frozenset(s)] = new_node
        return res

    elif ref.node.is_constant_graph():
        g = ref.node.value
        if g.parent is None:
            return {frozenset(): ref.node}
        else:
            # TODO
            return {frozenset(): ref.node}

    else:
        return {frozenset(): ref.node}


@macro
async def user_switch(info, condref, tbref, fbref):
    """Implement the switch functionality generated by the parser.

    If user_switch finds a Union in the condition, it will infer the value of
    the condition for each type in the union. If the condition is necessarily
    true or false for some types, the type of the variable for the
    corresponding conditional branch will be set to these types.
    """
    engine = info.engine
    g = info.graph

    async def type_trials(cond_trials, opnode, argrefs):
        """Handle `user_switch(hastype(x, typ), tb, fb)`.

        We want to evaluate tb in a context where x has type typ and fb
        in a context where it doesn't.
        """
        async def wrap(branch_ref, branch_types):
            # We transform branch_graph into a new graph which refers to a cast
            # version of x. We also transform all of the children of x's graph
            # so that closures called in the branch also refer to the cast
            # version of x.
            branch_graph = branch_ref.node.value
            nomod = True

            rval = branch_graph.make_new(relation='copy')
            children = set()
            fv_repl = {}
            for node, typ in branch_types.items():
                if branch_graph not in node.graph.scope:
                    continue
                nomod = False
                children.update(node.graph.children)
                cast = rval.apply(P.unsafe_static_cast, node, typ)
                fv_repl[node] = cast

            if nomod:
                return branch_graph

            cl = GraphCloner(
                *children,
                total=False,
                graph_repl={branch_graph: rval},
                remapper_class=_CastRemapper.partial(
                    fv_replacements=fv_repl
                )
            )
            assert rval is cl[branch_graph]
            engine.mng.add_graph(rval)
            return rval

        from collections import defaultdict
        groups = {
            True: defaultdict(list),
            False: defaultdict(list),
            ANYTHING: defaultdict(list),
        }

        for keys, cond in cond_trials.items():
            result = await engine.ref(cond, ctx).get()
            assert result.xtype() is Bool
            value = result.xvalue()
            for node, opt in keys:
                groups[value][node].append(opt)

        if groups[ANYTHING]:
            return await default()

        typemap = {}
        for key, mapping in groups.items():
            typemap[key] = {node: union_simplify(opts)
                            for node, opts in mapping.items()}

        if not groups[True]:
            return fbref
        elif not groups[False]:
            return tbref
        else:
            new_conds = []
            for node, types in groups[True].items():
                parts = [g.apply(P.hastype, node, t)
                         for t in types]
                new_cond = reduce(lambda x, y: g.apply(P.bool_or, x, y),
                                  parts)
                new_conds.append(new_cond)
            new_cond = reduce(lambda x, y: g.apply(P.bool_and, x, y),
                              new_conds)
            new_tb = await wrap(tbref, typemap[True])
            new_fb = await wrap(fbref, typemap[False])
            return g.apply(P.switch, new_cond, new_tb, new_fb)

    async def default():
        _, _, tb, fb = info.outref.node.inputs
        return g.apply(P.switch, cond, tb, fb)

    for branch_ref in [tbref, fbref]:
        if not branch_ref.node.is_constant_graph():
            raise MyiaTypeError(
                'Both branches of user_switch must be constant graphs.'
            )

    orig_cond = cond = condref.node
    ctx = condref.context

    condt = await condref.get()
    if not engine.check_predicate(Bool, condt):
        to_bool = engine.resources.convert(bool)
        cond = (cond.graph or g).apply(to_bool, cond)

    if orig_cond.graph is not None and cond.is_apply():
        opnode, *args = cond.inputs
        opref = engine.ref(opnode, ctx)
        ops = (await opref.get()).get_sync()
        if len(ops) == 1:
            op, = ops
            argrefs = [engine.ref(a, ctx) for a in args]

            new_condref = engine.ref(cond, ctx)
            cond_alts = await make_trials(engine, new_condref, {})
            if len(cond_alts) > 1:
                return await type_trials(cond_alts, opnode, argrefs)

    return await default()


__operation_defaults__ = {
    'name': 'user_switch',
    'registered_name': 'user_switch',
    'mapping': user_switch,
    'python_implementation': None,
}

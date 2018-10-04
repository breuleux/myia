"""Generate the gradient graphs"""


from collections import defaultdict
from functools import reduce
from typing import Set, Dict, Tuple, List

from .composite import zeros_like, hyper_add
from .info import About
from .ir import Apply, Constant, Graph, ANFNode, manage, clone
from .opt import sexp_to_node
from .prim import ops as primops, Primitive
from .prim.grad_implementations import augmented_graphs
from .utils import Partializable, overload


class GraphRemapper(Partializable):

    def __init__(self,
                 relation,
                 graphs,
                 remappers=None,
                 master=None,
                 in_remapper=None):
        self.relation = relation
        self.graphs = graphs
        self.graph_repl = {}
        self.repl = {}
        self.remappers = remappers
        self.in_remapper = in_remapper or self
        self.master = master or self
        self.to_link = []
        self._populated = False

    def prepare(self):
        if isinstance(self.in_remapper, str):
            self.in_remapper = self.remappers[self.in_remapper]
        if isinstance(self.master, str):
            self.master = self.remappers[self.master]

    def get(self, g, node):
        if (g, node) in self.repl:
            return self.repl[(g, node)]
        elif node in self.repl:
            return self.repl[node]
        elif node.is_constant_graph():
            return self.repl[node.value]
        else:
            raise KeyError(f'Unprocessed node: {node}')

    def get_graph(self, g):
        return self.graph_repl[g]

    def add_node(self, key, g, node, ng, new_node, link=None):
        if link is None:
            link = new_node.is_apply()
        self.repl[key] = new_node
        if link:
            self.to_link.append((key, g, node, ng, new_node))

    def gen_graph(self, g):
        if self.master is not self:
            ng = self.master.get_graph(g)
        else:
            with About(g.debug, self.relation):
                ng = Graph()
        self.graph_repl[g] = ng
        return ng

    def gen_parameter(self, g, ng, p):
        if self.master is not self:
            return
        with About(p.debug, self.relation):
            # self.repl[p] = ng.add_parameter()
            self.add_node(p, g, p, ng, ng.add_parameter())

    def gen_apply(self, g, ng, node):
        with About(node.debug, self.relation):
            # self.repl[node] = ng.apply()
            self.add_node(node, g, node, ng, ng.apply())

    def gen_child(self, g, ng, child):
        pass

    def gen_fv(self, g, ng, node):
        pass

    def gen_fv_graph(self, g, ng, g2):
        pass

    def gen_constant(self, g, ng, ct):
        if self.master is not self:
            return
        with About(ct.debug, self.relation):
            self.add_node(ct, g, ct, ng, ct)

    def gen_constant_graph(self, g, ng, ct):
        return self.gen_constant(g, ng, ct)

    def link_apply(self, g, ng, node, new_node):
        new_inputs = [self.in_remapper.get(g, inp)
                      for inp in node.inputs]
        new_node.inputs = new_inputs

    def finalize_graph(self, g, ng):
        ng.output = self.get(g, g.output)

    def populate(self):
        if self._populated:
            return

        for g in self.graphs:
            ng = self.gen_graph(g)

        for g in self.graphs:
            ng = self.get_graph(g)
            for p in g.parameters:
                self.gen_parameter(g, ng, p)

            for node in g.nodes:
                if node.is_apply():
                    self.gen_apply(g, ng, node)

            for child in g.children:
                self.gen_child(g, ng, child)

            for node in g.free_variables_indirect:
                self.gen_fv(g, ng, node)

            for graph in g.free_variables_graphs:
                self.gen_fv_graph(g, ng, graph)

            for ct in g.constants:
                if ct.is_constant_graph():
                    self.gen_constant_graph(g, ng, ct)
                else:
                    self.gen_constant(g, ng, ct)

        self._populated = True

    def link(self):
        for _, g, node, ng, new_node in self.to_link:
            self.link_apply(g, ng, node, new_node)

    def finalize(self):
        if self.master is self:
            for g in self.graphs:
                self.finalize_graph(g, self.get_graph(g))


class FPropAppRemapper(GraphRemapper):
    pass


class FPropRemapper(GraphRemapper):
    def gen_constant(self, g, ng, ct):
        self.repl[(g, ct)] = sexp_to_node((primops.J, ct), ng)

    def gen_constant_graph(self, g, ng, ct):
        if ct.value in self.graphs:
            new_ct = Constant(self.get_graph(ct.value))
            self.repl[ct] = new_ct
            self.repl[ct.value] = new_ct
        else:
            self.gen_constant(g, ng, ct)

    def gen_fv(self, g, ng, fv):
        if fv.graph not in self.graphs:
            return self.gen_constant(g, ng, fv)

    def gen_fv_graph(self, g, ng, fvg):
        if fvg in self.graphs:
            return self.gen_constant_graph(g, ng, Constant(fvg))
        else:
            return self.gen_constant(g, ng, fvg)

    def link_apply(self, g, ng, node, new_node):
        assert not node.is_parameter()
        app = self.remappers['grad_fprop_app'].get(g, node)
        new_node.inputs = sexp_to_node(
            (primops.tuple_getitem, app, 0),
            ng
        ).inputs

    def finalize_graph(self, g, ng):
        g.transforms['grad'] = ng
        ng.transforms['primal'] = g
        elems = self.get(g, g.output), self.remappers['grad_sens'].get_graph(g)
        ng.output = ng.apply(primops.make_tuple, *elems)


class BPropRemapper(GraphRemapper):
    def link_apply(self, g, ng, node, new_node):
        app = self.remappers['grad_fprop_app'].get(g, node)
        new_node.inputs = sexp_to_node(
            (primops.tuple_getitem, app, 1),
            ng
        ).inputs


class BPropAppRemapper(GraphRemapper):
    def link_apply(self, g, ng, node, new_node):
        if node.is_parameter():
            return
        fn = self.remappers['grad_bprop'].get(g, node)
        arg = self.remappers['grad_sens'].get(g, node)
        new_node.inputs = [fn, arg]


class SensRemapper(GraphRemapper):

    def __init__(self, relation, graphs, remappers=None):
        super().__init__(relation, graphs, remappers)
        self.fv_order = {g: list(g.free_variables_total) for g in graphs}

    def gen_graph(self, g):
        with About(g.debug, 'grad_bprop'):
            ng = Graph()
        self.graph_repl[g] = ng
        return ng

    def gen_parameter(self, g, ng, p):
        self.gen_apply(g, ng, p)

    def gen_apply(self, g, ng, node):
        with About(node.debug, self.relation):
            if node is g.output:
                new_node = ng.add_parameter()
            else:
                new_node = ng.apply()
        self.add_node((g, node), g, node, ng, new_node)

    def gen_child(self, g, ng, child):
        with About(child.debug, self.relation):
            self.add_node((g, child), g, child, ng, ng.apply())

    def gen_fv(self, g, ng, node):
        with About(node.debug, self.relation):
            self.add_node((g, node), g, node, ng, ng.apply())

    def gen_fv_graph(self, g, ng, g2):
        with About(g2.debug, self.relation):
            self.add_node((g, g2), g, g2, ng, ng.apply())

    def link_apply(self, g, ng, node, new_node):
        mng = g.manager
        assert not new_node.is_parameter()

        contribs = []

        if isinstance(node, Graph):
            uses = set()
            for ct in g.constants:
                if ct.value is node:
                    uses |= mng.uses[ct]
        else:
            uses = mng.uses[node]

        for user, key in uses:
            if user.graph is g:
                if user is user.graph.return_:
                    if len(ng.parameters) == 0:
                        with About(g.output.debug, 'grad_sens'):
                            ng.add_parameter()
                    sexp = (primops.identity, ng.parameters[0])
                    contribs.append(sexp)
                    continue
                src = self.remappers['grad_bprop_app'].get(g, user)
                sexp = (primops.tuple_getitem, src, key)
                contribs.append(sexp)

        # TODO: deconstruct nested graphs

        children = {g2 for g2 in self.graphs
                    if g2.parent is g
                    and node in g2.free_variables_total}

        for child in children:
            idx = self.fv_order[child].index(node)
            assert (g, child) in self.repl
            sexp = (primops.tuple_getitem, self.get(g, child), idx)
            contribs.append(sexp)

        n = len(contribs)
        if n == 0:
            sexp = (zeros_like,
                    (primops.Jinv,
                     self.remappers['grad_fprop'].get(g, node)))
        elif n == 1:
            sexp = contribs[0]
        else:
            sexp = (hyper_add, *contribs)

        new_node.inputs = sexp_to_node(sexp, ng).inputs

    def finalize_graph(self, g, ng):
        fv_sens = [self.get(g, fv) for fv in g.free_variables_total]
        in_sens = [self.get(g, p) for p in g.parameters]
        ng.output = ng.apply(primops.make_tuple,
                             ng.apply(primops.make_tuple, *fv_sens),
                             *in_sens)
        if len(ng.parameters) == 0:
            with About(g.output.debug, 'grad_sens'):
                ng.add_parameter()


class RemapperSet:
    def __init__(self, graphs, **remappers):
        self.remappers = {
            name: remapper(relation=name, graphs=graphs, remappers=self)
            for name, remapper in remappers.items()
        }
        self.graphs = graphs

    def run(self):
        for _, remapper in self.remappers.items():
            remapper.prepare()
        for _, remapper in self.remappers.items():
            remapper.populate()
        for _, remapper in self.remappers.items():
            remapper.link()
        for _, remapper in self.remappers.items():
            remapper.finalize()

    def __getitem__(self, item):
        return self.remappers[item]


class Grad:
    def __init__(self, mng, root):
        graphs = root.scope

        remappers = RemapperSet(
            graphs,
            grad_fprop=FPropRemapper.partial(),
            grad_fprop_app=FPropAppRemapper.partial(
                master='grad_fprop',
                in_remapper='grad_fprop'
            ),
            grad_bprop=BPropRemapper.partial(
                master='grad_fprop'
            ),
            grad_sens=SensRemapper.partial(
            ),
            grad_bprop_app=BPropAppRemapper.partial(
                master='grad_sens'
            ),
        )
        remappers.run()
        self.result = remappers['grad_fprop'].get_graph(root)


@overload
def J(prim: Primitive, resources):
    return clone(augmented_graphs[prim])


@overload
def J(graph: Graph, resources):
    mng = resources.manager
    if graph.transforms.get('grad', None):
        return graph.transforms['grad']
    mng.add_graph(graph)
    res = Grad(mng, graph).result
    return clone(res)


@overload
def J(other: object, resources):
    name = type(other).__qualname__
    raise NotImplementedError(f'J(::{name}) not implemented')

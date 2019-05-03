"""Graph generation from number of arguments or type signatures."""


class GraphGenerationError(Exception):
    """Raised when a graph could not be generated by a MetaGraph."""


class MetaGraph:
    """Graph generator.

    Can be called with a pipeline's resources and a list of argument types to
    generate a graph corresponding to these types.
    """

    def __init__(self, name):
        """Initialize a MetaGraph."""
        self.name = name
        self.cache = {}

    def normalize_args(self, args):
        """Return normalized versions of the arguments.

        By default, this returns args unchanged.
        """
        return args

    def generate_graph(self, args):
        """Generate a Graph for the given abstract arguments."""
        raise NotImplementedError('Override generate_graph in subclass.')

    def __str__(self):
        return self.name


class MultitypeGraph(MetaGraph):
    """Associates type signatures to specific graphs."""

    def __init__(self, name, entries={}):
        """Initialize a MultitypeGraph."""
        super().__init__(name)
        self.entries = list(entries.items())

    def normalize_args(self, args):
        """Return broadened arguments."""
        from ..abstract import broaden
        return tuple(broaden(a, None) for a in args)

    def register(self, *types):
        """Register a function for the given type signature."""
        from ..abstract import type_to_abstract, ANYTHING
        types = tuple(type_to_abstract(t) for t in types)
        if types == (ANYTHING,):
            breakpoint()
        def deco(fn):
            self.entries.append((types, fn))
            return fn
        return deco

    def _getfn(self, types):
        from ..abstract import typecheck, MyiaTypeError
        for sig, fn in self.entries:
            if typecheck(sig, types):
                return fn
        else:
            raise GraphGenerationError(types)

    def generate_graph(self, args):
        """Generate a Graph for the given abstract arguments."""
        from ..abstract.utils import build_type_limited
        from ..parser import parse
        types = tuple(args)
        if types not in self.cache:
            self.cache[types] = parse(self._getfn(types))
        return self.cache[types]

    def __call__(self, *args):
        """Call like a normal function."""
        from ..abstract import to_abstract
        types = tuple(to_abstract(arg) for arg in args)
        fn = self._getfn(types)
        return fn(*args)

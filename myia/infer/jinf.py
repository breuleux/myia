"""Implements JInferrer."""

from ..dtype import SensitivityMap
from ..prim import ops as P
from ..debug.label import short_relation_symbols as syms

from .graph_infer import Inferrer, ExplicitInferrer, TransformedReference


class JInferrer(Inferrer):
    """Inferrer for J(fn).

    Arguments:
        fn: The function to transform.
        mktuple: A function to create a tuple appropriate for the track.
    """

    def __init__(self, fn, mktuple):
        """Initialize a JInferrer."""
        super().__init__(fn.track, 'J')
        self.fn = fn
        assert isinstance(fn, Inferrer)
        self.mktuple = mktuple

    async def infer(self, *jargs):
        """Infer given the arguments."""
        args = [TransformedReference(self.engine, P.Jinv, jarg)
                for jarg in jargs]
        res = await self.fn(*args)
        res_t = self.track.jtag(res)
        bparams_t = [SensitivityMap]
        bparams_t += [self.track.stag(await x[self.track.name]) for x in args]
        bprop_t = ExplicitInferrer(
            self.track,
            [self.track.stag(res)],
            self.mktuple(bparams_t),
            name=f'{syms["grad_bprop"]}{self.fn.identifier}'
        )
        return self.mktuple([res_t, bprop_t])

    def provably_equivalent(self, other):
        """Whether two JInferrers are provably equivalent."""
        return (isinstance(other, JInferrer) and
                self.fn.provably_equivalent(other.fn))

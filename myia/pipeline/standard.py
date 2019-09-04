"""Pre-made pipelines."""


from ..abstract import Context, abstract_inferrer_constructors
from ..ir import GraphManager
from ..pipeline.resources import (
    BackendResource,
    ConverterResource,
    DebugVMResource,
    InferenceResource,
    scalar_object_map,
    standard_method_map,
    standard_object_map,
)
from ..prim import py_registry, vm_registry
from ..utils import Partial, Partializable
from ..validate import (
    validate_abstract as default_validate_abstract,
    whitelist as default_whitelist,
)
from . import steps
from .pipeline import PipelineDefinition


class Resources(Partializable):
    def __init__(self, **members):
        self._members = members
        self._inst = {}

    def __getattr__(self, attr):
        if attr in self._inst:
            return self._inst[attr]

        if attr in self._members:
            inst = self._members[attr]
            if isinstance(inst, Partial):
                try:
                    inst = inst.partial(resources=self)
                except TypeError:
                    pass
                inst = inst()
            self._inst[attr] = inst
            return inst


standard_resources = Resources.partial(
    manager=GraphManager.partial(),
    py_implementations=py_registry,
    method_map=standard_method_map,
    convert=ConverterResource.partial(
        object_map=standard_object_map,
    ),
    inferrer=InferenceResource.partial(
        constructors=abstract_inferrer_constructors,
        context_class=Context,
    ),
    backend=BackendResource.partial(),
    debug_vm=DebugVMResource.partial(
        implementations=vm_registry,
    ),
    operation_whitelist=default_whitelist,
    validate_abstract=default_validate_abstract,
    return_backend=False,
)


######################
# Pre-made pipelines #
######################


standard_pipeline = PipelineDefinition(
    resources=standard_resources,
    steps=dict(
        parse=steps.step_parse,
        resolve=steps.step_resolve,
        infer=steps.step_infer,
        specialize=steps.step_specialize,
        simplify_types=steps.step_simplify_types,
        opt=steps.step_opt,
        opt2=steps.step_opt2,
        cconv=steps.step_cconv,
        validate=steps.step_validate,
        compile=steps.step_compile,
        wrap=steps.step_wrap,
    )
)


scalar_pipeline = standard_pipeline.configure({
    'convert.object_map': scalar_object_map,
})


standard_debug_pipeline = PipelineDefinition(
    resources=standard_resources,
    steps=dict(
        parse=steps.step_parse,
        resolve=steps.step_resolve,
        infer=steps.step_infer,
        specialize=steps.step_specialize,
        simplify_types=steps.step_simplify_types,
        opt=steps.step_opt,
        opt2=steps.step_opt2,
        cconv=steps.step_cconv,
        validate=steps.step_validate,
        export=steps.step_debug_export,
        wrap=steps.step_wrap,
    )
).configure({
    'backend.name': False
})


scalar_debug_pipeline = standard_debug_pipeline.configure({
    'convert.object_map': scalar_object_map
})


######################
# Pre-made utilities #
######################


scalar_parse = scalar_pipeline \
    .select('parse', 'resolve') \
    .make_transformer('input', 'graph')


scalar_debug_compile = scalar_debug_pipeline \
    .select('parse', 'resolve', 'export') \
    .make_transformer('input', 'output')

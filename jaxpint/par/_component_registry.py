"""Self-registration registry for timing-model components.

A component declares its identity + ``build`` on the class itself, via
:func:`register_component` (the 1:1 class decorator) or :func:`register_family`
(many classes -> one :class:`~jaxpint.par.registry.Component`, e.g. the
binaries).  The registry / parser / model-builder wiring then *derive* from that
one declaration instead of from parallel manual tables.

Registration is a **side effect of importing the component module**.  This
module holds only the registry dict + the (de)registration API and imports
nothing from the component packages, so reading it (:func:`registered`) never
triggers a component import — callers import the components themselves first,
then read what registered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from jaxpint.par.registry import Component


@dataclass(frozen=True)
class RegisteredComponent:
    """One component's self-declared identity + build.

    ``classes`` are the component classes whose ``PARAMS`` feed the parser
    (usually one; a family lists several).  ``build`` is the ``build(ctx)``
    callable the model builder dispatches to.
    """

    component: Component
    classes: tuple[type, ...]
    pint_names: tuple[str, ...]
    is_binary: bool
    build: Callable


_REGISTRY: dict[Component, RegisteredComponent] = {}


def register_family(
    *,
    component: Component,
    classes,
    build: Callable,
    pint_names=(),
    is_binary: bool = False,
) -> None:
    """Register a family (>=1 classes) under one ``Component``.

    Used directly for the many-to-one cases (the binaries); the 1:1
    :func:`register_component` decorator delegates here.
    """
    _REGISTRY[component] = RegisteredComponent(
        component=component,
        classes=tuple(classes),
        pint_names=tuple(pint_names),
        is_binary=is_binary,
        build=build,
    )


def register_component(*, component: Component, pint_names=()):
    """Class decorator for the 1:1 case (one class <-> one ``Component``).

    The decorated class must provide a ``build(cls, ctx)`` classmethod; that
    bound classmethod becomes the component's builder.
    """

    def _decorator(cls):
        register_family(
            component=component,
            classes=(cls,),
            build=cls.build,
            pint_names=pint_names,
        )
        return cls

    return _decorator


def registered() -> dict[Component, RegisteredComponent]:
    """A snapshot of the registry (populated as component modules import).

    Pure accessor — imports nothing, so it is safe to call from the lazy table
    assembly once the component packages have finished importing.
    """
    return dict(_REGISTRY)

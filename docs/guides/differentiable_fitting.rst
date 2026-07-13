Differentiable Fitting via the Implicit Function Theorem
=========================================================

:meth:`~jaxpint.fitters.BaseFitter.fit_params` returns fitted timing
parameters that are **differentiable** with respect to the frozen parameter
values and any injected ``external_delay`` — without backpropagating through
a single Gauss-Newton iteration. This page explains how that works and when
the gradients can be trusted.

Why differentiate through a fit?
--------------------------------

Several pipelines need gradients *of the fitted parameters* (or of a
statistic built from them) with respect to inputs the fit holds fixed:

* **CW injection studies** — an injected continuous-wave signal enters as an
  ``external_delay`` subtracted from the residuals; sensitivity analyses
  differentiate detection statistics through the timing fit with respect to
  the signal parameters.
* **Frozen-parameter sensitivity** — how the maximum-likelihood values of the
  free parameters respond to a perturbation in a parameter held frozen
  (e.g. profiling a likelihood over a subset of parameters).

The naive route — ``jax.grad`` straight through the iteration loop — unrolls
every Gauss-Newton step into the computation graph. That is slow to compile,
stores every intermediate for the backward pass, and the gradient quality
degrades with loop length. The implicit function theorem (IFT) sidesteps all
of it.

The fit as a fixed point
------------------------

Write the free-parameter values as :math:`y` and collect everything the fit
holds fixed — frozen parameter values and the ``external_delay`` — as
:math:`\theta`. The fit minimizes the (generalized) least-squares objective

.. math::

   \chi^2(y, a;\, \theta)
   \;=\;
   \bigl(r(y, \theta) - a\mathbf{1}\bigr)^T
   C^{-1}
   \bigl(r(y, \theta) - a\mathbf{1}\bigr)

where

* :math:`r` are the timing residuals (narrowband), or the stacked
  ``[time; dm]`` residual vector (wideband);
* :math:`C` is the fitter's noise covariance — diagonal for WLS,
  :math:`N + U \Phi U^T` for GLS (applied via the Woodbury identity, never
  formed densely). :math:`C` depends only on the (frozen) noise parameters,
  i.e. on :math:`\theta`, so it is constant with respect to :math:`y`;
* :math:`a` is the implicit constant offset. The absolute-phase ambiguity
  means a constant can always be absorbed into the residuals, so fitters
  without an explicit ``PhaseOffset`` parameter carry a synthetic offset
  column :math:`\mathbf{1}` alongside the physical parameters. (With an
  explicit ``PhaseOffset``, the offset is just one of the entries of
  :math:`y`, there is no :math:`a`, and the projector derived below never
  appears — ``_offset_vector`` returns ``None``.)

Deriving the stationarity condition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First minimize over the offset :math:`a` at fixed :math:`y`. Setting

.. math::

   \frac{\partial \chi^2}{\partial a}
   \;=\;
   -2\, \mathbf{1}^T C^{-1} \bigl(r - a\mathbf{1}\bigr)
   \;=\; 0

gives the :math:`C`-weighted mean of the residuals, and substituting it back
turns the offset-corrected residuals into a projection:

.. math::

   a^*(y) \;=\; \frac{\mathbf{1}^T C^{-1} r}{\mathbf{1}^T C^{-1} \mathbf{1}},
   \qquad
   r - a^*\mathbf{1} \;=\; (I - P)\, r,
   \qquad
   P \;=\; \frac{\mathbf{1}\,\mathbf{1}^T C^{-1}}{\mathbf{1}^T C^{-1} \mathbf{1}}.

:math:`P` is the :math:`C`-weighted (oblique) projector onto the offset
direction — exactly what the ``o``/``zo`` terms in ``_optimality`` and
``_solve_gn_normal`` compute. Getting this projector wrong would make the
backward pass silently inconsistent with the forward fit.

Now differentiate the profiled objective :math:`\chi^2\bigl(y, a^*(y)\bigr)`
with respect to :math:`y`. The chain-rule term through :math:`a^*(y)`
vanishes because :math:`\partial \chi^2 / \partial a = 0` at the optimum
(the envelope theorem), leaving only the explicit :math:`y`-dependence:

.. math::

   \frac{d \chi^2}{d y}
   \;=\;
   2 \left(\frac{\partial r}{\partial y}\right)^{\!T} C^{-1} (I - P)\, r
   \;=\;
   -2\, M^T C^{-1} (I - P)\, r

where :math:`M = -\partial r / \partial y` is the design matrix restricted
to the free columns. A minimizer :math:`y^*` therefore satisfies the
**stationarity condition**

.. math::

   G(y^*, \theta) \;=\; M^T C^{-1} (I - P)\, r(y^*, \theta) \;=\; 0.

This is also exactly the fixed-point condition of the forward solver: one
Gauss-Newton step updates :math:`y` by :math:`\delta = H^{-1} G(y, \theta)`
(with :math:`H` the normal matrix defined below), so the iteration is
stationary precisely when :math:`G = 0` — the update
:meth:`~jaxpint.fitters.BaseFitter.fit_gap` measures.

The key point: :math:`G = 0` *defines* :math:`y^*` as an implicit function
of :math:`\theta`. The iteration that found :math:`y^*` is irrelevant to how
:math:`y^*` moves when :math:`\theta` moves.

Differentiating the fixed point
-------------------------------

Differentiate :math:`G(y^*(\theta), \theta) = 0` through :math:`\theta` and
solve for the sensitivity of the fixed point:

.. math::

   \frac{dy^*}{d\theta}
   \;=\;
   -\left(\frac{\partial G}{\partial y}\right)^{-1}
   \frac{\partial G}{\partial \theta}.

Under the **Gauss-Newton approximation** — dropping terms involving second
derivatives of :math:`r`, exactly the approximation the forward solver
already makes — the Jacobian of :math:`G` is (minus) the normal matrix:

.. math::

   \frac{\partial G}{\partial y} \;\approx\; -H,
   \qquad
   H = M_{ms}^T C^{-1} M_{ms},

where :math:`M_{ms}` is the design matrix with each column's
:math:`C`-weighted mean removed (the projector :math:`P` again, applied to
the columns).

The backward pass therefore costs **one linear solve plus one VJP**,
independent of how many forward iterations ran, and stores nothing per
iteration.

How the code maps onto the math
-------------------------------

:meth:`~jaxpint.fitters.BaseFitter.fit_params` wraps the iteration in
a ``jax.custom_vjp``:

* **Forward** (``_fp_fwd``): run the plain Gauss-Newton loop with no autodiff
  bookkeeping; save only :math:`y^*` and the external delay.
* **Backward** (``_fp_bwd``): given the incoming cotangent :math:`v`,

  1. solve :math:`H u = v` on the free entries
     (``_solve_gn_normal``). The solve
     deliberately reuses the forward step's column normalization and relative
     SVD threshold, so the backward pass truncates the *same* degenerate
     directions as the forward fit;
  2. pull :math:`u` back through the VJP of the stationarity map
     (``_optimality``), which yields
     the gradients with respect to the frozen parameter values and the
     ``external_delay``.

Two bookkeeping subtleties in the backward pass:

* the **free** entries of the input vector are only the iteration *seed* — a
  converged answer does not depend on where the iteration started, so they
  receive zero gradient;
* the **frozen** entries of the output pass through from the input unchanged,
  so they receive an identity gradient in addition to the IFT term.

Each concrete fitter supplies its own pieces through the hooks
``_core_step`` (one forward Gauss-Newton update), ``_fit_cinv`` (apply
:math:`C^{-1}` for its noise model), ``_fit_residuals`` (residual layout),
and ``_offset_vector`` — the IFT machinery in
:class:`~jaxpint.fitters.BaseFitter` is shared by WLS, GLS, and
wideband fitters alike.

When to trust the gradients
---------------------------

The IFT is exact only **at a true fixed point**: the derivation starts from
:math:`G(y^*, \theta) = 0`. If the fit has not converged (few iterations, a
poor starting point), the gradients are approximations with no error bound.

:meth:`~jaxpint.fitters.BaseFitter.fit_gap` is the diagnostic: it
returns the free-parameter update one further Gauss-Newton step would make.
When the gap is a small fraction of each parameter's posterior sigma, the
fixed-point assumption — and hence the implicit gradients — is sound.

References
----------

* JAX docs, *Custom derivative rules for Python code* — the "implicit
  function differentiation of iterative implementations" example is exactly
  this ``custom_vjp``-around-a-fixed-point pattern:
  https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html
* Blondel et al., *Efficient and Modular Implicit Differentiation*
  (NeurIPS 2022), https://arxiv.org/abs/2105.15183 — the general recipe
  (JAXopt) of pairing a solver with an optimality condition :math:`G = 0`
  and differentiating via the IFT.

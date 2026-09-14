import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # CGW sky localization vs. anchor pulsars

    A walkthrough reproduction of **Wen et al. 2026** (arXiv:2603.28897,
    *"From Detection to Host Galaxy Identification: Precision CGW
    Localization with a Few Anchor Pulsars"*): a small subset of pulsars
    with sub-wavelength PX precision is enough to phase-lock the array and
    dramatically shrink the 90% credible sky-localization area.

    ## Method — Fisher matrix at the truth point

    1. Each pixel of a HEALPix grid is a candidate *true* sky position. At
       that pixel the unit-strain signal power $Y = (\hat s \mid \hat s)$
       is read off the likelihood's quadratic dependence on the linear
       amplitude, and $h_0$ is calibrated per pixel so the optimal
       matched-filter SNR equals `SNR_TARGET`.
    2. With $h_0$ fixed at that calibration, the timing-marginalized
       log-likelihood is approximately quadratic in the sky parameters
       $(\cos\theta_{\rm gw}, \phi_{\rm gw})$ near the truth, so the 2-D
       Fisher information is $F = -\nabla^2_{\rm sky}\log L$.
    3. The credible area is $\pi\,\chi^2_2(p)\,\sqrt{\det F^{-1}}$ steradians
    """)
    return


@app.cell
def _():
    import os
    from pathlib import Path

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    from loguru import logger

    from jaxpint.notebook_utils import (
        SMOKE_SUBSET,
        healpix_grid,
        import_healpy,
        load_filtered_pta,
        marginalize_pta_timing,
        overlay_pulsars,
    )
    from jaxpint.pta.cw_localization import (
        credible_area_deg2,
        h0_for_snr,
        marginal_sky_fisher,
        signal_power_direct,
    )
    from jaxpint.pta.likelihood import PTAConfig
    from jaxpint.pta.signals.cw import CWInjector
    from jaxpint.types import GlobalParams

    hp = import_healpy()  # fails early if the optional `skymap` extra is missing
    logger.disable("pint")
    return (
        CWInjector,
        GlobalParams,
        PTAConfig,
        Path,
        SMOKE_SUBSET,
        credible_area_deg2,
        h0_for_snr,
        healpix_grid,
        hp,
        jax,
        jnp,
        load_filtered_pta,
        marginal_sky_fisher,
        marginalize_pta_timing,
        np,
        os,
        overlay_pulsars,
        plt,
        signal_power_direct,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Source and array

    Fiducial source matched to Wen et al.: $\mathcal{M}_c = 5\times10^8\,
    M_\odot$, $f_{\rm GW} = 10^{-8.4}\,$Hz $\approx 4\,$nHz, orientation
    held at face-on/optimal.
    """)
    return


@app.cell
def _():
    LOG10_FGW = -8.4  # f_GW = 10^-8.4 Hz ~ 4 nHz
    # Generic inclination, NOT face-on
    FIXED_ORIENTATION = (0.5, 0.3, 0.7)  # (cos_inc, psi, phase0)
    SIGMA_ANCHOR_KPC = 1.0e-3  # 1 pc, Wen's D_err (lambda_GW = 2.44 pc)

    # The 18 Wen pulsars that are also in ocarina.
    WEN_OCARINA_18 = (
        # EPTA+InPTA in ocarina (5 of 6)
        "J0613-0200",
        "J1024-0719",
        "J1600-3053",
        "J1730-2304",
        "J1843-1113",
        # MPTA in ocarina (1 of 5)
        "J0437-4715",
        # NANOGrav in ocarina (10 of 10)
        "J0030+0451",
        "J1640+2224",
        "J1741+1351",
        "J1909-3744",
        "J1911+1347",
        "J2017+0603",
        "J2043+1711",
        "J2234+0611",
        "J2234+0944",
        "J2317+1439",
        # PPTA in ocarina (2 of 4)
        "J1713+0747",
        "J1744-1134",
    )

    # Wen's three discrete array configurations (analogs).
    _A3 = ("J0437-4715", "J0030+0451", "J1713+0747")
    _A6 = _A3 + ("J1640+2224", "J1744-1134", "J1909-3744")
    WEN_CONFIGS = (
        # (label, subset_spec, anchors); subset_spec: None = full array,
        # ("drop", names) = full minus names, ("only", names) = just names.
        ("standard", None, ()),
        ("25-3", None, _A3),
        ("25-6", None, _A6),
        ("22-0", ("drop", _A3), ()),
        ("19-0", ("drop", _A6), ()),
        ("3-3", ("only", _A3), _A3),
    )
    # Shared styling: anchored arrays solid, controls dashed (Fig. 2 + Fig. 3).
    WEN_STYLES = {
        "standard": ("C0", "-"),
        "25-3": ("C1", "-"),
        "25-6": ("C2", "-"),
        "22-0": ("C0", "--"),
        "19-0": ("C4", "--"),
        "3-3": ("C3", "--"),
    }
    return (
        FIXED_ORIENTATION,
        LOG10_FGW,
        SIGMA_ANCHOR_KPC,
        WEN_CONFIGS,
        WEN_OCARINA_18,
        WEN_STYLES,
    )


@app.cell
def _(mo, os):
    # ocarina_white by default: white-noise-only synthetic build
    data_dir_ui = mo.ui.text(
        value=os.environ.get(
            "JAXPINT_OCARINA_DIR", "/home/hector/NYU/PTA/jax_pint/ocarina_white"
        ),
        label="ocarina par/tim directory",
        full_width=True,
    )
    array_ui = mo.ui.radio(
        options={
            "smoke — 4 pulsars (fast)": "smoke",
            "Wen-ocarina — 18 pulsars (slow)": "full",
        },
        value="Wen-ocarina — 18 pulsars (slow)",
        label="array",
    )
    nside_ui = mo.ui.dropdown(
        options=["1", "2", "4", "8"], value="4", label="HEALPix nside"
    )
    snr_ui = mo.ui.slider(5, 50, value=20, step=1, label="target SNR", show_value=True)
    chunk_ui = mo.ui.dropdown(
        options=["8", "16", "32", "64"], value="8", label="pixels per vmap chunk"
    )
    return array_ui, chunk_ui, data_dir_ui, nside_ui, snr_ui


@app.cell
def _(SMOKE_SUBSET, WEN_OCARINA_18, array_ui):
    pulsar_subset = (
        tuple(WEN_OCARINA_18) if array_ui.value == "full" else tuple(SMOKE_SUBSET)
    )
    return (pulsar_subset,)


@app.cell
def _(WEN_CONFIGS, mo, pulsar_subset):
    # Seed the anchor picker with Wen's 25-3 set, intersected with whichever
    # array is selected.
    _cfg_anchors = {name: anchors for name, _sub, anchors in WEN_CONFIGS}
    _seed = [a for a in _cfg_anchors["25-3"] if a in pulsar_subset]
    anchor_ui = mo.ui.multiselect(
        options=list(pulsar_subset),
        value=_seed,
        label="anchor pulsars (pulsar term included, PX pegged)",
    )
    return (anchor_ui,)


@app.cell(hide_code=True)
def _(anchor_ui, array_ui, chunk_ui, data_dir_ui, mo, nside_ui, snr_ui):
    mo.vstack(
        [
            data_dir_ui,
            mo.hstack([array_ui, nside_ui, chunk_ui], justify="start", gap=2),
            snr_ui,
            anchor_ui,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Load and filter the PTA
    """)
    return


@app.cell
def _(Path, data_dir_ui, load_filtered_pta, pulsar_subset):
    pta = load_filtered_pta(
        Path(data_dir_ui.value).expanduser(), pulsar_names=list(pulsar_subset)
    )
    names = list(pta.names)
    n_toa_total = int(sum(int(td.n_toas) for td in pta.toa_data_list))
    return n_toa_total, names, pta


@app.cell(hide_code=True)
def _(mo, n_toa_total, names):
    mo.md(f"""
    Loaded **{len(names)}** pulsars, **{n_toa_total}** TOAs: `{names}`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. The anchor mask

    `True` → that pulsar's pulsar term is in the model with PX pegged to the par value,
    `False` → Earth-term-only.
    """)
    return


@app.cell
def _(anchor_ui, names):
    anchor_set = set(anchor_ui.value)
    _unknown = anchor_set - set(names)
    if _unknown:
        raise ValueError(f"Anchors not in the loaded array: {sorted(_unknown)}")
    pulsar_term_mask = tuple(name in anchor_set for name in names)
    n_anchors = sum(pulsar_term_mask)
    return n_anchors, pulsar_term_mask


@app.cell(hide_code=True)
def _(mo, n_anchors, names, pulsar_term_mask):
    mo.md(
        f"**{n_anchors}/{len(names)}** anchors — "
        + ", ".join(
            f"{'**' + n + '**' if m else n}" for n, m in zip(names, pulsar_term_mask)
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. The CW template

    One `CWInjector` with `linear_amplitude=True`: the residual is *exactly*
    linear in $h_0$, so the delay at $h_0 = 1$ **is** the unit-strain waveform
    $\hat s$.
    """)
    return


@app.cell
def _(CWInjector, GlobalParams, LOG10_FGW, PTAConfig, jnp):
    def build_injectors(pta_obj, mask):
        """One linear-amplitude CW template (`cwt_`) + globals + PTAConfig."""
        positions_ = jnp.asarray(pta_obj.positions)
        template = CWInjector(
            positions_,
            prefix="cwt_",
            earth_term_only=False,
            linear_amplitude=True,
            pulsar_term_mask=mask,
            initial_values={"log10_fgw": LOG10_FGW},
        )
        gp_ = template.register_params(GlobalParams.empty())
        cfg = PTAConfig(
            toa_data_list=pta_obj.toa_data_list,
            timing_models=pta_obj.timing_models,
            noise_models=pta_obj.noise_models,
            signal_injectors=(template,),
        )
        return template, gp_, cfg

    return (build_injectors,)


@app.cell
def _(build_injectors, pta, pulsar_term_mask):
    template_injector, gp, config = build_injectors(pta, pulsar_term_mask)
    return config, gp


@app.cell(hide_code=True)
def _(gp, mo):
    mo.md(f"""
    Registered global params: `{sorted(gp.names)}`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Marginalize the timing models

    Analytic marginalization over the linear timing-model parameters with
    improper priors, done **once** — every pixel afterwards reuses the
    reduced likelihood `g`.
    """)
    return


@app.cell
def _(build_injectors, config, gp, marginalize_pta_timing, pta):
    g, _marg_extra, reduced_pp, marg_config = marginalize_pta_timing(
        pta,
        config,
        gp,
        validate_linearity=False,
        return_config=True,
    )
    # ALL-coherent calibration config: h0 for "SNR 20" is defined against the
    # full array with every pulsar term present NOT against the per-config model.
    _tc, cal_gp, cal_config = build_injectors(pta, tuple(True for _ in pta.names))
    g_cal, _e, cal_rpp, cal_marg_config = marginalize_pta_timing(
        pta, cal_config, cal_gp, validate_linearity=False, return_config=True
    )
    _ = g_cal(cal_gp, cal_rpp)  # eager warm-up (cached_property tracer guard)
    return cal_gp, cal_marg_config, g, marg_config, reduced_pp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Truth Source Location
    Specifies true global parameters of CGW for Fischer Matrix evaluation.
    """)
    return


@app.cell
def _(FIXED_ORIENTATION):
    def pin_orientation(gp_, orientation=FIXED_ORIENTATION):
        """Freeze (cos_inc, psi, phase0) in the injector at the truth point."""
        cos_inc_, psi_, phase0_ = (float(x) for x in orientation)
        return (
            gp_.with_value("cwt_cos_inc", cos_inc_)
            .with_value("cwt_psi", psi_)
            .with_value("cwt_phase0", phase0_)
        )

    return (pin_orientation,)


@app.cell
def _(gp, pin_orientation):
    gp_fixed = pin_orientation(gp)
    return (gp_fixed,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Per-pixel Fisher → credible area

    Per pixel: calibrate $h_0$ against the **all-coherent** array (the
    injection convention of Wen), then
    compute the **marginal** sky Fisher — orientation
    $(\cos\iota, \psi, \phi_0)$ and $\log_{10} f_{\rm gw}$ marginalized with
    flat priors, anchor distances with Gaussian priors
    $\sigma_d = 0.1\,\lambda_{\rm GW}$ — and invert to an area.
    """)
    return


@app.cell
def _(
    SIGMA_ANCHOR_KPC,
    credible_area_deg2,
    h0_for_snr,
    jax,
    jnp,
    marginal_sky_fisher,
    np,
    signal_power_direct,
):
    # Marginalize what the Wen paper marginalizes.
    NUISANCE = ("cos_inc", "psi", "phase0", "log10_fgw")

    def pixel_areas(
        config_,
        gp_,
        pulsar_params_,
        sky,
        *,
        snr_target,
        pixel_chunk=32,
        cal_config=None,
        cal_gp=None,
        cal_pulsar_params=None,
    ):
        """(90%, 50%) MARGINAL credible area in deg^2 for every pixel of `sky`.

        `cal_config`/`cal_gp`/`cal_pulsar_params`: the all-coherent FULL-array
        configuration used to define h0(SNR) -- the same source is evaluated in
        every configuration, so reduced arrays (22-0, 19-0, 3-3) share the full
        array's h0 rather than re-calibrating in their own, weaker array.
        Default to the evaluation config if not given.
        """
        cc = config_ if cal_config is None else cal_config
        cg = gp_ if cal_gp is None else cal_gp
        cp = pulsar_params_ if cal_pulsar_params is None else cal_pulsar_params
        sigmas = [SIGMA_ANCHOR_KPC] * config_.n_pulsars  # read for coherent only

        def area_for_pixel(sky_row):
            Y = signal_power_direct(cc, cg, cp, sky_row)
            h0 = h0_for_snr(jnp.float64(snr_target), Y)
            F = marginal_sky_fisher(
                config_,
                gp_,
                pulsar_params_,
                sky_row,
                h0=h0,
                global_nuisance=NUISANCE,
                dist_sigma_kpc=sigmas,
            )
            return (
                credible_area_deg2(F, level=0.9),
                credible_area_deg2(F, level=0.5),
            )

        @jax.jit
        def all_areas(sky_arr):
            return jax.lax.map(area_for_pixel, sky_arr, batch_size=pixel_chunk)

        a90, a50 = all_areas(sky)
        return np.asarray(a90), np.asarray(a50)

    return (pixel_areas,)


@app.cell
def _(healpix_grid, nside_ui):
    grid = healpix_grid(int(nside_ui.value))
    return (grid,)


@app.cell
def _(
    cal_gp,
    cal_marg_config,
    chunk_ui,
    g,
    gp,
    gp_fixed,
    grid,
    marg_config,
    pin_orientation,
    pixel_areas,
    pta,
    reduced_pp,
    snr_ui,
):
    _ = g(gp, reduced_pp)  # eager warm-up — see the note above
    area_90, area_50 = pixel_areas(
        marg_config,
        gp_fixed,
        pta.pulsar_params_list,
        grid.sky,
        snr_target=float(snr_ui.value),
        pixel_chunk=int(chunk_ui.value),
        cal_config=cal_marg_config,
        cal_gp=pin_orientation(cal_gp),
    )
    return area_50, area_90


@app.cell(hide_code=True)
def _(area_50, area_90, grid, mo, n_anchors, names, np):
    def _stats(a):
        f = a[np.isfinite(a)]
        return (
            f"{f.min():.3e} / {np.median(f):.3e} / {f.max():.3e} "
            f"({len(f)}/{len(a)} finite)"
        )

    mo.md(
        f"""
        `nside={grid.nside}` → **{grid.npix}** pixels,
        **{n_anchors}/{len(names)}** anchors.

        | level | min / median / max [deg²] |
        |---|---|
        | 90% | {_stats(area_90)} |
        | 50% | {_stats(area_50)} |
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The sky map
    """)
    return


@app.cell
def _(mo):
    level_ui = mo.ui.radio(
        options={"90%": "90", "50%": "50"}, value="90%", label="credible level"
    )
    return (level_ui,)


@app.cell
def _(level_ui):
    level_ui
    return


@app.cell
def _(
    area_50,
    area_90,
    hp,
    level_ui,
    n_anchors,
    names,
    np,
    overlay_pulsars,
    plt,
    pta,
    pulsar_term_mask,
    snr_ui,
):
    def plot_map(area, level, snr, n_anc, n_psr, positions, mask):
        # Log scale — area ranges over orders of magnitude across anchor configs.
        log_area = np.log10(np.where(np.isfinite(area) & (area > 0), area, np.nan))
        hp.mollview(
            log_area,
            title=(
                f"$\\log_{{10}}$ {level}% credible area [deg$^2$]  "
                f"($\\mathcal{{M}}=5\\times 10^8 M_\\odot$, "
                f"$f=10^{{-8.4}}$ Hz, SNR={snr:.0f})\n"
                f"{n_anc}/{n_psr} anchor pulsars"
            ),
            unit=f"$\\log_{{10}}$ {level}% area [deg$^2$]",
            cmap="magma_r",
            rot=[180, 0],
        )
        hp.graticule()
        overlay_pulsars(
            np.asarray(positions),
            np.asarray(mask),
            star_kwargs=dict(s=180, label="anchor"),
            dot_kwargs=dict(
                s=40,
                color="0.5",
                edgecolors="black",
                linewidths=0.4,
                zorder=4,
                label="non-anchor",
            ),
        )
        return plt.gcf()

    plot_map(
        area_90 if level_ui.value == "90" else area_50,
        level_ui.value,
        float(snr_ui.value),
        n_anchors,
        len(names),
        pta.positions,
        pulsar_term_mask,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Table 1 analog — the anchor sweep

    The actual Wen result: replay the pipeline for all **six** Fig.-3
    configurations. Three anchor arrays (standard / 25-3 / 25-6) show the
    credible area collapsing as anchors are added; three controls pin down
    *why*: **22-0** and **19-0** remove the anchor pulsars entirely , and **3-3** is the anchors alone, without the
    array's geometric baseline. All six share one h0, calibrated against the
    full coherent array — same source, different arrays.

    Each config re-runs steps 2–6 against the PTA step 1 already loaded — the
    subset is identical across configs, only the anchor mask changes — so the
    dataset is loaded once, not four times. Still gated behind a button: with
    the 18-pulsar array it takes minutes per config.
    """)
    return


@app.cell
def _(mo):
    run_sweep_ui = mo.ui.run_button(label="Run the 6-config sweep")
    return (run_sweep_ui,)


@app.cell
def _(run_sweep_ui):
    run_sweep_ui
    return


@app.cell
def _(
    build_injectors,
    cal_gp,
    cal_marg_config,
    marginalize_pta_timing,
    pin_orientation,
    pixel_areas,
):
    def subset_pta(pta_, spec):
        """Slice an already-loaded PTA in memory (no reload -- host-RAM fix).

        `spec`: None = full array; ("drop", names) = full minus names;
        ("only", names) = just those pulsars.  LoadedPTA is a NamedTuple of
        parallel tuples plus a positions array, so subsetting is index games.
        """
        if spec is None:
            return pta_
        mode, names_ = spec
        keep = [
            i
            for i, n in enumerate(pta_.names)
            if (n not in set(names_)) == (mode == "drop")
        ]
        import numpy as _np

        return pta_._replace(
            toa_data_list=tuple(pta_.toa_data_list[i] for i in keep),
            pulsar_params_list=tuple(pta_.pulsar_params_list[i] for i in keep),
            timing_models=tuple(pta_.timing_models[i] for i in keep),
            noise_models=tuple(pta_.noise_models[i] for i in keep),
            names=tuple(pta_.names[i] for i in keep),
            positions=_np.asarray(pta_.positions)[keep],
        )

    def config_setup(pta_, subset_spec, anchors):
        """Subset + injectors + timing marginalization for one configuration.

        Returns ``(marg_config, gp_pinned, pulsar_params)`` -- everything the
        direct Fisher helpers need.  Shared by the area sweep (Fig. 3) and the
        single-pixel credible-region computation (Fig. 2).
        """
        sub_ = subset_pta(pta_, subset_spec)
        mask_ = tuple(n in set(anchors) for n in sub_.names)
        _t, gp_, config_ = build_injectors(sub_, mask_)
        g_, _extra, reduced_pp_, marg_config_ = marginalize_pta_timing(
            sub_, config_, gp_, validate_linearity=False, return_config=True
        )
        _ = g_(gp_, reduced_pp_)  # eager warm-up before any autodiff
        return marg_config_, pin_orientation(gp_), sub_.pulsar_params_list

    def localization_map(
        pta_, anchors, *, grid_, snr_target, pixel_chunk, subset_spec=None
    ):
        """Steps 2-6 for one (subset, anchor) config, against the loaded PTA.

        Subsets are sliced in memory from the step-1 `pta` -- reloading per
        config would hold a full dataset resident alongside step 1's
        (the host-OOM failure mode) and invalidate the single-map compilation
        via `load_filtered_pta`'s default `clear_jit_cache=True`.  h0 stays
        calibrated against the FULL all-coherent array (cal_* args below), so
        every configuration sees the same source.
        """
        full_ppl = pta_.pulsar_params_list
        marg_config_, gp_pinned_, ppl_ = config_setup(pta_, subset_spec, anchors)
        return pixel_areas(
            marg_config_,
            gp_pinned_,
            ppl_,
            grid_.sky,
            snr_target=snr_target,
            pixel_chunk=pixel_chunk,
            cal_config=cal_marg_config,
            cal_gp=pin_orientation(cal_gp),
            cal_pulsar_params=full_ppl,
        )

    return config_setup, localization_map


@app.cell
def _(
    WEN_CONFIGS,
    chunk_ui,
    grid,
    localization_map,
    mo,
    np,
    pta,
    pulsar_subset,
    run_sweep_ui,
    snr_ui,
):
    mo.stop(
        not run_sweep_ui.value,
        mo.md("*Press **Run the 6-config sweep** above to compute.*"),
    )

    sweep_rows = []
    for _cfg_name, _cfg_subset, _cfg_anchors in WEN_CONFIGS:
        _in_subset = tuple(a for a in _cfg_anchors if a in pulsar_subset)
        _missing = tuple(a for a in _cfg_anchors if a not in pulsar_subset)
        _a90, _a50 = localization_map(
            pta,
            _in_subset,
            grid_=grid,
            snr_target=float(snr_ui.value),
            pixel_chunk=int(chunk_ui.value),
            subset_spec=_cfg_subset,
        )
        if _cfg_subset is None:
            _n_psr = len(pulsar_subset)
        elif _cfg_subset[0] == "drop":
            _n_psr = len([n for n in pulsar_subset if n not in set(_cfg_subset[1])])
        else:
            _n_psr = len([n for n in pulsar_subset if n in set(_cfg_subset[1])])
        _f90, _f50 = _a90[np.isfinite(_a90)], _a50[np.isfinite(_a50)]
        if len(_f90) == 0 or len(_f50) == 0:
            # A degenerate configuration (e.g. a 1-pulsar analog subset on the
            # smoke array) legitimately yields a singular Fisher at every
            # pixel.  Report it rather than crashing the sweep on np.min([]).
            _f90 = _f50 = np.array([np.nan])
        sweep_rows.append(
            {
                "config": _cfg_name,
                "n_pulsars": _n_psr,
                "n_anchors": len(_in_subset),
                "area_90_median": float(np.median(_f90)),
                "area_90_min": float(np.min(_f90)),
                "area_90_max": float(np.max(_f90)),
                "area_90_p10": float(np.percentile(_f90, 10)),
                "area_90_p90": float(np.percentile(_f90, 90)),
                "area_50_median": float(np.median(_f50)),
                "area_50_min": float(np.min(_f50)),
                "area_50_max": float(np.max(_f50)),
                "area_50_p10": float(np.percentile(_f50, 10)),
                "area_50_p90": float(np.percentile(_f50, 90)),
                "anchors": " ".join(_in_subset),
                "dropped_anchors": " ".join(_missing),
            }
        )
    return (sweep_rows,)


@app.cell
def _(mo, sweep_rows):
    mo.ui.table(sweep_rows, selection=None)
    return


@app.cell
def _(np, plt, pulsar_subset, snr_ui, sweep_rows):
    def plot_scaling(rows, snr, n_psr):
        ks = np.arange(len(rows))
        fig, ax = plt.subplots(figsize=(7, 4))
        for level, marker, color in [("90", "o", "C0"), ("50", "s", "C1")]:
            med = np.array([r[f"area_{level}_median"] for r in rows])
            lo = np.array([r[f"area_{level}_p10"] for r in rows])
            hi = np.array([r[f"area_{level}_p90"] for r in rows])
            ax.fill_between(ks, lo, hi, alpha=0.2, color=color)
            ax.plot(
                ks,
                med,
                marker=marker,
                lw=2,
                color=color,
                label=f"{level}% area (median)",
            )
        ax.set_xticks(ks)
        ax.set_xticklabels([r["config"] for r in rows])
        ax.set_yscale("log")
        ax.set_xlabel("array configuration")
        ax.set_ylabel("credible localization area [deg$^2$]")
        ax.set_title(
            "CGW localization area, Wen et al. 2026 Table 1 analog\n"
            f"($\\mathcal{{M}}=5\\times 10^8 M_\\odot$, SNR={snr:.0f}, "
            f"{n_psr}-pulsar ocarina subset)"
        )
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        return fig

    plot_scaling(sweep_rows, float(snr_ui.value), len(pulsar_subset))
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 3 analog — credible area vs. credibility level

    Wen et al. Figure 3 plots the credible sky area against the credibility
    level for each array configuration at a reference sky location.

    The shaded band spans min→max across the sky, the same framing the paper's
    abstract uses: *"~0.1 to 9.2 deg² at a signal-to-noise ratio of 20"* for the
    25-6 configuration, drawn as the black marker at 90% for reference.
    """)
    return


@app.cell
def _(WEN_STYLES, np, plt, pulsar_subset, snr_ui, sweep_rows):
    # Wen et al. 2026 abstract: the 25-6 config gives 90% credible areas of
    # ~0.1 to 9.2 deg^2 at S/N = 20, across sky directions.
    WEN_25_6_RANGE_90 = (0.1, 9.2)

    def plot_area_vs_level(rows, snr, n_psr):
        levels = np.linspace(0.0, 0.99, 2000)
        # A(p) = A_90 * ln(1-p)/ln(0.1): shape analytic, normalisation measured.
        shape = np.log(1.0 - levels) / np.log(0.10)
        x = 100.0 * levels

        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        for r in rows:
            color, ls = WEN_STYLES.get(r["config"], ("C5", "-"))
            # if ls == "-":
            # ax.fill_between(
            #     x,
            #     r["area_90_min"] * shape,
            #     r["area_90_max"] * shape,
            #     alpha=0.15,
            #     color=color,
            #     lw=0,
            # )
            ax.plot(
                x,
                r["area_90_median"] * shape,
                lw=2,
                ls=ls,
                color=color,
                label=f"{r['config']} ({r['n_pulsars']} psr, {r['n_anchors']} anch)",
            )

        # Wen's quoted 25-6 range at 90%, for direct comparison.
        lo, hi = WEN_25_6_RANGE_90
        ax.plot(
            [90, 90],
            [lo, hi],
            color="k",
            lw=3,
            solid_capstyle="butt",
            zorder=5,
            label="Wen 25-6, 90% (abstract)",
        )
        for lv in (50, 68, 90):
            ax.axvline(lv, color="0.7", ls=":", lw=1, zorder=0)

        ax.set_yscale("log")
        ax.set_xlabel("credibility level [%]")
        ax.set_ylabel("credible localization area [deg$^2$]")
        ax.set_title(
            "Wen et al. 2026 Fig. 3 analog\n"
            f"SNR={snr:.0f}, {n_psr}-pulsar ocarina subset "
            "(band = min–max over sky)",
            fontsize=10,
        )
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig

    plot_area_vs_level(sweep_rows, float(snr_ui.value), len(pulsar_subset))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 2 analog — credible regions at the Max-sensitivity direction

    Wen et al. Figure 2 overlays, on one sky map, the 90% credible
    localization **regions** of every configuration for a single source
    injected at the array's **maximum-sensitivity** direction ("Max"), with
    the pulsars, five galaxy clusters, and the minimum-sensitivity point
    ("Min") marked.

    Two protocol points inherited from the paper:

    * **Max/Min come from the array itself** — the per-pixel unit-strain
      signal power $Y = (\hat s \mid \hat s)$ of the full coherent array.
      At fixed SNR, the most sensitive direction needs the *weakest* source,
      so Max is where localization is **worst** (Wen Table 1's inverted
      ordering).
    * **Regions here are Fisher ellipses** from `marginal_sky_fisher` — exact
      for the unanchored configurations, **central-lobe lower bounds** for the
      anchored ones (the true anchored posterior is a multi-lobe comb; see
      the validity table in step 6). Honest sampled regions live in the
      enterprise sandbox.
    """)
    return


@app.cell
def _(
    cal_gp,
    cal_marg_config,
    chunk_ui,
    grid,
    jax,
    jnp,
    np,
    pin_orientation,
    pta,
    signal_power_direct,
):
    # Sensitivity map: Y per pixel for the full coherent array (no Fisher, no
    # jacfwd -- much cheaper than the area map).  Max/Min sky directions fall
    # out as argmax/argmin.
    _cg = pin_orientation(cal_gp)

    @jax.jit
    def _all_Y(sky_arr):
        def one(row):
            return signal_power_direct(
                cal_marg_config, _cg, pta.pulsar_params_list, row
            )

        return jax.lax.map(one, sky_arr, batch_size=int(chunk_ui.value))

    sens_Y = np.asarray(_all_Y(grid.sky))
    _imax, _imin = int(np.nanargmax(sens_Y)), int(np.nanargmin(sens_Y))
    sky_max = jnp.asarray(grid.sky[_imax])
    sky_min = jnp.asarray(grid.sky[_imin])
    return sens_Y, sky_max, sky_min


@app.cell(hide_code=True)
def _(mo, np, sens_Y, sky_max, sky_min):
    mo.md(f"""
    Sensitivity span: $\\sqrt{{Y}}$ varies "
        f"{np.sqrt(np.nanmax(sens_Y) / np.nanmin(sens_Y)):.1f}× over the sky.  "
        f"**Max** at (cosθ, φ) = ({float(sky_max[0]):+.3f}, {float(sky_max[1]):.3f}), "
        f"**Min** at ({float(sky_min[0]):+.3f}, {float(sky_min[1]):.3f}).
    """)
    return


@app.cell
def _(np):
    def fisher_ellipse_path(F, center, level=0.9, n=181):
        """Boundary of the `level` credible ellipse of a 2x2 sky Fisher.

        Works in the area-preserving (cos_gwtheta, gwphi) plane -- the same
        coordinates the Fisher is computed in -- then converts to healpy's
        (theta, phi) for projplot.  Returns (theta_path, phi_path).
        """
        cov = np.linalg.pinv(np.asarray(F, dtype=float))
        vals, vecs = np.linalg.eigh(cov)
        vals = np.clip(vals, 0.0, None)
        dchi2 = -2.0 * np.log(1.0 - level)
        t = np.linspace(0.0, 2.0 * np.pi, n)
        circ = np.stack([np.cos(t), np.sin(t)])  # (2, n)
        path = (vecs * np.sqrt(dchi2 * vals)) @ circ  # (2, n)
        ct = np.clip(float(center[0]) + path[0], -1.0, 1.0)
        ph = (float(center[1]) + path[1]) % (2.0 * np.pi)
        return np.arccos(ct), ph

    return (fisher_ellipse_path,)


@app.cell
def _(mo):
    run_fig2_ui = mo.ui.run_button(label="Compute Fig. 2 analog")
    return (run_fig2_ui,)


@app.cell
def _(run_fig2_ui):
    run_fig2_ui
    return


@app.cell
def _(
    WEN_CONFIGS,
    cal_gp,
    cal_marg_config,
    config_setup,
    credible_area_deg2,
    h0_for_snr,
    jnp,
    marginal_sky_fisher,
    mo,
    pin_orientation,
    pta,
    run_fig2_ui,
    signal_power_direct,
    sky_max,
    snr_ui,
):
    mo.stop(
        not run_fig2_ui.value,
        mo.md("*Press **Compute Fig. 2 analog** above (re-marginalizes per config).*"),
    )
    # SIGMA_ANCHOR + nuisance conventions live inside pixel_areas; here we call
    # marginal_sky_fisher directly with the same settings for ONE pixel.
    _NUI = ("cos_inc", "psi", "phase0", "log10_fgw")
    _Y = signal_power_direct(
        cal_marg_config, pin_orientation(cal_gp), pta.pulsar_params_list, sky_max
    )
    _h0 = h0_for_snr(jnp.float64(float(snr_ui.value)), _Y)

    fig2_rows = []
    for _name, _subset, _anchors in WEN_CONFIGS:
        _mcfg, _gpp, _ppl = config_setup(pta, _subset, _anchors)
        _F = marginal_sky_fisher(
            _mcfg,
            _gpp,
            _ppl,
            sky_max,
            h0=_h0,
            global_nuisance=_NUI,
            dist_sigma_kpc=[1.0e-3] * _mcfg.n_pulsars,  # Wen D_err = 1 pc
        )
        fig2_rows.append(
            {
                "config": _name,
                "F": _F,
                "area_90": float(credible_area_deg2(_F, level=0.9)),
            }
        )
    return (fig2_rows,)


@app.cell
def _(
    WEN_STYLES,
    fig2_rows,
    fisher_ellipse_path,
    hp,
    np,
    overlay_pulsars,
    plt,
    pta,
    sens_Y,
    sky_max,
    sky_min,
    snr_ui,
):
    def plot_fig2():
        import matplotlib.lines as mlines

        # Background: log10 sensitivity (sqrt Y), the quantity that defines Max/Min.
        hp.mollview(
            0.5 * np.log10(np.where(sens_Y > 0, sens_Y, np.nan)),
            title=(
                "Wen et al. 2026 Fig. 2 analog — 90% credible regions at Max\n"
                f"SNR={float(snr_ui.value):.0f}; anchored regions are "
                "central-lobe lower bounds"
            ),
            unit=r"$\log_{10}\sqrt{Y}$ (array sensitivity)",
            cmap="plasma",
            # Center the projection on the Max (injection) direction.
            rot=[
                np.degrees(float(sky_max[1])),
                90.0 - np.degrees(np.arccos(float(sky_max[0]))),
            ],
        )
        hp.graticule()

        handles = []
        for row in fig2_rows:
            color, ls = WEN_STYLES.get(row["config"], ("C5", "-"))
            th, ph = fisher_ellipse_path(row["F"], np.asarray(sky_max))
            hp.projplot(th, ph, color=color, ls=ls, lw=2)
            handles.append(
                mlines.Line2D(
                    [],
                    [],
                    color=color,
                    ls=ls,
                    lw=2,
                    label=f"{row['config']} ({row['area_90']:.3g} deg$^2$)",
                )
            )

        # Pulsars (anchors of the 25-6 set drawn as stars).
        _A6 = {
            "J0437-4715",
            "J0030+0451",
            "J1713+0747",
            "J1640+2224",
            "J1744-1134",
            "J1909-3744",
        }
        overlay_pulsars(
            np.asarray(pta.positions),
            np.array([n in _A6 for n in pta.names]),
            star_kwargs=dict(s=140),
            dot_kwargs=dict(s=25, color="0.8", edgecolors="k", zorder=4),
        )
        # Max (injection, red cross) and Min.
        for _sky, _mk, _c, _lbl in (
            (sky_max, "x", "red", "Max (source)"),
            (sky_min, "P", "lime", "Min"),
        ):
            hp.projscatter(
                [np.arccos(float(_sky[0]))],
                [float(_sky[1])],
                marker=_mk,
                s=130,
                color=_c,
                zorder=7,
            )
            handles.append(
                mlines.Line2D([], [], color=_c, marker=_mk, ls="", label=_lbl)
            )
        plt.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.85)
        return plt.gcf()

    plot_fig2()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

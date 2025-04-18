from typing import Literal, TypeVar
import numpy as np
import pandas as pd
from cmap import Color
from himena import (
    StandardType,
    WidgetDataModel,
    Parametric,
    create_model,
    create_dataframe_model,
)
from himena.plugins import register_function, configure_gui, configure_submenu
from himena.standards import plotting as hplt
from himena_builtins.tools.conversions import table_to_dataframe

from sfHMM.base import sfHMMBase
from sfHMM.single_sfhmm import sfHMM1
from sfHMM.multi_sfhmm import sfHMMn
from sfHMM.motor import sfHMM1Motor, sfHMMnMotor
from sfHMM.himena.consts import Types
from sfHMM.utils import gauss_mix

MENUS = ["tools/sfHMM"]
configure_submenu(MENUS, "sfHMM")


@register_function(
    menus=MENUS,
    title="Construct sfHMM",
    command_id="sfHMM:construct-sfHMM",
)
def construct_sfhmm() -> Parametric:
    """Construct sfHMM object."""

    @configure_gui(input_data={"types": [StandardType.TABLE, StandardType.DATAFRAME]})
    def run(
        input_data: WidgetDataModel,
        is_multiple_data: bool = False,
        sg0: float | None = None,
        psf: float | None = None,
        krange: list[int] = (),
        noise_model: Literal["gauss", "poisson"] = "gauss",
        name: str = "",
    ):
        """Construct a sfHMM1 or sfHMMn object."""
        df = _model_to_dataframe(input_data)
        krange = _norm_krange(krange)

        if is_multiple_data:
            sf = sfHMMn(sg0=sg0, psf=psf, krange=krange, model=noise_model, name=name)
            return create_model(
                sf.from_pandas(pd.DataFrame(df)),
                type=Types.SF_HMMN,
                title=f"sfHMMn {name}",
            )
        else:
            val = _get_single_column(df)
            return create_model(
                sfHMM1(
                    val, sg0=sg0, psf=psf, krange=krange, model=noise_model, name=name
                ),
                type=Types.SF_HMM1,
                title=f"sfHMM1 {name}",
            )

    return run


@register_function(
    menus=MENUS,
    title="Construct sfHMM for motor",
    command_id="sfHMM:construct-sfHMM-motor",
)
def construct_sfhmm_motor() -> Parametric:
    """Construct sfHMM object for motor stepping."""

    @configure_gui(
        input_data={"types": [StandardType.TABLE, StandardType.DATAFRAME]},
        max_stride={"min": 1, "max": 10},
    )
    def run(
        input_data: WidgetDataModel,
        is_multiple_data: bool = False,
        sg0: float | None = None,
        psf: float | None = None,
        krange: list[int] = (),
        max_stride: int = 2,
        noise_model: Literal["gauss", "poisson"] = "gauss",
        name: str = "",
    ):
        """Construct a sfHMM1 or sfHMMn object."""
        df = _model_to_dataframe(input_data)
        krange = _norm_krange(krange)

        if is_multiple_data:
            sf = sfHMMnMotor(
                sg0=sg0,
                psf=psf,
                krange=krange,
                model=noise_model,
                name=name,
                max_stride=max_stride,
            )
            return create_model(
                sf.from_pandas(pd.DataFrame(df)),
                type=Types.SF_HMMN_MOTOR,
                title=f"sfHMMnMotor {name}",
            )
        else:
            val = _get_single_column(df)
            return create_model(
                sfHMM1Motor(
                    val,
                    sg0=sg0,
                    psf=psf,
                    krange=krange,
                    model=noise_model,
                    max_stride=max_stride,
                    name=name,
                ),
                type=Types.SF_HMM1_MOTOR,
                title=f"sfHMM1Motor {name}",
            )

    return run


def _model_to_dataframe(input_data: WidgetDataModel) -> pd.DataFrame:
    if input_data.is_subtype_of(StandardType.DATAFRAME):
        df = pd.DataFrame(input_data.value)
    else:
        df = pd.DataFrame(table_to_dataframe(input_data).value)
    return df


def _norm_krange(krange: list[int]) -> tuple[int, int] | None:
    # normalize krange
    if len(krange) == 0:
        krange = None
    elif len(krange) == 1:
        krange = (krange[0], krange[0])
    elif len(krange) != 2:
        raise ValueError("krange must be a list of two integers.")
    return krange


def _get_single_column(df: pd.DataFrame) -> pd.Series:
    cols = df.columns
    if len(cols) != 1:
        raise ValueError(
            "Input data must be one column. If the input data has multiple entries, "
            "check the 'is_multiple_data' option."
        )
    return df[cols[0]]


@register_function(
    menus=MENUS,
    types=[Types.SF_HMM1, Types.SF_HMM1_MOTOR],
    title="Get step finding result",
    command_id="sfHMM:run:get-step-finding",
)
def get_step_finding(model: WidgetDataModel) -> WidgetDataModel:
    """Get step finding result as a dataframe."""
    sf = _cast_sfhmm(model, sfHMM1)
    step_list = np.array(sf.step.step_list, dtype=np.uint32)
    return create_dataframe_model(
        pd.DataFrame(
            {
                "mean": sf.step.mu_list,
                "length": sf.step.len_list,
                "start": step_list[:-1],
                "end": step_list[1:],
            },
        ),
        title=f"Steps of {model.title}",
    )


@register_function(
    menus=MENUS,
    types=[Types.SF_HMM1, Types.SF_HMM1_MOTOR, Types.SF_HMMN, Types.SF_HMMN_MOTOR],
    title="Step finding",
    command_id="sfHMM:run:step-finding",
)
def step_finding(model: WidgetDataModel) -> WidgetDataModel:
    """Run step finding."""
    sf = _cast_sfhmm(model, sfHMMBase)
    sf = sf.clone()
    sf.step_finding()
    return model.with_value(sf)


@register_function(
    menus=MENUS,
    types=[Types.SF_HMM1, Types.SF_HMM1_MOTOR, Types.SF_HMMN, Types.SF_HMMN_MOTOR],
    title="Denoise",
    command_id="sfHMM:run:denoise",
)
def denoise(model: WidgetDataModel) -> WidgetDataModel:
    """Denoise the trajectory using the result of step finding."""
    sf = _cast_sfhmm(model, sfHMMBase)
    sf = sf.clone()
    sf.denoising()
    return model.with_value(sf)


@register_function(
    menus=MENUS,
    types=[Types.SF_HMM1, Types.SF_HMM1_MOTOR, Types.SF_HMMN, Types.SF_HMMN_MOTOR],
    title="GMM clustering",
    command_id="sfHMM:run:gmmfit",
)
def gmmfit(model: WidgetDataModel) -> Parametric:
    """Run GMM clustering on the sfHMM model."""

    @configure_gui(
        n_init={"min": 1, "label": "Number of initializations"},
        random_state={"min": 0, "max": 1000000, "label": "Random seed"},
    )
    def run(
        method: Literal["AIC", "BIC", "Dirichlet"] = "BIC",
        n_init: int = 10,
        random_state: int = 0,
    ):
        sf = _cast_sfhmm(model, sfHMMBase)
        sf = sf.clone()
        sf.gmmfit(method=method.lower(), n_init=n_init, random_state=random_state)
        return model.with_value(sf)

    return run


@register_function(
    menus=MENUS,
    types=[Types.SF_HMM1, Types.SF_HMM1_MOTOR, Types.SF_HMMN, Types.SF_HMMN_MOTOR],
    title="HMM and find Viterbi path",
    command_id="sfHMM:run:hmmfit",
)
def hmmfit(model: WidgetDataModel) -> WidgetDataModel:
    """Initialize HMM parameters, optimize, and find Viterbi path."""
    sf = _cast_sfhmm(model, sfHMMBase)
    sf = sf.clone()
    sf.hmmfit()
    return model.with_value(sf)


@register_function(
    menus=MENUS,
    types=[Types.SF_HMM1, Types.SF_HMM1_MOTOR, Types.SF_HMMN, Types.SF_HMMN_MOTOR],
    title="Run all",
    command_id="sfHMM:run-all:run-all",
)
def run_all(model: WidgetDataModel) -> WidgetDataModel:
    """Run all the sfHMM workflow."""
    sf = _cast_sfhmm(model, sfHMMBase)
    sf = sf.clone()
    return model.with_value(sf.run_all(plot=False))


@register_function(
    menus=MENUS,
    types=[Types.SF_HMMN, Types.SF_HMMN_MOTOR],
    title="Run all separately",
    command_id="sfHMM:run-all:run-all-separately",
)
def run_all_separately(model: WidgetDataModel) -> WidgetDataModel:
    """Run all the sfHMM workflow for each item."""
    sf = _cast_sfhmm(model, sfHMMn)
    sf = sf.clone()
    return model.with_value(sf.run_all_separately(plot_every=-1))


@register_function(
    menus=MENUS,
    types=[Types.SF_HMM1, Types.SF_HMM1_MOTOR],
    title="Plot raw data",
    command_id="sfHMM:plot:raw",
)
def plot_raw(model: WidgetDataModel) -> WidgetDataModel:
    """Plot the raw data."""
    sf = _cast_sfhmm(model, sfHMM1)
    fig = hplt.figure()
    x = np.arange(sf.data_raw.size)
    y0 = sf.data_raw
    fig.plot(x, y0, color=sf.colors["raw data"])
    return create_model(
        fig,
        type=StandardType.PLOT,
        title=f"Raw data of {model.title}",
    )


@register_function(
    menus=MENUS,
    types=[Types.SF_HMM1, Types.SF_HMM1_MOTOR],
    title="Plot step fit",
    command_id="sfHMM:plot:step-fit",
)
def plot_step_fit(model: WidgetDataModel) -> WidgetDataModel:
    """Plot the raw data and the step finding result."""
    sf = _cast_sfhmm(model, sfHMM1)
    fig = hplt.figure()
    _plot_raw_and_step(sf, fig)
    return create_model(
        fig,
        type=StandardType.PLOT,
        title=f"Step fit of {model.title}",
    )


@register_function(
    menus=MENUS,
    types=[Types.SF_HMMN, Types.SF_HMMN_MOTOR],
    title="Plot step fit",
    command_id="sfHMM:plot:step-fit-multi",
)
def plot_step_fits(model: WidgetDataModel) -> WidgetDataModel:
    """Plot the raw data and the step finding result for each trajectory."""
    msf = _cast_sfhmm(model, sfHMMn)
    fig = hplt.figure_stack(len(msf))
    for i, sf in enumerate(msf):
        _plot_raw_and_step(sf, fig[i])
    return create_model(
        fig,
        type=StandardType.PLOT_STACK,
        title=f"Step fit of {model.title}",
    )


@register_function(
    menus=MENUS,
    types=[Types.SF_HMM1, Types.SF_HMM1_MOTOR, Types.SF_HMMN, Types.SF_HMMN_MOTOR],
    title="Plot GMM",
    command_id="sfHMM:plot:gmm",
)
def plot_gmm(model: WidgetDataModel) -> WidgetDataModel:
    """Plot histogram and GMM fitting result."""
    sf = _cast_sfhmm(model, sfHMMBase)
    fig = hplt.figure()
    _plot_gmm(sf, fig)
    return create_model(
        fig,
        type=StandardType.PLOT,
        title=f"GMM of {model.title}",
    )


@register_function(
    menus=MENUS,
    types=[Types.SF_HMMN, Types.SF_HMMN_MOTOR],
    title="Plot each GMM",
    command_id="sfHMM:plot:gmm-multi",
)
def plot_gmm_multi(model: WidgetDataModel) -> WidgetDataModel:
    """Plot histogram for each trajectory."""
    msf = _cast_sfhmm(model, sfHMMn)
    fig = hplt.figure_stack(len(msf))
    for i, sf in enumerate(msf):
        _plot_gmm(sf, fig[i])
        _plot_gmm_fit(msf, fig[i])
    return create_model(
        fig,
        type=StandardType.PLOT_STACK,
        title=f"GMMs of {model.title}",
    )


@register_function(
    menus=MENUS,
    types=[Types.SF_HMM1, Types.SF_HMM1_MOTOR],
    title="Plot Viterbi path",
    command_id="sfHMM:plot:viterbi-path",
)
def plot_viterbi_path(model: WidgetDataModel) -> WidgetDataModel:
    """Plot the raw data and the Viterbi path."""
    sf = _cast_sfhmm(model, sfHMM1)
    fig = hplt.figure()
    _plot_raw_and_viterbi(sf, fig)
    return create_model(
        fig,
        type=StandardType.PLOT,
        title=f"Viterbi path of {model.title}",
    )


@register_function(
    menus=MENUS,
    types=[Types.SF_HMMN, Types.SF_HMMN_MOTOR],
    title="Plot Viterbi path",
    command_id="sfHMM:plot:viterbi-path-multi",
)
def plot_viterbi_paths(model: WidgetDataModel) -> WidgetDataModel:
    """Plot the raw data and the Viterbi path for each trajectory."""
    msf = _cast_sfhmm(model, sfHMMn)
    fig = hplt.figure_stack(len(msf))
    for i, sf in enumerate(msf):
        _plot_raw_and_viterbi(sf, fig[i])
    return create_model(
        fig,
        type=StandardType.PLOT_STACK,
        title=f"Viterbi path of {model.title}",
    )


_T = TypeVar("_T", bound=sfHMMBase)


def _cast_sfhmm(model: WidgetDataModel, typ: type[_T]) -> _T:
    if not isinstance(sf := model.value, typ):
        raise ValueError("Input model must be sfHMM object.")
    return sf


def _with_alpha(color, alpha):
    r, g, b, _ = Color(color)
    return Color((r, g, b, alpha))


def _plot_raw_and_step(sf: sfHMM1, axes: hplt.SingleAxes) -> None:
    x = np.arange(sf.data_raw.size)
    y0 = sf.data_raw
    y1 = sf.step.fit
    axes.plot(x, y0, color=sf.colors["raw data"])
    axes.plot(x, y1, color=sf.colors["step finding"])


def _plot_raw_and_viterbi(sf: sfHMM1, axes: hplt.SingleAxes) -> None:
    x = np.arange(sf.data_raw.size)
    y0 = sf.data_raw
    y1 = sf.viterbi
    axes.plot(x, y0, color=sf.colors["raw data"])
    axes.plot(x, y1, color=sf.colors["Viterbi path"])


def _plot_gmm(sf: sfHMM1, axes: hplt.SingleAxes) -> None:
    data_raw = sf.data_raw
    data_fil = sf.data_fil
    n_bin = min(int(np.sqrt(data_raw.size * 2)), 256)
    if sf.gmm_opt is not None:
        _plot_gmm_fit(sf, axes)
    axes.hist(
        data_raw,
        bins=n_bin,
        color=_with_alpha(sf.colors["raw data"], 0.7),
        stat="density",
        width=0,
    )
    axes.hist(
        data_fil,
        bins=n_bin,
        edge_color=_with_alpha(sf.colors["denoised"], 0.7),
        face_color="#00000000",
        stat="density",
        width=1,
    )


def _plot_gmm_fit(sf: sfHMMBase, axes: hplt.SingleAxes) -> None:
    fit_x = np.linspace(sf.ylim[0], sf.ylim[1], 256)
    fit_y = gauss_mix(fit_x, sf.gmm_opt)
    axes.plot(fit_x, fit_y, color=sf.colors["GMM"], style="-.")
    peak_x = sf.gmm_opt.means_.ravel()
    peak_y = gauss_mix(peak_x, sf.gmm_opt)
    peak_y += np.max(peak_y) * 0.1
    axes.scatter(
        peak_x,
        peak_y,
        symbol="v",
        color=sf.colors["GMM"],
        face_color=sf.colors["GMM marker"],
        size=10,
    )

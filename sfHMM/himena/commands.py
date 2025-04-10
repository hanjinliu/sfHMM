from typing import Literal, TypeVar
import numpy as np
import pandas as pd
from cmap import Color
from himena import StandardType, WidgetDataModel, Parametric, create_model
from himena.plugins import register_function, configure_gui
from himena.standards import plotting as hplt
from himena_builtins.tools.conversions import table_to_dataframe

from sfHMM.base import sfHMMBase
from sfHMM.single_sfhmm import sfHMM1
from sfHMM.multi_sfhmm import sfHMMn
from sfHMM.himena.consts import Types
from sfHMM.utils import gauss_mix

MENUS = ["sfHMM"]


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
        if input_data.is_subtype_of(StandardType.DATAFRAME):
            df = pd.DataFrame(input_data.value)
        else:
            df = pd.DataFrame(table_to_dataframe(input_data).value)

        # normalize krange
        if len(krange) == 0:
            krange = None
        elif len(krange) == 1:
            krange = (krange[0], krange[0])
        elif len(krange) != 2:
            raise ValueError("krange must be a list of two integers.")

        if is_multiple_data:
            sf = sfHMMn(sg0=sg0, psf=psf, krange=krange, model=noise_model, name=name)
            return create_model(
                sf.from_pandas(pd.DataFrame(df)),
                type=Types.SF_HMMN,
                title=f"sfHMMn {name}",
            )
        else:
            cols = df.columns
            if len(cols) != 1:
                raise ValueError("Input data must be one column.")
            val = df[cols[0]]
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
    types=[Types.SF_HMM1, Types.SF_HMM1_MOTOR, Types.SF_HMMN, Types.SF_HMMN_MOTOR],
    title="Run all",
    command_id="sfHMM:run-all",
)
def run_all(model: WidgetDataModel) -> WidgetDataModel:
    sf = _cast_sfhmm(model, sfHMMBase)
    sf = sf.clone()
    return model.with_value(sf.run_all(plot=False))


@register_function(
    menus=MENUS,
    types=[Types.SF_HMM1, Types.SF_HMM1_MOTOR],
    title="Plot step fit",
    command_id="sfHMM:plot-step-fit",
)
def plot_step_fit(model: WidgetDataModel) -> WidgetDataModel:
    sf = _cast_sfhmm(model, sfHMM1)
    x = np.arange(sf.data_raw.size)
    y0 = sf.data_raw
    y1 = sf.step.fit
    fig = hplt.figure()
    fig.plot(x, y0, color=sf.colors["raw data"])
    fig.plot(x, y1, color=sf.colors["step finding"])
    return create_model(
        fig,
        type=StandardType.PLOT,
        title=f"Step fit of {model.title}",
    )


@register_function(
    menus=MENUS,
    types=[Types.SF_HMM1, Types.SF_HMM1_MOTOR, Types.SF_HMMN, Types.SF_HMMN_MOTOR],
    title="Plot GMM",
    command_id="sfHMM:plot-gmm",
)
def plot_gmm(model: WidgetDataModel) -> WidgetDataModel:
    sf = _cast_sfhmm(model, sfHMMBase)
    data_raw = sf.data_raw
    data_fil = sf.data_fil
    fig = hplt.figure()
    n_bin = min(int(np.sqrt(data_raw.size * 2)), 256)
    fit_x = np.linspace(sf.ylim[0], sf.ylim[1], 256)
    fit_y = gauss_mix(fit_x, sf.gmm_opt)
    peak_x = sf.gmm_opt.means_.ravel()
    peak_y = gauss_mix(peak_x, sf.gmm_opt)
    peak_y += np.max(peak_y) * 0.1
    fig.plot(fit_x, fit_y, color=sf.colors["GMM"], style="-.")
    fig.scatter(
        peak_x,
        peak_y,
        symbol="v",
        color=sf.colors["GMM"],
        face_color=sf.colors["GMM marker"],
        size=10,
    )
    fig.hist(
        data_raw,
        bins=n_bin,
        color=_with_alpha(sf.colors["raw data"], 0.7),
        stat="density",
        width=0,
    )
    fig.hist(
        data_fil,
        bins=n_bin,
        edge_color=_with_alpha(sf.colors["denoised"], 0.7),
        face_color="#00000000",
        stat="density",
        width=1,
    )
    return create_model(
        fig,
        type=StandardType.PLOT,
        title=f"GMM of {model.title}",
    )


@register_function(
    menus=MENUS,
    types=[Types.SF_HMM1, Types.SF_HMM1_MOTOR],
    title="Plot Viterbi path",
    command_id="sfHMM:plot-viterbi-path",
)
def plot_viterbi_path(model: WidgetDataModel) -> WidgetDataModel:
    sf = _cast_sfhmm(model, sfHMM1)
    x = np.arange(sf.data_raw.size)
    y0 = sf.data_raw
    y1 = sf.viterbi
    fig = hplt.figure()
    fig.plot(x, y0, color=sf.colors["raw data"])
    fig.plot(x, y1, color=sf.colors["Viterbi path"])
    return create_model(
        fig,
        type=StandardType.PLOT,
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

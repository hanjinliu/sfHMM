from __future__ import annotations

from himena import WidgetDataModel
from himena.plugins import register_widget_class, validate_protocol
from qtpy import QtWidgets as QtW, QtCore
from sfHMM import sfHMM1, sfHMMn
from sfHMM.base import sfHMMBase
from sfHMM.motor import sfHMM1Motor, sfHMMnMotor
from sfHMM.himena.consts import Types


class QsfHMM1Widget(QtW.QWidget):
    def __init__(self):
        super().__init__()
        layout = QtW.QVBoxLayout(self)
        self._step_finding_checkbox = _checkbox("Step finding")
        self._denoise_checkbox = _checkbox("Denoise")
        self._gmm_checkbox = _checkbox("GMM")
        self._hmm_checkbox = _checkbox("HMM")
        self._sfhmm_object: sfHMM1 | None = None
        self._sg0_value = QtW.QLabel()
        self._psf_value = QtW.QLabel()
        self._krange_value = QtW.QLabel()
        self._noise_model_value = QtW.QLabel()

        layout.addLayout(_labeled("sg0 = ", self._sg0_value))
        layout.addLayout(_labeled("psf = ", self._psf_value))
        layout.addLayout(_labeled("krange = ", self._krange_value))
        layout.addLayout(_labeled("noise model = ", self._noise_model_value))
        layout.addWidget(self._step_finding_checkbox)
        layout.addWidget(self._denoise_checkbox)
        layout.addWidget(self._gmm_checkbox)
        layout.addWidget(self._hmm_checkbox)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

    @validate_protocol
    def update_model(self, model: WidgetDataModel):
        sf = model.value
        if not isinstance(sf, sfHMM1):
            raise ValueError(f"Expected sfHMM1, got {type(sf)}")
        self._sfhmm_object = sf
        self._update_labels(sf)
        self._step_finding_checkbox.setChecked(sf.step is not None)
        self._denoise_checkbox.setChecked(sf.data_fil is not None)
        self._gmm_checkbox.setChecked(hasattr(sf, "gmm"))
        self._hmm_checkbox.setChecked(sf.viterbi is not None)

    def _update_labels(self, sf: sfHMMBase):
        if sf.sg0 < 0:
            sg0_value = "<To be determined>"
        else:
            sg0_value = f"{sf.sg0:.2e}"
        self._sg0_value.setText(sg0_value)
        if sf.psf < 0:
            psf_value = "<To be determined>"
        elif sf.psf < 0.001:
            psf_value = f"{sf.psf:.2e}"
        else:
            psf_value = f"{sf.psf:.3f}"
        self._psf_value.setText(psf_value)
        if sf.krange is None:
            self._krange_value.setText("<To be determined>")
        else:
            self._krange_value.setText(f"min: {sf.krange[0]}, max: {sf.krange[1]}")
        self._noise_model_value.setText(sf.model)

    @validate_protocol
    def to_model(self) -> WidgetDataModel:
        if self._sfhmm_object is None:
            raise ValueError("No sfHMM1 object to convert to model.")
        return WidgetDataModel(
            value=self._sfhmm_object,
            type=self.model_type(),
        )

    @validate_protocol
    def model_type(self) -> str:
        if isinstance(self._sfhmm_object, sfHMM1Motor):
            return Types.SF_HMM1_MOTOR
        else:
            return Types.SF_HMM1

    @validate_protocol
    def size_hint(self) -> tuple[int, int]:
        return 280, 240


class QsfHMMnWidget(QsfHMM1Widget):
    @validate_protocol
    def update_model(self, model: WidgetDataModel):
        sf = model.value
        if not isinstance(sf, sfHMMn):
            raise ValueError(f"Expected sfHMMn, got {type(sf)}")
        self._sfhmm_object = sf
        self._update_labels(sf)
        self._step_finding_checkbox.setChecked(sf._step_finding_done)
        self._denoise_checkbox.setChecked(sf._denoise_done)
        self._gmm_checkbox.setChecked(hasattr(sf, "gmm"))
        self._hmm_checkbox.setChecked(sf._hmm_done)

    @validate_protocol
    def to_model(self) -> WidgetDataModel:
        if self._sfhmm_object is None:
            raise ValueError("No sfHMMn object to convert to model.")
        return WidgetDataModel(
            value=self._sfhmm_object,
            type=self.model_type(),
        )

    @validate_protocol
    def model_type(self) -> str:
        if isinstance(self._sfhmm_object, sfHMMnMotor):
            return Types.SF_HMMN_MOTOR
        else:
            return Types.SF_HMMN


def _checkbox(text: str) -> QtW.QCheckBox:
    """Create a checkbox with the given text."""
    checkbox = QtW.QCheckBox(text)
    checkbox.setChecked(False)
    checkbox.setEnabled(False)
    return checkbox


def _labeled(label: str, widget: QtW.QWidget) -> QtW.QHBoxLayout:
    """Create a labeled widget."""
    layout = QtW.QHBoxLayout()
    label_widget = QtW.QLabel(label)
    layout.addWidget(label_widget)
    layout.addWidget(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
    return layout


register_widget_class(Types.SF_HMM1, QsfHMM1Widget)
register_widget_class(Types.SF_HMM1_MOTOR, QsfHMM1Widget)
register_widget_class(Types.SF_HMMN, QsfHMMnWidget)
register_widget_class(Types.SF_HMMN_MOTOR, QsfHMMnWidget)

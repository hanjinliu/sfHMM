from __future__ import annotations
from typing import Iterable, Union
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from qtpy.QtWidgets import (
    QGridLayout,
    QWidget,
    QCheckBox,
    QMainWindow,
    QDockWidget,
    QSpinBox,
    QAction,
    QApplication,
)
from qtpy.QtCore import Qt
from .canvas import EventedCanvas

DictLike = Union[dict, pd.DataFrame]

APPLICATION = None


def type_check(data) -> dict[str, np.ndarray]:
    if isinstance(data, pd.DataFrame):
        data = data.to_dict()
    elif isinstance(data, dict):
        data = {k: np.asarray(v) for k, v in data.items()}
    else:
        raise TypeError(f"data must be dict or pd.DataFrame, got {type(data)}")
    return data


class TrajectoryViewer(QMainWindow):
    """
    Show interactive multi-trajectory plot.

    >>> data = {"traj-1": np.array([...]),
                "traj-2": np.array([...])
                }
    >>> viewer = TrajectoryViewer(data)
    >>> viewer.show()
    """

    def __init__(
        self,
        data: DictLike | list[DictLike] | tuple[DictLike],
        styles: dict = None,
        colors: dict = None,
    ):
        app = get_app()  # noqa: F841
        super().__init__(parent=None)
        # check input
        if isinstance(data, (tuple, list)):
            data = list(map(type_check, data))
        else:
            data = [type_check(data)]

        self.data = data
        self.checkbox = Controller(self, data[0].keys())

        self.addDockWidget(Qt.BottomDockWidgetArea, self.checkbox)
        self.checkbox.visibilityChanged.connect(self.checkboxes_visibility_changed)

        self.setUnifiedTitleAndToolBarOnMac(True)

        self.plot_style = styles.copy() if styles is not None else {}
        self.plot_style["legend.frameon"] = True
        self.plot_color = colors.copy() if colors is not None else {}
        self.lines: dict[str, plt.Line2D] = dict()
        self.current_index = 0
        # prepare a figure canvas
        # To block additional figure being opened, matplotlib backend must be changed temporary.
        backend = mpl.get_backend()
        mpl.use("Agg")
        with plt.style.context(self.plot_style):
            self.fig = plt.figure()
            self.ax: Axes = self.fig.add_subplot(111)
            for key, data in self.current_data.items():
                self.lines[key] = self.ax.plot(
                    data, color=self.plot_color[key], label=key
                )[0]

        mpl.use(backend)

        self.setCentralWidget(EventedCanvas(self.fig))
        self.fig.tight_layout()
        self.fig.canvas.draw()

        self.menu = self.menuBar().addMenu("&Menu")

        show_legend = QAction("Show legend", parent=self, checkable=True, checked=False)
        show_legend.triggered.connect(self.change_legend_visibility)
        self.menu.addAction(show_legend)

        self.show_controller = QAction(
            "Show checkboxes", parent=self, checkable=True, checked=True
        )
        self.show_controller.triggered.connect(self.change_checkboxes_visibility)
        self.menu.addAction(self.show_controller)

        close = QAction("Close", parent=self)
        close.triggered.connect(self.close)
        self.menu.addAction(close)

    @property
    def current_data(self) -> dict[str, np.ndarray]:
        """
        Return the dataset dict that should be displayed. Independent of whether they are
        visible or not.
        """
        return self.data[self.current_index]

    def update_plot(self):
        checked = self.checkbox.isEachChecked()
        for i, key in enumerate(self.current_data.keys()):
            if checked[i]:
                self.lines[key].set_color(self.plot_color[key])
            else:
                self.lines[key].set_color([0, 0, 0, 0])

        self.fig.tight_layout()
        self.fig.canvas.draw()

    def change_data(self):
        for key, value in self.current_data.items():
            self.lines[key].set_xdata(np.arange(len(value)))
            self.lines[key].set_ydata(value)

        self.fig.tight_layout()
        self.fig.canvas.draw()

    def change_checkboxes_visibility(self):
        if self.show_controller.isChecked():
            self.checkbox.show()
        else:
            self.checkbox.hide()

    def checkboxes_visibility_changed(self):
        self.show_controller.setChecked(self.checkbox.isVisible())
        return None

    def change_legend_visibility(self):
        legend = self.ax.get_legend()
        if legend:
            legend.remove()
        else:
            with plt.style.context(self.plot_style):
                # temporary set all the lines visible.
                # Without this code, lines will not be displayed in legend.
                for key in self.current_data.keys():
                    self.lines[key].set_color(self.plot_color[key])
                self.ax.legend()

        self.update_plot()

    def show(self):
        super().show()
        self.raise_()
        self.activateWindow()
        return None

    def closeEvent(self, event):
        """
        When QMainWindow is closing, its dock widget will not automatically close when it is
        floating. We have to catch the close event and disable floating before actually close
        the main window.
        """
        for dock in self.findChildren(Controller):
            dock.setFloating(False)

        event.accept()


class Controller(QDockWidget):
    def __init__(self, parent: TrajectoryViewer, names: Iterable[str]):
        super().__init__("Trajectories", parent=parent)

        self.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.RightDockWidgetArea)
        self.qcheckbox_list: list[QCheckBox] = []

        layout = QGridLayout()
        layout.setAlignment(Qt.AlignCenter)
        self.central_widget = QWidget(self)
        self.central_widget.setLayout(layout)

        self._set_checkboxes(names)
        self._set_spinbox()
        self.setWidget(self.central_widget)

    def parent(self) -> TrajectoryViewer:  # for IDE
        return super().parent()

    def isEachChecked(self) -> list[bool]:
        return [widget.isChecked() for widget in self.qcheckbox_list]

    def _set_checkboxes(self, names):
        for name in names:
            checkbox = QCheckBox(self)
            checkbox.setText(name)
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self.parent().update_plot)
            self.qcheckbox_list.append(checkbox)
            self.central_widget.layout().addWidget(checkbox)

    def _set_spinbox(self):
        if len(self.parent().data) == 1:
            return None
        self.spinbox = QSpinBox(parent=self)
        self.spinbox.setRange(0, len(self.parent().data) - 1)

        @self.spinbox.valueChanged.connect
        def _():
            self.parent().current_index = int(self.spinbox.value())
            self.parent().change_data()

        self.central_widget.layout().addWidget(self.spinbox)


def gui_qt():
    try:
        from IPython import get_ipython
    except ImportError:
        return None

    shell = get_ipython()

    if shell and shell.active_eventloop != "qt":
        shell.enable_gui("qt")
    return None


def get_app():
    gui_qt()
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    global APPLICATION
    APPLICATION = app
    return app

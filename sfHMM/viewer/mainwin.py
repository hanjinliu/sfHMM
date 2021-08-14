from __future__ import annotations
from typing import Iterable
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from qtpy.QtWidgets import (QGridLayout, QWidget, QCheckBox, QMainWindow, QDockWidget, QApplication)
from qtpy.QtCore import Qt
from .canvas import EventedCanvas

class TrajectoryViewer(QMainWindow):
    def __init__(self, data:dict|pd.DataFrame|Iterable, styles:dict, colors:dict):
        super().__init__(parent=None)
        
        if isinstance(data, pd.DataFrame):
            data = data.to_dict()
        elif isinstance(data, dict):
            data = {k: np.asarray(v) for k, v in data.items()}
        else:
            data = {f"Data {i}": np.asarray(d) for i, d in enumerate(data)}
        
        self.data = data
        self.checkbox = CompositCheckBoxes(self, data.keys())        
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.checkbox)
        self.setUnifiedTitleAndToolBarOnMac(True)
        self.setWindowTitle("sfHMM plot")
        self.plot_style = styles
        self.plot_color = colors
        self.lines = dict()
        # prepare a figure canvas
        # To block additional figure opened, matplotlib backend must be changed temporary.
        backend = mpl.get_backend()
        mpl.use("Agg")
        with plt.style.context(self.plot_style):
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
        mpl.use(backend)
        
        self.setCentralWidget(EventedCanvas(self.fig))
        
        # First plot
        for key, data in self.data.items():
            self.lines[key] = self.ax.plot(data, color=self.plot_color[key])[0]
        self.fig.tight_layout()
        self.fig.canvas.draw()        

    def update_plot(self):
        checked = self.checkbox.isChecked()
        for i, key in enumerate(self.data.keys()):
            if checked[i]:
                self.lines[key].set_color(self.plot_color[key])
            else:
                self.lines[key].set_color([0,0,0,0])
                
        self.fig.canvas.draw()
            
    def close(self):
        self.checkbox.close()
        del self.checkbox
        super().close()

class CompositCheckBoxes(QDockWidget):
    def __init__(self, parent:TrajectoryViewer, names:Iterable[str]):
        super().__init__("Trajectories", parent=parent)
        self.widget_list:list[QCheckBox] = []
        self.checkboxeswidget = QWidget(self)
        layout = QGridLayout()
        layout.setAlignment(Qt.AlignCenter)
        self.checkboxeswidget.setLayout(layout)
        
        for name in names:
            checkbox = QCheckBox(self)
            checkbox.setText(name)
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(parent.update_plot)
            self.widget_list.append(checkbox)
            self.checkboxeswidget.layout().addWidget(checkbox)
        
        self.setWidget(self.checkboxeswidget)
    
    def isChecked(self):
        return [wid.isChecked() for wid in self.widget_list]

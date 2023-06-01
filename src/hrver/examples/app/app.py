import os
import sys

import numpy as np

from hrver.utils import loadModel
from hrver.features import getPhysioRecordFeatures, getRawEcgFeatures
from hrver.constants import REFERENCE_MEAN_MS, REFERENCE_STD_MS, DetectorName, DetectorNames

from PyQt5.QtWidgets import QApplication, QWidget, \
    QFileDialog, QVBoxLayout, QHBoxLayout, QFormLayout, \
    QLineEdit, QPushButton, QLabel, QCheckBox, QRadioButton

from PyQt5.QtGui import QIntValidator, QDoubleValidator

MODELS_FOLDER = '..\\..\\..\\..\\models'

SCALER_FILE = 'zscaler.model'
PCA_FILE = 'pca-12.model'
MODEL_FILE = 'random-forest-scaled-pca12.model'


WINDOW_X = 300
WINDOW_Y = 300
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 100


class WidgetsContainer:
    def __init__(self):
        self.__disabled = True
        self.__widgets: list[QWidget] = []

    def add(self, widget: QWidget):
        widget.setDisabled(self.__disabled)
        self.__widgets.append(widget)

    def setDisabled(self, disabled: bool):
        self.__disabled = disabled
        for widget in self.__widgets:
            widget.setDisabled(disabled)


class MainWindow(QWidget):

    def __init__(self):
        super().__init__()
        
        self.__scaler = loadModel(os.path.join(MODELS_FOLDER, SCALER_FILE))
        self.__pca = loadModel(os.path.join(MODELS_FOLDER, PCA_FILE))
        self.__model = loadModel(os.path.join(MODELS_FOLDER, MODEL_FILE))

        self.setGeometry(WINDOW_X, WINDOW_Y, WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setWindowTitle('Heart diseases prediction')

        #region Main layout
        self.outer_layout = QVBoxLayout()

            #region Select file layout
        self.select_file_layout = QHBoxLayout()
        
        self.selected_file_line_edit = QLineEdit("")
        self.selected_file_line_edit.setReadOnly(True)
        self.select_file_layout.addWidget(self.selected_file_line_edit)

        self.select_file_button = QPushButton("Select file")
        self.select_file_layout.addWidget(self.select_file_button)

        self.outer_layout.addLayout(self.select_file_layout)

        self.select_file_result_label = QLabel()
        self.outer_layout.addWidget(self.select_file_result_label)
            #endregion
        
        self.file_selected_widgets = WidgetsContainer()
            
            #region Settings
        self.settings_layout = QVBoxLayout()
        
                #region Sampling rate
        self.sampling_rate_layout = QFormLayout()
        self.sampling_rate_line_edit = QLineEdit()
        self.sampling_rate_line_edit.setValidator(QIntValidator())
        self.file_selected_widgets.add(self.sampling_rate_line_edit)
    
        self.sampling_rate_layout.addRow("Sampling rate:", self.sampling_rate_line_edit)
        self.settings_layout.addLayout(self.sampling_rate_layout)
                #endregion
    
                #region Use smoothing
        self.use_smoothing_checkbox = QCheckBox("Use smoothing")
        self.file_selected_widgets.add(self.use_smoothing_checkbox)
        self.settings_layout.addWidget(self.use_smoothing_checkbox)
                    
                    #region Use smoothing settings
        self.smoothing_settings_layout = QFormLayout()
        self.smoothing_settings = WidgetsContainer()

        self.smooth_reference_mean_lineedit = QLineEdit(str(REFERENCE_MEAN_MS))
        self.smooth_reference_mean_lineedit.setValidator(QDoubleValidator())
        self.smoothing_settings_layout.addRow("Smoothing reference mean (ms):", self.smooth_reference_mean_lineedit)
        self.smoothing_settings.add(self.smooth_reference_mean_lineedit)

        self.smooth_reference_std_lineedit = QLineEdit(str(REFERENCE_STD_MS))
        self.smooth_reference_std_lineedit.setValidator(QDoubleValidator())
        self.smoothing_settings_layout.addRow("Smoothing reference standard deviation (ms):", self.smooth_reference_std_lineedit)
        self.smoothing_settings.add(self.smooth_reference_std_lineedit)

        self.smooth_iterations = QLineEdit("10")
        self.smooth_iterations.setValidator(QIntValidator())
        self.smoothing_settings_layout.addRow("Smoothing iterations:", self.smooth_iterations)
        self.smoothing_settings.add(self.smooth_iterations)

        self.smooth_window = QLineEdit("3")
        self.smooth_window.setValidator(QIntValidator())
        self.smoothing_settings_layout.addRow("Smoothing window:", self.smooth_window)
        self.smoothing_settings.add(self.smooth_window)

        self.smoothing_settings.setDisabled(True)
        self.settings_layout.addLayout(self.smoothing_settings_layout)
                    #endregion
                
                #endregion

                #region Split record
        self.split_record_checkbox = QCheckBox("Split record into chunks")
        self.file_selected_widgets.add(self.split_record_checkbox)
        self.settings_layout.addWidget(self.split_record_checkbox)
        
        self.split_record_settings = WidgetsContainer()
                   
                    #region Split record settings
        self.split_record_settings_layout = QFormLayout()
        
        self.split_record_legnth = QLineEdit("90")
        self.split_record_legnth.setValidator(QDoubleValidator())
        self.split_record_settings_layout.addRow("Chunks length (s):", self.split_record_legnth)
        self.split_record_settings.add(self.split_record_legnth)

        self.split_record_settings.setDisabled(True)
        self.settings_layout.addLayout(self.split_record_settings_layout)
                    #endregion

                #endregion

                #region Detector
        self.detector_label = QLabel("R-peak detector:")
        self.settings_layout.addWidget(self.detector_label)

        self.detector_radio_buttons = []
        for detector_name in DetectorNames.ALL:
            detector_radio_button = QRadioButton(detector_name)
            self.file_selected_widgets.add(detector_radio_button)
                #endregion
        
        self.outer_layout.addLayout(self.settings_layout)

            #endregion
        
            #region Results
        self.results_layout = QVBoxLayout()

        self.predict_button = QPushButton("Predict")
        self.results_layout.addWidget(self.predict_button)
        self.file_selected_widgets.add(self.predict_button)

        self.results_label = QLabel()
        self.results_layout.addWidget(self.results_label)
        self.file_selected_widgets.add(self.results_label)

        self.outer_layout.addLayout(self.results_layout)
            #endregion

        self.file_selected_widgets.setDisabled(True)
        
        #endregion

        self.setLayout(self.outer_layout)

        self.show()
    
    def openFileDialog(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, 'Select ECG file', "", "All Files (*); WFDB Files (*.dat);")
        if file_name:
            self.__processData(file_name)

    # def openFileDialog(self):
    #     options = QFileDialog.Options()
    #     file_name, _ = QFileDialog.getOpenFileName(self, 'Select raw ECG file', "", "All Files (*);", options=options)
    #     if file_name:
    #         self.__processData(file_name)
        
        
    # def __processData(self, file_name):
    #     ok = False
    #     sampling_rate = 0
    #     msg = "Sampling rate"
    #     while not ok:
    #         sampling_rate, ok = QInputDialog().getInt(self, msg, "Enter sampling rate:", QLineEdit.Normal)
    #         msg = "Sampling rate: wrong sampling rate entered"

    #     raw_ecg = np.loadtxt(file_name)
    #     if np.any(np.isnan(raw_ecg)):
    #         self.results_label.setText('Data contains NaN elements')
    #         return
    #     features = None
    #     try:
    #         features = utils.getAllFeatures(raw_ecg, sampling_rate)
    #     except Exception as e:
    #         self.results_label.setText('Got error getting features')
    #         return
    #     features = utils.preprocessFeatures(features, self.__scaler, self.__pca)
    #     answer = self.__model.predict(features)[0]
    #     self.results_label.setText(answer)
        
        
if __name__=='__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
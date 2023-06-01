import os
import sys
import pathlib

import numpy as np

from collections import Counter

from hrver.api import PhysioRecord
from hrver.utils import loadModel
from hrver.features import getPhysioRecordFeatures, getRawEcgFeatures
from hrver.constants import REFERENCE_MEAN_MS, REFERENCE_STD_MS, DetectorName, DetectorNames

from PyQt5.QtCore import QThread

from PyQt5.QtWidgets import QApplication, QWidget, \
    QFileDialog, QVBoxLayout, QHBoxLayout, QFormLayout, \
    QLineEdit, QPushButton, QLabel, QCheckBox, QRadioButton, \
    QMessageBox

from PyQt5.QtGui import QValidator, QIntValidator, QDoubleValidator, QCloseEvent


MODELS_FOLDER = '..\\..\\..\\..\\models'

SCALER_FILE = 'zscaler.model'
PCA_FILE = 'pca-12.model'
MODEL_FILE = 'random-forest-scaled-pca12.model'


WINDOW_X = 300
WINDOW_Y = 300
WINDOW_WIDTH = 700
WINDOW_HEIGHT = 100


OBSERVER_MSLEEP = 500

class MainWindow:
    selected_detector: DetectorName

class DetectorRadioButton(QRadioButton):
    def __init__(self, window: QWidget, name: DetectorName):
        super().__init__(name)
        self.__window = window
        self.__name = name

    def changeDetector(self):
        self.__window.selected_detector = self.__name


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


class SamplingRateValidator(QIntValidator):
    def validate(self, a0: str, a1: int) -> tuple[QValidator.State, str, int]:
        top_validation = super().validate(a0, a1)
        if top_validation[0] == QValidator.State.Acceptable and int(a0) > 0:
            return top_validation
        return QValidator.State.Invalid, top_validation[1], top_validation[2]


class MainWindow(QWidget):

    def __init__(self):
        super().__init__()
        
        self.__scaler = loadModel(os.path.join(MODELS_FOLDER, SCALER_FILE))
        self.__pca = loadModel(os.path.join(MODELS_FOLDER, PCA_FILE))
        self.__model = loadModel(os.path.join(MODELS_FOLDER, MODEL_FILE))

        self.__record: PhysioRecord = None

        self.file_path = ""
        self.selected_detector = DetectorNames.DETECTOR_HAMILTON

        self.setGeometry(WINDOW_X, WINDOW_Y, WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setWindowTitle('Heart diseases prediction')

        #region Main layout
        self.outer_layout = QVBoxLayout()

            #region Select file layout
        self.select_file_layout = QHBoxLayout()
        
        self.selected_file_line_edit = QLineEdit(self.file_path)
        self.selected_file_line_edit.setReadOnly(True)
        self.select_file_layout.addWidget(self.selected_file_line_edit)

        self.select_file_button = QPushButton("Select file")
        self.select_file_button.clicked.connect(self.selectFile)
        self.select_file_layout.addWidget(self.select_file_button)

        self.outer_layout.addLayout(self.select_file_layout)
        
        self.record_annotation_layout = QFormLayout()
        self.record_annotation_line_edit = QLineEdit()
        self.record_annotation_line_edit.setDisabled(True)
        self.record_annotation_line_edit.textChanged.connect(lambda: self.__record.setAnnotation(self.record_annotation_line_edit.text()))
        self.record_annotation_layout.addRow("Annotation file extension:", self.record_annotation_line_edit)

        self.outer_layout.addLayout(self.record_annotation_layout)

        self.select_file_result_label = QLabel()
        self.select_file_result_label.setStyleSheet("QLabel { color: rgba(255, 0, 0, 1); }")
        self.outer_layout.addWidget(self.select_file_result_label)
            #endregion
        
        self.file_selected_widgets = WidgetsContainer()
            
            #region Settings
        self.settings_layout = QVBoxLayout()
        
                #region Sampling rate
        self.sampling_rate_layout = QFormLayout()
        self.sampling_rate_line_edit = QLineEdit("0")
        self.sampling_rate_line_edit.setValidator(SamplingRateValidator())
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
        self.use_smoothing_checkbox.toggled.connect(lambda: self.smoothing_settings.setDisabled(not self.use_smoothing_checkbox.isChecked()))

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

        self.smoothing_settings.setDisabled(not self.use_smoothing_checkbox.isEnabled())
        self.settings_layout.addLayout(self.smoothing_settings_layout)
                    #endregion
                
                #endregion

                #region Split record
        self.split_record_checkbox = QCheckBox("Split record into chunks")
        self.split_record_checkbox.setChecked(True)
        self.file_selected_widgets.add(self.split_record_checkbox)
        self.settings_layout.addWidget(self.split_record_checkbox)
        
        self.split_record_settings = WidgetsContainer()
        self.split_record_checkbox.toggled.connect(lambda: self.split_record_settings.setDisabled(not self.split_record_checkbox.isChecked()))

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

        for detector_name in DetectorNames.ALL:
            detector_radio_button = DetectorRadioButton(self, detector_name)
            detector_radio_button.setChecked(detector_name == self.selected_detector)
            detector_radio_button.clicked.connect(detector_radio_button.changeDetector)
            self.file_selected_widgets.add(detector_radio_button)
            self.settings_layout.addWidget(detector_radio_button)
                #endregion
        
        self.outer_layout.addLayout(self.settings_layout)

            #endregion
        
            #region Results
        self.results_layout = QVBoxLayout()

        self.predict_button = QPushButton("Predict")
        self.predict_button.clicked.connect(self.__process)
        self.results_layout.addWidget(self.predict_button)
        self.file_selected_widgets.add(self.predict_button)

        self.results_label = QLabel()
        self.results_layout.addWidget(self.results_label)
        self.file_selected_widgets.add(self.results_label)

        self.outer_layout.addLayout(self.results_layout)
            #endregion

        self.file_selected_widgets.setDisabled(not self.is_file_selected)
        
        #endregion

        self.setLayout(self.outer_layout)

        self.show()

    def showMessageBox(self, title: str, msg: str) -> None:
        mb = QMessageBox()
        mb.setWindowTitle(title)
        mb.setText(msg)
        mb.exec_()


    @property
    def is_file_selected(self) -> bool:
        return self.file_path != ""

    def setDetector(self, name: DetectorName):
        print(f"<<<<< [changeDetector: {name}] current detector: {self.selected_detector}")
        if name != self.selected_detector:
            self.selected_detector = name
        print(f"[changeDetector: {name}] current detector: {self.selected_detector} >>>>>")

    def selectFile(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, 'Select ECG file', "", "All Files (*);; WFDB Files (*.dat);")
        if file_path:
            self.file_selected_widgets.setDisabled(False)
            self.file_path = file_path
            self.selected_file_line_edit.setText(self.file_path)

            if os.path.splitext(self.file_path)[-1] == '.dat':
                try:
                    self.__record = PhysioRecord.createFromPath(self.file_path, None)
                except:
                    self.__record = None
                    file_name = pathlib.Path(self.file_path).stem
                    self.showMessageBox("Could not read WFDB data", f"Got .dat file. Could not find header file {file_name}.hea or header syntax is wrong. Accepting {file_name}.dat file as raw ECG.")
                    
                self.record_annotation_line_edit.setDisabled(self.__record is None)


    def __process(self):
        sampling_rate = int(self.sampling_rate_line_edit.text())
        if sampling_rate <= 0:
            self.showMessageBox("Wrong sampling rate", "Got wrong sampling rate value")
            return
        
        annotation = self.record_annotation_line_edit.text()
        if self.__record and annotation:
            self.__record.annotation = annotation
        use_smoothing = self.use_smoothing_checkbox.isChecked()
        smoothing_reference_mean = float(self.smooth_reference_mean_lineedit.text())
        smoothing_reference_std = float(self.smooth_reference_std_lineedit.text())
        smoothing_iterations = int(self.smooth_iterations.text())
        smoothing_window = int(self.smooth_window.text())
        use_chunks = self.split_record_checkbox.isChecked()
        chunks_length = float(self.split_record_legnth.text())
        detector = self.selected_detector
        try:
            if self.__record:
                features = getPhysioRecordFeatures(
                    self.__record,
                    detector_name=detector,
                    use_chunks=use_chunks,
                    chunks_length_s=chunks_length,
                    use_smoothing=use_smoothing,
                    smooth_reference_mean_ms=smoothing_reference_mean,
                    smooth_reference_std_ms=smoothing_reference_std,
                    smooth_iterations=smoothing_iterations,
                    smooth_window_size=smoothing_window
                )
            else:
                raw_ecg = np.loadtxt(self.file_path)
                if raw_ecg.ndim > 1:
                    confirmation = QMessageBox.question(self, 'Confirm', f'Read raw ECG with shape {raw_ecg.shape}. Going to use the first column for data. Continue?', QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes)
                    if confirmation == QMessageBox.StandardButton.No:
                        return
                features = getRawEcgFeatures(
                    raw_ecg, sampling_rate, 
                    detector_name=detector,
                    use_chunks=use_chunks,
                    chunks_length_s=chunks_length,
                    use_smoothing=use_smoothing,
                    smooth_reference_mean_ms=smoothing_reference_mean,
                    smooth_reference_std_ms=smoothing_reference_std,
                    smooth_iterations=smoothing_iterations,
                    smooth_window_size=smoothing_window
                )
        except Exception as e:
            self.showMessageBox("Error while getting features", str(e))
        features = self.__scaler.transform(features)
        features = self.__pca.transform(features)
        proba = self.__model.predict_proba(features).mean(axis=0)
        mean_prediction = self.__model.classes_[np.argmax(proba)]
        self.showMessageBox("Results", f"Mean prediction result: {mean_prediction} (also showed at the bottom of program)")
        self.results_label.setText(f"Last result: {mean_prediction}")
        
        
if __name__=='__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
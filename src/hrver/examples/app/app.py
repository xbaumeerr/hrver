import os
import sys
import time
import pathlib

import pandas as pd
import numpy as np

from hrver.api import PhysioRecord
from hrver.utils import loadModel
from hrver.features import getPhysioRecordFeatures, getRawEcgFeatures
from hrver.constants import REFERENCE_MEAN_MS, REFERENCE_STD_MS, DetectorName, DetectorNames, DiseaseType

from PyQt5.QtCore import QThread, pyqtSignal, QObject

from PyQt5.QtWidgets import QApplication, QWidget, \
    QFileDialog, QVBoxLayout, QHBoxLayout, QFormLayout, \
    QLineEdit, QPushButton, QLabel, QCheckBox, QRadioButton, \
    QMessageBox, QProgressBar

from PyQt5.QtGui import QValidator, QIntValidator, QDoubleValidator, QCloseEvent


MODELS_FOLDER = '..\\..\\..\\..\\models'
SCALER_FILE = 'zscaler.model'
PCA_FILE = 'pca-12.model'
MODEL_FILE = 'random-forest-scaled-pca12.model'

WINDOW_X = 300
WINDOW_Y = 300
WINDOW_WIDTH = 700
WINDOW_HEIGHT = 100


class WidgetsContainer(QObject):
    toggled_disable = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.__disabled = True
        self.__widgets: list[QWidget] = []

    def add(self, widget: QWidget):
        widget.setDisabled(self.__disabled)
        self.__widgets.append(widget)

    def setDisabled(self, disabled: bool):
        self.toggled_disable.emit(disabled)
        self.__disabled = disabled
        for widget in self.__widgets:
            widget.setDisabled(disabled)


class SamplingRateValidator(QIntValidator):
    def validate(self, a0: str, a1: int) -> tuple[QValidator.State, str, int]:
        state, o0, o1 = base_validation = super().validate(a0, a1)
        if state == QValidator.State.Acceptable and int(a0) > 0:
            return base_validation
        return QValidator.State.Invalid, o0, o1


class ProcessingThread(QThread):
    process_ended = pyqtSignal(pd.DataFrame, DiseaseType, float, bool, str, float)
    progress_stepped = pyqtSignal(int, str)

    def __init__(self, parameters: dict, pca, scaler, model, record: PhysioRecord = None, file_path: str = None):
        super().__init__()
        self.__record = record
        self.__parameters = parameters
        self.__scaler = scaler
        self.__pca = pca
        self.__model = model
        self.__file_path = file_path

        self.__error = None
        self.__features = None
        self.__prediction = None

        self.progress_stepped.emit(0, '')
        total_steps = 2
        total_steps += 1 if self.__record and self.__record.annotation else len(DetectorNames.ALL)
        self.__progress_step = 100 // total_steps
        self.__progress = 0

    def step(self, msg: str = ''):
        self.__progress += self.__progress_step
        self.progress_stepped.emit(self.__progress, msg)

    def run(self):
        proba = 0
        start_time = time.time()
        try:
            if self.__record and self.__record.annotation:
                self.__features = getPhysioRecordFeatures(self.__record, **self.__parameters)
                self.step("Extracted features from physio record")
            else:
                ecg = self.__record if self.__record else np.loadtxt(self.__file_path)
                featureExtractor = getPhysioRecordFeatures if self.__record else getRawEcgFeatures
                detector_name = DetectorNames.ALL[0]
                self.__features = featureExtractor(ecg, detector_name=detector_name, **self.__parameters)
                self.step(f"Extracted features using {detector_name}")
                for detector_name in DetectorNames.ALL[1:]:
                    next_features = featureExtractor(ecg, detector_name=detector_name, **self.__parameters)
                    self.step(f"Extracted features using {detector_name}")
                    self.__features = pd.concat((self.__features, next_features))
            
            processed_features = self.__pca.transform(self.__scaler.transform(self.__features))
            self.step("Preprocessed features")
            proba = self.__model.predict_proba(processed_features)
            for i, dn in enumerate(DetectorNames.ALL):
                print(dn)
                for j, dt in enumerate(self.__model.classes_):
                    print("\t", dt, proba[i, j])
            proba = proba.mean(axis=0)
            self.step("Got prediction")
            self.__prediction = self.__model.classes_[np.argmax(proba)]
            proba = proba.max()
        except Exception as e:
            self.__error = e
            proba = 0
            self.__features = pd.DataFrame()
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.process_ended.emit(self.__features, self.__prediction, proba, self.__error is not None, str(self.__error), elapsed_time)



class MainWindow(QWidget):

    def __init__(self):
        super().__init__()
        
        self.__processing_thread = None

        self.__scaler = loadModel(os.path.join(MODELS_FOLDER, SCALER_FILE))
        self.__pca = loadModel(os.path.join(MODELS_FOLDER, PCA_FILE))
        self.__model = loadModel(os.path.join(MODELS_FOLDER, MODEL_FILE))

        self.__record: PhysioRecord = None

        self.__file_path = ""

        self.setGeometry(WINDOW_X, WINDOW_Y, WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setWindowTitle('Heart diseases prediction')

        self.__process_depended_widgets = WidgetsContainer()
        self.__process_depended_widgets.setDisabled(False)

        self.__outer_layout = QVBoxLayout()
        self.__addSelectFileRegion()
        self.__addSettingsRegion()
        self.__addResultsRegion()

        self.__file_selected_widgets.setDisabled(self.__file_path == "")
        
        self.setLayout(self.__outer_layout)
        self.show()

    def closeEvent(self, event: QCloseEvent):
        if self.__processing_thread and self.__processing_thread.isRunning():
            reply = QMessageBox.question(self, "Confirm exit", "Prediction in progress... Are you sure you want to exit?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
            self.__processing_thread.terminate()
            self.__processing_thread.wait()
        super().closeEvent(event)

    @property
    def is_file_selected(self) -> bool:
        return self.__file_path != ""

    def __selectFile(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, 'Select ECG file', "", "All Files (*);; WFDB Files (*.dat);", options=options)
        if file_path:
            self.__file_selected_widgets.setDisabled(False)
            
            if self.__file_path != file_path:
                self.__record = None
                self.__sampling_rate_line_edit.setText("0")

            
            self.__file_path = file_path
            self.__selected_file_line_edit.setText(self.__file_path)

            
            if os.path.splitext(self.__file_path)[-1] == '.dat':
                try:
                    self.__record = PhysioRecord.createFromPath(self.__file_path, None)
                except Exception as e:
                    self.__record = None
                    file_name = pathlib.Path(self.__file_path).stem
                    QMessageBox.information(self, "Could not read WFDB data", f"Got .dat file. Could not find header file {file_name}.hea or header syntax is wrong. Accepting {file_name}.dat file as raw ECG. Error: {e}")
                
                self.__record_annotation_line_edit.setDisabled(self.__record is None)
                if self.__record is not None:
                    self.__sampling_rate_line_edit.setText(str(self.__record.sampling_rate))
                    


    def __process(self):
        sampling_rate = int(self.__sampling_rate_line_edit.text())
        if sampling_rate <= 0:
            QMessageBox.warning(self, "Wrong sampling rate", "Got wrong sampling rate value")
            return
        
        self.__results_label.setText("Processing...")
        
        annotation = self.__record_annotation_line_edit.text()
        if self.__record and annotation:
            self.__record.annotation = annotation
        
        parameters = {
                "use_chunks": self.__split_record_checkbox.isChecked(),
                "chunks_length_s": float(self.__split_record_legnth.text()),
                "use_smoothing": self.__use_smoothing_checkbox.isChecked(),
                "smooth_reference_mean_ms": float(self.__smooth_reference_mean_lineedit.text()),
                "smooth_reference_std_ms": float(self.__smooth_reference_std_lineedit.text()),
                "smooth_iterations": int(self.__smooth_iterations.text()),
                "smooth_window_size": int(self.__smooth_window.text())
        }
        if not self.__record:
            parameters['sampling_rate'] = sampling_rate

        self.__process_depended_widgets.setDisabled(True)

        self.__processing_progress_bar.setStyleSheet("QProgressBar::chunk { background-color: green; }")
        self.__processing_progress_bar.setTextVisible(False)
        self.__processing_thread = ProcessingThread(parameters, self.__pca, self.__scaler, self.__model, record=self.__record, file_path=self.__file_path)
        self.__processing_thread.start()

        self.__processing_thread.process_ended.connect(self.__onProcessEnded)
        self.__processing_thread.progress_stepped.connect(self.__onProgressStepped)
    
    def __onProcessEnded(self, features: pd.DataFrame, prediction: DiseaseType, probability: float, with_error: bool, error: str, elapsed_time: float):
        self.__process_depended_widgets.setDisabled(False)
        
        if with_error:
            QMessageBox.critical(self, "Error", f"Got error during prediction: {error}")
            self.__results_label.setText('')
            self.__processing_progress_bar.setValue(0)
            self.__processing_progress_bar.setStyleSheet("QProgressBar::chunk { background-color: red; }")
            return
        self.__processing_progress_bar.setTextVisible(True)
        self.__processing_progress_bar.setFormat(f"Elapsed time: {elapsed_time:.2f} s")
        result_message = f"Results: {prediction} ({probability * 100:.2f}% probability)"
        QMessageBox.information(self, "Results", result_message)
        self.__results_label.setText(result_message)
        self.__processing_thread.quit()
        self.__processing_thread.wait()
        self.__processing_progress_bar.setValue(100)
        
    def __onProgressStepped(self, progress: int, message: str):
        self.__results_label.setText(f"Processing... {progress} % {f'[{message}]' if message else ''}")
        self.__processing_progress_bar.setValue(progress)

    def __addSelectFileRegion(self):
        self.__select_file_layout = QHBoxLayout()
        
        self.__selected_file_line_edit = QLineEdit(self.__file_path)
        self.__selected_file_line_edit.setReadOnly(True)
        self.__process_depended_widgets.add(self.__selected_file_line_edit)
        self.__select_file_layout.addWidget(self.__selected_file_line_edit)

        self.__select_file_button = QPushButton("Select file")
        self.__process_depended_widgets.add(self.__select_file_button)
        self.__select_file_button.clicked.connect(self.__selectFile)
        self.__select_file_layout.addWidget(self.__select_file_button)

        self.__outer_layout.addLayout(self.__select_file_layout)
        
        self.__record_annotation_layout = QFormLayout()
        self.__record_annotation_line_edit = QLineEdit()
        self.__process_depended_widgets.add(self.__record_annotation_line_edit)
        self.__record_annotation_line_edit.setDisabled(True)
        self.__record_annotation_line_edit.textChanged.connect(lambda: self.__record.setAnnotation(self.__record_annotation_line_edit.text()))
        self.__record_annotation_layout.addRow("Annotation file extension:", self.__record_annotation_line_edit)

        self.__outer_layout.addLayout(self.__record_annotation_layout)

        self.__select_file_result_label = QLabel()
        self.__process_depended_widgets.add(self.__select_file_result_label)
        self.__select_file_result_label.setStyleSheet("QLabel { color: rgba(255, 0, 0, 1); }")
        self.__outer_layout.addWidget(self.__select_file_result_label)

        self.__file_selected_widgets = WidgetsContainer()

    def __addSettingsRegion(self):
        self.__settings_layout = QVBoxLayout()
        self.__addSamplingRateSettingsRegion()
        self.__addSmoothingSettingsRegion()
        self.__addSplitingSettingsRegion()
        self.__outer_layout.addLayout(self.__settings_layout)

    def __addResultsRegion(self):
        self.__results_layout = QVBoxLayout()

        self.__predict_button = QPushButton("Predict")
        self.__predict_button.clicked.connect(self.__process)
        self.__process_depended_widgets.add(self.__predict_button)
        self.__results_layout.addWidget(self.__predict_button)
        self.__file_selected_widgets.add(self.__predict_button)

        self.__results_label = QLabel()
        self.__results_label.font().setPointSize(14)
        self.__results_layout.addWidget(self.__results_label)
        self.__file_selected_widgets.add(self.__results_label)

        self.__processing_progress_bar = QProgressBar()
        self.__processing_progress_bar.setValue(0)
        self.__processing_progress_bar.setTextVisible(False)
        self.__results_layout.addWidget(self.__processing_progress_bar)

        self.__outer_layout.addLayout(self.__results_layout)
    
    def __addSamplingRateSettingsRegion(self):
        self.__sampling_rate_layout = QFormLayout()
        self.__sampling_rate_line_edit = QLineEdit("0")
        self.__sampling_rate_line_edit.setValidator(SamplingRateValidator())
        self.__process_depended_widgets.add(self.__sampling_rate_line_edit)
        self.__file_selected_widgets.add(self.__sampling_rate_line_edit)
    
        self.__sampling_rate_layout.addRow("Sampling rate:", self.__sampling_rate_line_edit)
        self.__settings_layout.addLayout(self.__sampling_rate_layout)

    def __addSmoothingSettingsRegion(self):
        self.__use_smoothing_checkbox = QCheckBox("Use smoothing")
        self.__process_depended_widgets.add(self.__use_smoothing_checkbox)
        self.__file_selected_widgets.add(self.__use_smoothing_checkbox)
        self.__settings_layout.addWidget(self.__use_smoothing_checkbox)
                    
        self.__smoothing_settings_layout = QFormLayout()
        self.__smoothing_settings = WidgetsContainer()
        update_smothing_checkbox = lambda: self.__smoothing_settings.setDisabled(not self.__use_smoothing_checkbox.isChecked())
        self.__use_smoothing_checkbox.toggled.connect(update_smothing_checkbox)
        self.__file_selected_widgets.toggled_disable.connect(update_smothing_checkbox)

        self.__smooth_reference_mean_lineedit = QLineEdit(str(REFERENCE_MEAN_MS))
        self.__process_depended_widgets.add(self.__smooth_reference_mean_lineedit)
        self.__smooth_reference_mean_lineedit.setValidator(QDoubleValidator())
        self.__smoothing_settings_layout.addRow("Smoothing reference mean (ms):", self.__smooth_reference_mean_lineedit)
        self.__smoothing_settings.add(self.__smooth_reference_mean_lineedit)

        self.__smooth_reference_std_lineedit = QLineEdit(str(REFERENCE_STD_MS))
        self.__process_depended_widgets.add(self.__smooth_reference_std_lineedit)
        self.__smooth_reference_std_lineedit.setValidator(QDoubleValidator())
        self.__smoothing_settings_layout.addRow("Smoothing reference standard deviation (ms):", self.__smooth_reference_std_lineedit)
        self.__smoothing_settings.add(self.__smooth_reference_std_lineedit)

        self.__smooth_iterations = QLineEdit("10")
        self.__process_depended_widgets.add(self.__smooth_iterations)
        self.__smooth_iterations.setValidator(QIntValidator())
        self.__smoothing_settings_layout.addRow("Smoothing iterations:", self.__smooth_iterations)
        self.__smoothing_settings.add(self.__smooth_iterations)

        self.__smooth_window = QLineEdit("3")
        self.__process_depended_widgets.add(self.__smooth_window)
        self.__smooth_window.setValidator(QIntValidator())
        self.__smoothing_settings_layout.addRow("Smoothing window:", self.__smooth_window)
        self.__smoothing_settings.add(self.__smooth_window)

        self.__smoothing_settings.setDisabled(not self.__use_smoothing_checkbox.isEnabled())
        self.__settings_layout.addLayout(self.__smoothing_settings_layout)

        
    def __addSplitingSettingsRegion(self):
        self.__split_record_checkbox = QCheckBox("Split record into chunks")
        self.__process_depended_widgets.add(self.__split_record_checkbox)
        self.__split_record_checkbox.setChecked(True)
        self.__file_selected_widgets.add(self.__split_record_checkbox)
        self.__settings_layout.addWidget(self.__split_record_checkbox)
        
        self.__split_record_settings = WidgetsContainer()
        update_split_record_checkbox = lambda: self.__split_record_settings.setDisabled(not self.__split_record_checkbox.isChecked())
        self.__split_record_checkbox.toggled.connect(update_split_record_checkbox)
        self.__file_selected_widgets.toggled_disable.connect(update_split_record_checkbox)

        self.__split_record_settings_layout = QFormLayout()
        
        self.__split_record_legnth = QLineEdit("90")
        self.__process_depended_widgets.add(self.__split_record_legnth)
        self.__split_record_legnth.setValidator(QDoubleValidator())
        self.__split_record_settings_layout.addRow("Chunks length (s):", self.__split_record_legnth)
        self.__split_record_settings.add(self.__split_record_legnth)

        self.__split_record_settings.setDisabled(True)
        self.__settings_layout.addLayout(self.__split_record_settings_layout)

if __name__=='__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
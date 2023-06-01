import os
import sys

import numpy as np

from hrver.utils import loadModel
from hrver.features import getPhysioRecordFeatures, getRawEcgFeatures

from PyQt5.QtWidgets import QWidget, QApplication

MODELS_FOLDER = '..\\..\\..\\..\\models'

SCALER_FILE = 'zscaler.model'
PCA_FILE = 'pca-12.model'
MODEL_FILE = 'random-forest-scaled-pca12.model'


WINDOW_X = 300
WINDOW_Y = 300
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 100


class MainWindow(QWidget):

    def __init__(self):
        super().__init__()
        
        self.__scaler = loadModel(os.path.join(MODELS_FOLDER, SCALER_FILE))
        self.__pca = loadModel(os.path.join(MODELS_FOLDER, PCA_FILE))
        self.__model = loadModel(os.path.join(MODELS_FOLDER, MODEL_FILE))

        self.setGeometry(WINDOW_X, WINDOW_Y, WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setWindowTitle('Heart diseases prediction')
        


        self.show()
    
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
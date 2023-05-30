import pickle
import utils
import numpy as np

from PyQt5.QtWidgets import QWidget, QPushButton, QFileDialog, QVBoxLayout, QLabel, QInputDialog, QLineEdit

SCALER_FILE = 'scaler'
PCA_FILE = 'pca'
MODEL_FILE = 'model'

class MainWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.__current_ecg_filename = None
        self.__model = None
        with open(MODEL_FILE, 'rb') as f:
            self.__model = pickle.load(f)
        self.__scaler = None
        with open(SCALER_FILE, 'rb') as f:
            self.__scaler = pickle.load(f)
        self.__pca = None
        with open(PCA_FILE, 'rb') as f:
            self.__pca = pickle.load(f)
        self.initUI()

    def initUI(self):
        X, Y, W, H = 300, 300, 400, 100
        self.setGeometry(X, Y, W, H)
        self.setWindowTitle('Heart diseases prediction')
        
        self.select_file_button = QPushButton(text="Select raw ECG file", parent=self)
        self.select_file_button.setFixedWidth(W)
        self.select_file_button.clicked.connect(self.openFileDialog)

        self.results_label = QLabel(text="", parent=self)

        layout = QVBoxLayout()
        layout.addWidget(self.select_file_button)
        layout.addWidget(self.results_label)
        self.setLayout(layout)

        self.show()
    
    def openFileDialog(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, 'Select raw ECG file', "", "All Files (*);", options=options)
        if file_name:
            self.__processData(file_name)
        
        
    def __processData(self, file_name):
        ok = False
        sampling_rate = 0
        msg = "Sampling rate"
        while not ok:
            sampling_rate, ok = QInputDialog().getInt(self, msg, "Enter sampling rate:", QLineEdit.Normal)
            msg = "Sampling rate: wrong sampling rate entered"

        raw_ecg = np.loadtxt(file_name)
        if np.any(np.isnan(raw_ecg)):
            self.results_label.setText('Data contains NaN elements')
            return
        features = None
        try:
            features = utils.getAllFeatures(raw_ecg, sampling_rate)
        except Exception as e:
            self.results_label.setText('Got error getting features')
            return
        features = utils.preprocessFeatures(features, self.__scaler, self.__pca)
        answer = self.__model.predict(features)[0]
        self.results_label.setText(answer)
        
        
        

            
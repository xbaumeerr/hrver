from typing import TypeAlias

REFERENCE_MEAN_MS = 800.0
REFERENCE_STD_MS = 80.0

INTERPOLATION_DELTA_TIME_MS = 250.0
CHUNKS_LENGTH_S = 90.0

class AllFields(type):
    """Metaclass to create additional tuple ALL that contains all fields of a class"""
    def __init__(cls, name, bases, dct):
        all = []
        for key, value in dct.items():
            if not key.startswith('__'):
                all.append(value)
        cls.ALL = tuple(all)
        types = type(cls.ALL[-1])

class DiseaseTypes(metaclass=AllFields):
    NORM = "Normal"
    AF = "Artial fibrillation"
    VT = "Ventricular tachyarrhythmia"
    MI = "Myocardial infarction"
    HF = "Heart failure"
    BBB = "Bundle branch block"
    ANG = "Angina pectoris"
    CHD = "Coronary heart disease"
DiseaseType: TypeAlias = str

class FeatureNames(metaclass=AllFields):
    HR = "hr"
    RMSSD = "rmssd"
    PNN50 = "pnn50"
    MODE = "mode"
    AMO50 = "amo50"
    MXRMN = "mxrmn"
    MXDMN = "mxdmn"
    SI = "si"
    CC1 = "cc1"
    CC0 = "cc0"
    LFMAX = "lfmax"
    HFMAX = "hfmax"
    LFHF = "lfhf"
FeatureName: TypeAlias = str

class DetectorNames(metaclass=AllFields):
    DETECTOR_CHRISTOV = "christov_detector"
    DETECTOR_ENGZEE = "engzee_detector"
    DETECTOR_HAMILTON = "hamilton_detector"
    DETECTOR_PAN_TOMPKINS = "pan_tompkins_detector"
    DETECTOR_SWT = "swt_detector"
    DETECTOR_WQRS = "wqrs_detector"
DetectorName: TypeAlias = str
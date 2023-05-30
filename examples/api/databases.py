import sys
sys.path.append('..\\..')

import os
import wfdb
from collections import defaultdict

from hrver.constants import DiseaseTypes, DiseaseType
from hrver.api import PhysioBase, PhysioRecord

class AFTDatabase(PhysioBase):
    """AF Termination Challenge Database"""
    def getDiseaseType(self, record: PhysioRecord) -> DiseaseType:
        return DiseaseTypes.AF

class CEBSDatabase(PhysioBase):
    """Combined measurement of ECG, Breathing and Seismocardiograms"""
    def getDiseaseType(self, record: PhysioRecord) -> DiseaseType:
        return DiseaseTypes.NORM

class CUDatabase(PhysioBase):
    """CU Ventricular Tachyarrhythmia Database"""
    def getDiseaseType(self, record: PhysioRecord) -> DiseaseType:
        return DiseaseTypes.VT

class CHFDatabase(PhysioBase):
    """BIDMC Congestive Heart Failure Database"""
    def getDiseaseType(self, record: PhysioRecord) -> DiseaseType:
        return DiseaseTypes.HF

class PTBDatabase(PhysioBase):
    """PTB Diagnostic ECG Database"""

    __DIAGNOSIS_TO_DISEASE_TYPE = {
        "myocardial infarction" : DiseaseTypes.MI,
        "healthy control" : DiseaseTypes.NORM,
        "heart failure (nyha 2)": DiseaseTypes.HF,
        "heart failure (nyha 3)": DiseaseTypes.HF,
        "heart failure (nyha 4)": DiseaseTypes.HF,
        "stable angina" : DiseaseTypes.ANG,
        "bundle branch block" : DiseaseTypes.BBB,
        "unstable angina" : DiseaseTypes.ANG
    }

    def getDiseaseType(self, record: PhysioRecord) -> DiseaseType:
        for comment in record.header.comments:
            if comment.startswith("Reason for admission:"):
                return self.__DIAGNOSIS_TO_DISEASE_TYPE.get(comment.split(':')[-1].strip().lower())
        return None

class INCARTDatabase(PhysioBase):
    """St Petersburg INCART 12-lead Arrhythmia Database"""

    __DIAGNOSIS_TO_DISEASE_TYPE = {
        "Coronary artery disease, arterial hypertension" : DiseaseTypes.CHD,
        "Acute MI" : DiseaseTypes.MI,
        "Transient ischemic attack" : DiseaseTypes.ANG,
        "Earlier MI" : DiseaseTypes.MI,
        "tachycardia" : DiseaseTypes.VT,
        "AV nodal block" : DiseaseTypes.BBB,
        "bundle branch block" : DiseaseTypes.BBB,
        "blocked APCs" : DiseaseTypes.BBB,
        "Coronary artery disease, arterial hypertension, left ventricular hypertrophy" : DiseaseTypes.CHD,
    }
    
    __PATIENTS_DIAGNOSES_FILENAME = "files-patients-diagnoses.txt"
    __RECORDS_DESCRIPTION_FILENAME = "record-descriptions.txt"

    def __init__(self, link: str, annotation: str = None):
        super().__init__(link, annotation)

        wfdb.dl_files(self.label, os.getcwd(), [self.__PATIENTS_DIAGNOSES_FILENAME], False, True)

        self.__record_to_patient_disease = defaultdict(list)
        with open(self.__PATIENTS_DIAGNOSES_FILENAME) as f:
            patient_subindex = 1
            records = []
            for i, line in enumerate(f.readlines()):
                line = line.strip()
                if patient_subindex == 2:
                    records = line.split(" ")
                elif patient_subindex == 3:
                    disease = self.__DIAGNOSIS_TO_DISEASE_TYPE.get(line)
                    if disease:
                        for record in records:
                            self.__record_to_patient_disease[record] = disease
                    patient_subindex = 0
                patient_subindex += 1

        wfdb.dl_files(self.label, os.getcwd(), [self.__RECORDS_DESCRIPTION_FILENAME], False, True)
        self.__record_to_disease = {}
        with open(self.__RECORDS_DESCRIPTION_FILENAME) as f:
            record = None
            for i, line in enumerate(f.readlines()):
                line = line.strip()
                if i % 2 == 0:
                    record = line
                else:
                    if record in self.__record_to_patient_disease: continue
                    for disease_name, disease_type in self.__DIAGNOSIS_TO_DISEASE_TYPE.items():
                        if disease_name in line:
                            self.__record_to_disease[record] = disease_type

    def getDiseaseType(self, record: PhysioRecord) -> DiseaseType:
        record_name = record.path.upper()
        disease_type = self.__record_to_patient_disease.get(record_name)
        if disease_type:
            return disease_type
        return self.__record_to_disease.get(record_name)
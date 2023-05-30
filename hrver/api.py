import os
import pickle
import typing

import numpy as np
import wfdb

from ecgdetectors import Detectors
from constants import DiseaseType, DetectorNames, DetectorName


class PhysioRecord:
    pass

class PhysioBase:
    """
    An abstract API class to work with WFDB library
    
    Parameters
    ----------
    link
        URL link to PhysioNet database
    annotation
        Extention of annotation file of database

    Attributes
    ----------
    link : str
        URL link to PhysioNet database
    annotation : str
        Extention of annotation file of database
    label : str
        Label of PhysioNet database
    version : str
        Version of database
    records_list : tuple
        Tuple of all records in a database

    """

    def __init__(self, link: str, annotation: str = None):
        self.__link = link
        suburl = link.split('/')
        if not suburl[-1]:
            suburl.pop()
        self.__label = suburl[-2]
        self.__version = suburl[-1]
        self.__annotation = annotation
        self.__records_list = tuple(wfdb.get_record_list(self.label))
        self.__records_cache = {}

    @property
    def link(self):
        return self.__link
    
    @property
    def label(self):
        return self.__label
    
    @property
    def version(self):
        return self.__version

    @property
    def annotation(self):
        return self.__annotation

    @property
    def records_list(self) -> tuple[str]:
        """Get a tuple of records belonging to a database"""
        return self.__records_list 

    def __getitem__(self, index: int) -> PhysioRecord:
        """Get PhysioRecord by index in record list"""
        if index in self.__records_cache:
            return self.__records_cache[index]
        else:
            record = PhysioRecord(self, self.__records_list[index])
            self.__records_cache[index] = record
            return record

    def clearCache(self):
        """Clears records cache to free RAM"""
        for record in self.__records_cache.values():
            record.clearCache()
        self.__records_cache.clear()
    
    def dumpCache(self):
        """Dumps cache of loaded records"""
        for record in self.__records_cache.values():
            record.dumpCache()

    def getDownloadURL(self, path: str) -> str:
        return f"https://physionet.org/files/{self.label}/{self.version}/{path}?download"

    def getDiseaseType(self, record: PhysioRecord) -> DiseaseType:
        """Get disease type of current record"""
        raise NotImplementedError()

    def __str__(self):
        return self.__label
    

class PhysioRecord:
    """
    An API class to work with WFDB ECG records
    
    Parameters
    ----------
    database
        PhysioBase API class object
    record_path
        path of this record in database
    cache_folder
        directory for dumping cache data

    Attributes
    ----------
    database
        source PhysioBase database object
    path
        record`s path in PhysioNet database
    cache_folder
        directory for dumping cache data
    sampling_rate
        record`s sampling rate
    disease_type
        disease class label
    name
        record`s file name
    folder
        record`s source directory
    header
        record`s header WFDB object
    data
        record`s sample data
    """

    DEFAULT_CACHE_FOLDER = 'cache'

    def __init__(self, database: PhysioBase, record_path: str, cache_folder: str = DEFAULT_CACHE_FOLDER):
        self.cache_folder = cache_folder
        
        self.__database = database
        self.__path = record_path

        if '/' in record_path:
            folder, self.__name = record_path.split('/')
            self.__folder = f"{self.__database.label}/{folder}"
        else:
            self.__name, self.__folder = record_path, self.__database.label
        
        self.__header = wfdb.rdheader(self.__name, pn_dir=self.__folder)
        self.__sampling_rate = self.__header.fs

        self.__detectors = Detectors(self.__sampling_rate)

        self.__disease_type = self.__database.getDiseaseType(self)

        self.__signal = None
        self.__channel_index = None
        self.__r_indexes = None
        self.__rr_ms = None

    @property
    def database(self) -> PhysioBase:
        return self.__database

    @property
    def path(self) -> str:
        return self.__path

    @property
    def sampling_rate(self) -> float:
        return self.__sampling_rate

    @property
    def disease_type(self) -> DiseaseType:
        return self.__disease_type

    @property
    def name(self) -> str:
        return self.__name
    
    @property
    def folder(self) -> str:
        return self.__folder

    @property
    def header(self) -> typing.Union[wfdb.Record, wfdb.MultiRecord]:
        return self.__header

    def getAnnotation(self) -> wfdb.Annotation:
        """Get annotation of the record"""
        return wfdb.rdann(self.__name, self.__database.annotation, pn_dir=self.__folder)

    def getRRFromAnnotations(self) -> np.ndarray:
        """Get record`s RR-intervals indexes from database annotations file. Removes the first interval as it is an interval from not existing R peak to existing one."""
        return wfdb.processing.ann2rr(self.__name, self.__database.annotation, pn_dir=self.__folder)[1:]

    def __updateRRData(self, channel_index: int = 0, detector_name: DetectorName = DetectorNames.DETECTOR_HAMILTON, force_use_detector: bool = False, reupdate_data: bool = False):
        """
        Updates cache R data if needed. This method must be called every time we want to get R data
        """
        if reupdate_data or self.__channel_index != channel_index or self.__r_indexes is None or self.__rr_ms is None or self.__signal is None:
            if self.__channel_index == channel_index and self.__doesCacheFileExist():
                if not self.loadCache():
                    print(f"Failed to load data from cache: {self.__signal is None} :: {self.__rr_ms is None} :: {self.__r_indexes is None}. Uploading from PhysioNet")
                    self.__uploadDataFromSource(channel_index, force_use_detector, detector_name)
            else:
                self.__uploadDataFromSource(channel_index, force_use_detector, detector_name)
            self.__channel_index = channel_index
    
    def __uploadDataFromSource(self, channel_index: int, force_use_detector: bool, detector_name: DetectorName):
        """Uploads data from PhysioNet"""
        sample, info = wfdb.rdsamp(self.__name, pn_dir=self.__folder)
        self.__signal = sample[:, channel_index]

        if self.__database.annotation and not force_use_detector:
            annotation = self.getAnnotation()
            self.__r_indexes = annotation.sample
            self.__rr_ms = (self.getRRFromAnnotations() / self.__sampling_rate) * 1e3
        else:
            detector = getattr(self.__detectors, detector_name, None)
            assert detector is not None, f"Wrong detector name {detector_name}"
            
            self.__r_indexes = np.array(detector(self.__signal))
            self.__rr_ms = (np.diff(self.__r_indexes) / self.__sampling_rate) * 1e3

    def getRData(self, channel_index: int = 0, detector_name: DetectorName = DetectorNames.DETECTOR_HAMILTON, force_use_detector: bool = False, reupdate_data: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Base method to get signal of specific channel, R peaks and RR-intervals in milliseconds using either annotations, or if there are no annotations use segmenter
         
         Parameters
        ----------
        channel_index
            index of channel to use in signal
        detector_name
            name of detector to use for R-peaks segmentation (if needed)
        force_use_detector
            force to use detector instead of annotations
        reupdate_data
            reload all data
        """
        self.__updateRRData(channel_index, detector_name, force_use_detector, reupdate_data)
        return self.__signal, self.__r_indexes, self.__rr_ms

    def clearCache(self):
        """Clear cached data to free RAM"""
        self.__r_indexes = None
        self.__rr_ms = None
        self.__signal = None
        if self.__doesCacheFileExist():
            os.remove(self.__getCachePath())

    def __getCachePath(self) -> str:
        """Returns path to cache file"""
        return os.path.join(self.cache_folder, str(hash(self.__path)))

    def __doesCacheFileExist(self) -> bool:
        """Check whether cache file already exists and we can load cache"""
        cache_path = self.__getCachePath()
        return os.path.exists(cache_path)

    def dumpCache(self, debug: bool = False) -> bool:
        """Save cached data to disk and free RAM"""
        if self.__rr_ms is None or self.__r_indexes is None or self.__signal is None:
            if debug: print("Tried to dump None values. Use getRData to upload it first")
        else:
            if not os.path.exists(self.cache_folder):
                os.mkdir(self.cache_folder)
            cache_path = self.__getCachePath()
            with open(cache_path, 'wb') as f:
                pickle.dump({"rr": self.__rr_ms, "r": self.__r_indexes, "signal": self.__signal}, f)
            self.clearCache()

    def loadCache(self) -> bool:
        """Load cache data from disk"""
        if not os.path.exists(self.cache_folder):
            print(f"Path {self.cache_folder} does not exist!")
            return False
    
        cache_path = self.__getCachePath()
        if not os.path.exists(cache_path):
            print(f"Path {cache_path} does not exist!")
            return False

        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
            self.__r_indexes = data['r']
            self.__rr_ms = data['rr']
            self.__signal = data['signal']
        return self.__r_indexes is not None and self.__rr_ms is not None and self.__signal is not None

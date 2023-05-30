import numpy as np
import unittest

from hrv import HRV

from hrver.features import getRRChunks, interpolate, \
    meanHR, RMSSD, pNN50, \
    calculateACFStatistics, calculateHistogramStatistics, calculateSpectralStatistics, \
    getPhysioRecordFeatures
from hrver.constants import FeatureNames
from hrver.examples.api.databases import init

Databases = None

class TestData:
    def __init__(self, record, r_indexes, rr_ms, hrv):
        self.record = record
        self.r_indexes = r_indexes
        self.rr_ms = rr_ms
        self.hrv = hrv


class TestFeatureExtraction(unittest.TestCase):

    def setUp(self):
        global Databases
        if not Databases:
            Databases = init()

        self.chunks_max_time_s = 10.0
        self.interpolation_delta_time_ms = 250.0
        
        self.database = Databases.ptbdb
        records_amount = 1

        self.test_datas = []
        for i in range(records_amount):
            test_record = self.database[i]
            test_hrv = HRV(test_record.sampling_rate)
            test_signal, test_r_indexes, test_rr_ms = test_record.getRData()
            self.test_datas.append(TestData(test_record, test_r_indexes, test_rr_ms, test_hrv))
    
    def infTest(self, value: float):
        self.assertGreater(value, -np.inf)
        self.assertLess(value, np.inf)

    def test_chunks(self):
        chunks_max_time_ms = self.chunks_max_time_s * 1e3

        for test_data in self.test_datas:
            rr_ms = test_data.rr_ms
            total_record_time_ms = rr_ms.sum()
            chunks = getRRChunks(rr_ms, self.chunks_max_time_s)
            total_chunks_time_ms = 0.0

            for chunk in chunks:
                chunk_time_ms = chunk.sum()
                total_chunks_time_ms += chunk_time_ms
                self.infTest(chunk_time_ms)
                self.assertGreaterEqual(chunk_time_ms, chunks_max_time_ms, 1)

            self.assertLess(total_record_time_ms - total_chunks_time_ms, chunks_max_time_ms)
            

    def test_HR(self):
        for test_data in self.test_datas:
            lib_hr = test_data.hrv.HR(test_data.r_indexes).mean()
            own_hr = meanHR(test_data.rr_ms)

            self.infTest(own_hr)
            self.assertAlmostEqual(lib_hr, own_hr)
            self.assertGreater(own_hr, 0)

    def test_RMSSD(self):
        for test_data in self.test_datas:
            lib_rmssd = test_data.hrv.RMSSD(test_data.r_indexes)
            own_rmssd = RMSSD(test_data.rr_ms)

            self.infTest(own_rmssd)
            self.assertAlmostEqual(lib_rmssd, own_rmssd)
            self.assertGreater(own_rmssd, 0)

    def test_pNN50(self):
        for test_data in self.test_datas:
            lib_pnn50 = test_data.hrv.pNN50(test_data.r_indexes)
            own_pnn50 = pNN50(test_data.rr_ms)
            
            self.infTest(own_pnn50)
            self.assertAlmostEqual(lib_pnn50, own_pnn50, places=1)

    def test_histogram_data(self):
        for test_data in self.test_datas:
            histogram_data = calculateHistogramStatistics(test_data.rr_ms)
            
            for value in histogram_data.values():
                self.infTest(value)
            
            self.assertGreater(histogram_data[FeatureNames.MODE], 0.0)
            self.assertGreater(histogram_data[FeatureNames.AMO50], 0.0)
            self.assertGreater(histogram_data[FeatureNames.MXRMN], 0.0)
    
    def test_interpolation(self):
        for test_data in self.test_datas:
            test_rr = test_data.rr_ms
            rr_t_ms, rr_ms_interpolated = interpolate(test_rr, self.interpolation_delta_time_ms)

            self.infTest(rr_ms_interpolated.max())
            self.infTest(rr_ms_interpolated.min())
            self.infTest(rr_ms_interpolated.mean())
            self.assertAlmostEqual(rr_t_ms.max().round(), test_rr[:-1].sum().round())
            self.assertAlmostEqual(rr_ms_interpolated[0], test_rr[0])
            self.assertAlmostEqual(rr_ms_interpolated[-1], test_rr[-1])

    def test_autocorrelationData(self):
        for test_data in self.test_datas:
            rr_ms = test_data.rr_ms
            rr_t_ms, rr_ms_interpolated = interpolate(rr_ms, self.interpolation_delta_time_ms)
            result = calculateACFStatistics(rr_ms, rr_ms_interpolated, self.interpolation_delta_time_ms)
            cc1 = result[FeatureNames.CC1]
            cc0 = result[FeatureNames.CC0]

            self.infTest(cc1)
            self.infTest(cc0)

    def test_spectral_statistics(self):
        for test_data in self.test_datas:
            rr_ms = test_data.rr_ms
            rr_t_ms, rr_ms_interpolated = interpolate(rr_ms, self.interpolation_delta_time_ms)
            result = calculateSpectralStatistics(rr_ms_interpolated, self.interpolation_delta_time_ms)
            lf_max = result[FeatureNames.LFMAX]
            hf_max = result[FeatureNames.HFMAX]
            lf_hf = result[FeatureNames.LFHF]

            self.infTest(lf_max)
            self.infTest(hf_max)
            self.infTest(lf_hf)

            self.assertGreater(lf_hf, 0.0)
            self.assertLess(abs(lf_hf - test_data.hrv.fAnalysis(test_data.r_indexes)), 2.0)

    def test_dump(self):
        for test_data in self.test_datas:
            record = test_data.record
            record.dumpCache()
            self.assertIsNone(record._PhysioRecord__rr_ms)
            self.assertIsNone(record._PhysioRecord__r_indexes)
            self.assertIsNone(record._PhysioRecord__signal)
            
            signal, r_indexes, rr_ms = record.getRData()
            self.assertIsNotNone(rr_ms)
            self.assertIsNotNone(r_indexes)
            self.assertIsNotNone(signal)
        
        self.database.dumpCache()
        for test_data in self.test_datas:
            record = test_data.record
            self.assertIsNone(record._PhysioRecord__rr_ms)
            self.assertIsNone(record._PhysioRecord__r_indexes)
            self.assertIsNone(record._PhysioRecord__signal)

            signal, r_indexes, rr_ms = record.getRData()
            self.assertIsNotNone(rr_ms)
            self.assertIsNotNone(r_indexes)
            self.assertIsNotNone(signal)


    def test_get_r_data(self):
        database = Databases.aftdb
        for i in range(5):
            record = database[i]
            signal, r_indexes, rr_ms = record.getRData()
            self.assertIsNotNone(rr_ms)
            self.assertIsNotNone(r_indexes)
            self.assertIsNotNone(signal)
        
        database.dumpCache()

        for i in range(5):
            record = database[i]
            self.assertIsNone(record._PhysioRecord__rr_ms)
            self.assertIsNone(record._PhysioRecord__r_indexes)
            self.assertIsNone(record._PhysioRecord__signal)
            signal, r_indexes, rr_ms = record.getRData()
            self.assertIsNotNone(rr_ms)
            self.assertIsNotNone(r_indexes)
            self.assertIsNotNone(signal)
    
    def test_getAllFeatures(self):
        for test_data in self.test_datas:
            record = test_data.record
            features = getPhysioRecordFeatures(record)
            self.assertEqual(len(features.columns), 13)
            for feature_name in FeatureNames.ALL:
                self.assertIn(feature_name, features)

    def tearDown(self):
        super().tearDown()
        self.database.clearCache()
        self.database = None


if __name__=='__main__':
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=DeprecationWarning)
        unittest.main(argv=[''], defaultTest='TestFeatureExtraction', verbosity=2)

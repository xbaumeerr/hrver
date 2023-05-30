import numpy as np
import pandas as pd

from scipy.interpolate import CubicSpline
from statsmodels.tsa.stattools import acf as autocorrelationFunction

from collections import defaultdict
from ecgdetectors import Detectors

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

REFERENCE_MEAN_MS = 800.0
REFERENCE_STD_MS = 80.0

class AllFields(type):
    """Metaclass to create additional tuple ALL that contains all fields of a class"""
    def __init__(cls, name, bases, dct):
        all = []
        for key, value in dct.items():
            if not key.startswith('__'):
                all.append(value)
        cls.ALL = tuple(all)
    
class DiseaseType(metaclass=AllFields):
    NORM = "Normal"
    AF = "Artial fibrillation"
    VT = "Ventricular tachyarrhythmia"
    MI = "Myocardial infarction"
    HF = "Heart failure"
    BBB = "Bundle branch block"
    ANG = "Angina pectoris"
    CHD = "Coronary heart disease"

DT = DiseaseType

class FeatureName(metaclass=AllFields):
    HR = "hr"
    RMSSD = "rmssd"
    pNN50 = "pnn50"
    Mode = "mode"
    Amo50 = "amo50"
    MxRMn = "mxrmn"
    MxDMn = "mxdmn"
    SI = "si"
    CC1 = "cc1"
    CC0 = "cc0"
    LFmax = "lfmax"
    HFmax = "hfmax"
    LFHF = "lfhf"


def shouldSmooth(rr_ms: np.ndarray, reference_mean_ms: float = REFERENCE_MEAN_MS, reference_std_ms: float = REFERENCE_STD_MS) -> bool:
    """
    Check if RR-intervals time series is in reference range values
    """
    return np.abs(rr_ms.mean() - reference_mean_ms) > reference_std_ms*3.0


def smoothAnomalies(rr_ms: np.ndarray, reference_mean_ms: float = REFERENCE_MEAN_MS, reference_std_ms: float = REFERENCE_STD_MS, iterations: int = 10, window_size: int = 3) -> np.ndarray:
    """
    Smoothes anomalies of RR-intervals by averaging them

    Parameters
    ----------
    rr_ms
        RR-intervals in milliseconds
    reference_mean_ms, reference_std_ms
        reference mean and std values of RR-interval in milliseconds
    iterations
        amount of iterations to apply smoothing
    window_size
        size of window in average calculation

    Returns
    -------
    smoothed_rr_ms    
    """
    sigma3 = reference_std_ms * 3.0
    rr_ms = np.copy(rr_ms)
    rr_ms_length = len(rr_ms)
    half_window_size = window_size // 2
    for i in range(iterations):
        for rr_index, rr in enumerate(rr_ms):
            if abs(rr - reference_mean_ms) < sigma3:
                continue
            left_index = max(0, rr_index - half_window_size)
            left_mean = reference_mean_ms if left_index == 0 else rr_ms[left_index:rr_index].mean()
            
            right_index = min(rr_ms_length - 1, rr_index + 1 + half_window_size)
            right_mean = reference_mean_ms if right_index == rr_ms_length - 1 else rr_ms[rr_index + 1:right_index].mean()

            rr_ms[rr_index] = (left_mean + right_mean) / 2.0
    return rr_ms


def getRRChunks(rr_ms: np.ndarray, chunks_max_time_s: float = 90.0) -> np.ndarray:
    """Splits array of RR intervals into chunks of time about maximum time. Returns list of arrays."""
    chunks_max_time_ms = chunks_max_time_s * 1e3
    chunk_time = 0
    chunks = []
    chunk = []
    for rr in rr_ms:
        chunk_time += rr
        chunk.append(rr)
        if chunk_time >= chunks_max_time_ms:
            chunk_time = 0
            chunks.append(np.array(chunk))
            chunk = []
    return chunks


def meanHR(rr_ms: np.ndarray) -> float:
    """
    Parameters
    ----------
    rr_ms
        RR-intervals in milliseconds
    """
    return (60.0 / rr_ms).mean() * 1e3


def RMSSD(rr_ms: np.ndarray) -> float:
    """
    Parameters
    ----------
    rr_ms
        RR-intervals in milliseconds
    """
    rr_ms_differances = np.diff(rr_ms)
    rr_ms_differances_squared = rr_ms_differances * rr_ms_differances
    rr_ms_differances_squared_mean = rr_ms_differances_squared.mean()
    return np.sqrt(rr_ms_differances_squared_mean)


def pNN50(rr_ms: np.ndarray) -> float:
    """
    Parameters
    ----------
    rr_ms
        RR-intervals in milliseconds
    """
    rr_ms_differances = np.diff(rr_ms)
    nn50 = np.where(np.abs(rr_ms_differances) > 50.0)[0]
    return len(nn50) / len(rr_ms)


def calculateHistogramStatistics(rr_ms: np.ndarray) -> dict[FeatureName, float]:
    """
    Parameters
    ----------
    rr_ms
        RR-intervals in milliseconds

    Returns
    -------
    dict
        Dictionary with results. Contains keys FeatureName.Mode, FeatureName.Amo50, FeatureName.MxDMn, FeatureName.MxRMn, FeatureName.SI
    """
    rr_min_ms = rr_ms.min()
    rr_max_ms = rr_ms.max()

    hist_values = []
    hist_amounts = []
    for value in range(int(rr_min_ms), int(rr_max_ms), 50):
        hist_amounts.append(np.count_nonzero((rr_ms > value) & (rr_ms < value + 50.0)))
        hist_values.append(value)

    hist_values.append(rr_max_ms + 50.0)

    hist_values = np.array(hist_values)
    hist_amounts = np.array(hist_amounts)

    mode_s = hist_values[hist_amounts.argmax()] * 1e-3
    amo50_p = hist_amounts.max() / hist_amounts.sum() * 100.0
    
    mxdmn_ms = (rr_max_ms - rr_min_ms) * 1e-3
    mxrmn = rr_max_ms / rr_min_ms
    si = amo50_p / (2.0 * mode_s * mxdmn_ms)

    return {FeatureName.Mode: mode_s, FeatureName.Amo50: amo50_p, FeatureName.MxDMn: mxdmn_ms, FeatureName.MxRMn: mxrmn, FeatureName.SI: si}


def interpolate(rr_ms: np.ndarray, delta_time_ms: float = 250.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns interpolation

    Parameters
    ----------
    rr_ms
        RR-intervals in milliseconds
    delta_time_ms
        interpolation time interval
    
    Returns
    -------
    t_ms_interpolated
        interpolated time axis build using RR-intervals values as a references
    rr_ms_interpolated
        interpolated RR-intervals in milliseconds
    """
    current_time = 0.0
    rr_t_ms = []
    for rr in rr_ms:
        rr_t_ms.append(current_time)
        current_time += rr + 0.0001
    
    rr_t_ms = np.array(rr_t_ms)
    rr_t_ms_max = rr_t_ms.max()
    rr_t_ms_interpolated = np.linspace(0.0, rr_t_ms_max, int(rr_t_ms_max // delta_time_ms))

    rr_ms_interpolated = CubicSpline(rr_t_ms, rr_ms)(rr_t_ms_interpolated)
    return rr_t_ms_interpolated, rr_ms_interpolated


def calculateACFStatistics(rr_ms: np.ndarray, rr_ms_interpolated: np.ndarray, interpolation_delta_time_ms: float = 250.0) -> dict[FeatureName, float]:
    """
    Calculates CC1 and CC0 using autocorrelation function

    Parameters
    ----------
    rr_ms
        RR-intervals in milliseconds
    rr_ms_interpolated
        interpolation of RR-intervals in milliseconds
    interpolation_delta_time_ms
        delta time used for interpolation of RR-intervals
    
    Returns
    -------
    dict
        dictionary with keys FeatureName.CC1 - correlation coefficient after the first shift of autocorrelation function
        and FeatureName.CC0 - time to the first zero value of autocorrelation function
    """
    max_shift = int(rr_ms.sum() // interpolation_delta_time_ms)
    acf_data = autocorrelationFunction(rr_ms_interpolated, nlags=max_shift)
    cc1 = acf_data[1]
    first_zero_value_shift = 0
    for acf_value in acf_data:
        if acf_value <= 0.0:
            break
        first_zero_value_shift += 1
    first_zero_value_shift -= 1
    first_zero_value_shift = max(0, first_zero_value_shift)
    cc0 = first_zero_value_shift * interpolation_delta_time_ms * 1e-3
    return {FeatureName.CC1: cc1, FeatureName.CC0: cc0}


def getSpectralRange(fft_freqs: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    """
    Returns indexes of frequences in specific range
    """
    return np.where(np.logical_and(fft_freqs >= fmin, fft_freqs <= fmax))


def calculateSpectralStatistics(rr_ms_interpolated: np.ndarray, interpolation_delta_time_ms: float = 250.0) -> dict[FeatureName, float]:
    """
    Calculates LF-HF spectral ranges statistics

    Parameters
    ----------
    rr_ms_interpolated
        interpolation of RR-intervals in milliseconds
    interpolation_delta_time_ms
        delta time used for interpolation of RR-intervals

    Returns
    -------
    dict
        dictionary with keys FeatureName.LFmax, FeatureName.HFmax - maximum values of spectral powers of LF and HF ranges in ms^2
        and FeatureName.LFHF - value of LF/HF - ratio of LF/HF summary powers
    """
    LF_MIN = 0.04
    LF_MAX = 0.15
    HF_MIN = 0.15
    HF_MAX = 0.4

    fft = np.abs(np.fft.rfft(rr_ms_interpolated))
    N = fft.size * 2 - 1
    frequencies = np.fft.rfftfreq(N, interpolation_delta_time_ms * 1e-3)

    spectral_power = (fft / N)**2

    lf_range = getSpectralRange(frequencies, LF_MIN, LF_MAX)
    hf_range = getSpectralRange(frequencies, HF_MIN, HF_MAX)
    lf = spectral_power[lf_range]
    hf = spectral_power[hf_range]

    lf_power_ms2 = lf.sum()
    hf_power_ms2 = hf.sum()
    
    lf_max_ms2 = lf.max()
    hf_max_ms2 = hf.max()

    lf_hf = lf_power_ms2 / hf_power_ms2

    return {FeatureName.LFmax: lf_max_ms2, FeatureName.HFmax: hf_max_ms2, FeatureName.LFHF: lf_hf}

def raiseOnInfinity(features: dict):
    """Raise exception if any of feature value is infinite"""
    infinite_features = [feature for feature, value in features.items() if np.isinf(value)]
    if infinite_features:
        raise Exception(f"Got infinite value for features {infinite_features}")

def getAllFeatures(raw_ecg, sampling_rate):
    interpolation_delta_time_ms = 250.0
    detector = Detectors(sampling_rate)
    r_indexes = np.array(detector.hamilton_detector(raw_ecg))
    rr_ms = (np.diff(r_indexes) / sampling_rate) * 1e3
    resulting_data = defaultdict(list)
    mean_hr = meanHR(rr_ms)
    raiseOnInfinity({FeatureName.HR: mean_hr})
    rmssd = RMSSD(rr_ms)
    raiseOnInfinity({FeatureName.RMSSD: rmssd})
    pnn50 = pNN50(rr_ms)
    raiseOnInfinity({FeatureName.pNN50: pnn50})
    histogram_statistics = calculateHistogramStatistics(rr_ms)
    raiseOnInfinity(histogram_statistics)
    rr_t_ms, rr_ms_interpolated = interpolate(rr_ms, interpolation_delta_time_ms)
    acf_statistics = calculateACFStatistics(rr_ms, rr_ms_interpolated, interpolation_delta_time_ms)
    raiseOnInfinity(acf_statistics)
    spectral_statistics = calculateSpectralStatistics(rr_ms_interpolated, interpolation_delta_time_ms)
    raiseOnInfinity(spectral_statistics)
    
    resulting_data[FeatureName.HR].append(mean_hr)
    resulting_data[FeatureName.RMSSD].append(rmssd)
    resulting_data[FeatureName.pNN50].append(pnn50)
    resulting_data[FeatureName.Mode].append(histogram_statistics[FeatureName.Mode])
    resulting_data[FeatureName.Amo50].append(histogram_statistics[FeatureName.Amo50])
    resulting_data[FeatureName.MxRMn].append(histogram_statistics[FeatureName.MxRMn])
    resulting_data[FeatureName.MxDMn].append(histogram_statistics[FeatureName.MxDMn])
    resulting_data[FeatureName.SI].append(histogram_statistics[FeatureName.SI])
    resulting_data[FeatureName.CC1].append(acf_statistics[FeatureName.CC1])
    resulting_data[FeatureName.CC0].append(acf_statistics[FeatureName.CC0])
    resulting_data[FeatureName.LFmax].append(spectral_statistics[FeatureName.LFmax])
    resulting_data[FeatureName.HFmax].append(spectral_statistics[FeatureName.HFmax])
    resulting_data[FeatureName.LFHF].append(spectral_statistics[FeatureName.LFHF])
    data_frame = pd.DataFrame(data=resulting_data)
    return data_frame

def preprocessFeatures(features, scaler, pca):
    features = scaler.transform(features)
    features = pca.transform(features)
    return features
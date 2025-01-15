# DataDriftDetector.py
#
# This data drift detector uses the IncrementalKSTest for each feature.
# Drift detection in a significant percentage of features leads to drift
# detection for all data.

import numpy as np
from frouros.detectors.data_drift import IncrementalKSTest

class DataDriftDetector:
    def __init__(self, logger, window_size=100, alpha=0.0001, percent_features=0.6):
        self.log = logger
        self.num_features = 0
        self.window_size = window_size
        self.detectors = None
        self.alpha = alpha
        self.percent_features = percent_features # Percent of features drifting for global drift
        self.drift_flags = None
        self.drift = False
        self.count = 0
        return
    
    def fit(self, data):
        """Initialize and fit each IncKSTest for each feature in data. The data is a list of NumPy arrays."""
        self.num_features = len(data[0])
        self.detectors = [IncrementalKSTest(window_size=self.window_size) for i in range(self.num_features)]
        self.drift_flags = [False] * self.num_features
        for i in range(self.num_features):
            feature_data = np.array([arr[i] for arr in data])
            _ = self.detectors[i].fit(feature_data)
        return
    
    def update(self, feature_arr):
        """Update each IncKSTest with corresponding value from given NumPy feature array.
        Return True if drift detected on this or previous update."""
        self.count += 1
        for i in range(self.num_features):
            result, _ = self.detectors[i].update(feature_arr[i])
            if result:
                p_value = result.p_value
                if (not self.drift_flags[i]) and (p_value <= self.alpha):
                    self.log.info(f'Drift detected in feature {i} at instance {self.count} with p_value={p_value}.')
                    self.drift_flags[i] = True
        count_drifts = sum(self.drift_flags)
        min_drifts = self.num_features * self.percent_features
        if count_drifts > min_drifts:
            self.drift = True
            self.log.info(f'Drift detected in significant number of features ({count_drifts} of {self.num_features}) at instance {self.count}.')
        return self.drift

# ConceptDriftDetector.py
#
# Detects drift using the DDM method based on significant changes in the
# error of a classifier.

from frouros.detectors.concept_drift import DDM, DDMConfig
from frouros.metrics import PrequentialError

class ConceptDriftDetector:
    def __init__(self, logger):
        self.log = logger
        # Detector configuration and instantiation
        config = DDMConfig(
            warning_level=4.0, # default=2.0
            drift_level=6.0, # default=3.0
            min_num_instances=10000,  # default=30
            )
        self.detector = DDM(config=config)
        # Metric to compute accuracy
        self.metric = PrequentialError(alpha=1.0)  # alpha=1.0 is equivalent to normal accuracy
        self.drift_flag = False
        self.count = 0

    def drift(self, error):
        """Check for drift based on latest error signal (0 or 1)."""
        self.count += 1
        metric_error = self.metric(error_value=error)
        _ = self.detector.update(value=error)
        status = self.detector.status
        if status["drift"] and not self.drift_flag:
            self.drift_flag = True
            self.log.info(f"Concept drift detected at instance {self.count}. Accuracy: {1 - metric_error:.4f}")
        return self.drift_flag

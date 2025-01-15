# ************************************************************************************************ #
# **                                                                                            ** #
# **    AIQ-SAIL-ON TA2 Agent Example (Local)                                                   ** #
# **                                                                                            ** #
# **        Brian L Thomas, 2020                                                                ** #
# **                                                                                            ** #
# **  Tools by the AI Lab - Artificial Intelligence Quotient (AIQ) in the School of Electrical  ** #
# **  Engineering and Computer Science at Washington State University.                          ** #
# **                                                                                            ** #
# **  Copyright Washington State University, 2020                                               ** #
# **  Copyright Brian L. Thomas, 2020                                                           ** #
# **                                                                                            ** #
# **  All rights reserved                                                                       ** #
# **  Modification, distribution, and sale of this work is prohibited without permission from   ** #
# **  Washington State University.                                                              ** #
# **                                                                                            ** #
# **  Contact: Brian L. Thomas (bthomas1@wsu.edu)                                               ** #
# **  Contact: Larry Holder (holder@wsu.edu)                                                    ** #
# **  Contact: Diane J. Cook (djcook@wsu.edu)                                                   ** #
# ************************************************************************************************ #

import copy
import optparse
import random
import pickle
import json
import sys

from ActivityLearner import ActivityManager
from ConceptDriftDetector import ConceptDriftDetector
from DataDriftDetector import DataDriftDetector

TESTBED_IDS = ['12bc911e4b7e', '4cdb124c42f4', '72c576abcdda']

TESTBED_SAMPLE_FEATURE_VECTORS = {
    '12bc911e4b7e': {'image': None, 'testbed_id': '12bc911e4b7e', 'time_stamp': 1310231785.895927, 'door_sensors': [{'id': 'D006', 'value': 0.0}, {'id': 'D002', 'value': 0.0}, {'id': 'D003', 'value': 0.0}, {'id': 'D004', 'value': 1.0}, {'id': 'D005', 'value': 0.0}, {'id': 'D001', 'value': 0.0}], 'motion_sensors': [{'id': 'M021', 'value': 0.0}, {'id': 'M003', 'value': 0.0}, {'id': 'M004', 'value': 0.0}, {'id': 'M014', 'value': 0.0}, {'id': 'M020', 'value': 0.0}, {'id': 'M007', 'value': 0.0}, {'id': 'M013', 'value': 0.0}, {'id': 'M006', 'value': 0.0}, {'id': 'M016', 'value': 0.0}, {'id': 'M010', 'value': 0.0}, {'id': 'M017', 'value': 0.0}, {'id': 'M018', 'value': 0.0}, {'id': 'M002', 'value': 0.0}, {'id': 'M011', 'value': 0.0}, {'id': 'M001', 'value': 0.0}, {'id': 'M008', 'value': 0.0}, {'id': 'M005', 'value': 0.0}, {'id': 'M012', 'value': 0.0}, {'id': 'M009', 'value': 0.0}, {'id': 'M015', 'value': 0.0}], 'motion_area_sensors': [{'id': 'MA022', 'value': 0.0}, {'id': 'MA024', 'value': 1.0}, {'id': 'MA023', 'value': 0.0}, {'id': 'MA019', 'value': 0.0}], 'light_switch_sensors': [{'id': 'L003', 'value': 0.0}, {'id': 'L007', 'value': 1.0}, {'id': 'L005', 'value': 1.0}, {'id': 'L006', 'value': 1.0}, {'id': 'L009', 'value': 0.0}, {'id': 'L001', 'value': 1.0}, {'id': 'L004', 'value': 1.0}, {'id': 'L002', 'value': 0.0}]},
    '4cdb124c42f4': {'image': None, 'testbed_id': '4cdb124c42f4', 'time_stamp': 1308211383.759684, 'door_sensors': [{'id': 'D005', 'value': 0.0}, {'id': 'D004', 'value': 0.0}, {'id': 'D003', 'value': 0.0}, {'id': 'D002', 'value': 0.0}], 'motion_sensors': [{'id': 'M015', 'value': 0.0}, {'id': 'M013', 'value': 0.0}, {'id': 'M003', 'value': 0.0}, {'id': 'M010', 'value': 0.0}, {'id': 'M009', 'value': 0.0}, {'id': 'M006', 'value': 0.0}, {'id': 'M007', 'value': 0.0}, {'id': 'M011', 'value': 0.0}, {'id': 'M004', 'value': 0.0}, {'id': 'M014', 'value': 0.0}, {'id': 'M005', 'value': 0.0}], 'motion_area_sensors': [{'id': 'MA008', 'value': 0.0}, {'id': 'MA002', 'value': 0.0}, {'id': 'MA012', 'value': 0.0}, {'id': 'MA001', 'value': 0.0}], 'light_switch_sensors': [{'id': 'L003', 'value': 0.0}, {'id': 'L002', 'value': 0.0}, {'id': 'L009', 'value': 0.0}, {'id': 'L006', 'value': 0.0}, {'id': 'L004', 'value': 0.0}, {'id': 'L007', 'value': 0.0}, {'id': 'L005', 'value': 0.0}, {'id': 'L008', 'value': 0.0}, {'id': 'L001', 'value': 0.0}]},
    '72c576abcdda': {'image': None, 'testbed_id': '72c576abcdda', 'time_stamp': 1318060196.102204, 'door_sensors': [{'id': 'D003', 'value': 1.0}, {'id': 'D001', 'value': 0.0}, {'id': 'D005', 'value': 1.0}, {'id': 'D004', 'value': 0.0}, {'id': 'D002', 'value': 0.0}], 'motion_sensors': [{'id': 'M016', 'value': 0.0}, {'id': 'M020', 'value': 0.0}, {'id': 'M009', 'value': 0.0}, {'id': 'M007', 'value': 0.0}, {'id': 'M021', 'value': 0.0}, {'id': 'M022', 'value': 0.0}, {'id': 'M015', 'value': 0.0}, {'id': 'M013', 'value': 0.0}, {'id': 'M004', 'value': 0.0}, {'id': 'M017', 'value': 0.0}, {'id': 'M002', 'value': 0.0}, {'id': 'M006', 'value': 0.0}, {'id': 'M001', 'value': 0.0}, {'id': 'M011', 'value': 0.0}, {'id': 'M010', 'value': 0.0}, {'id': 'M003', 'value': 0.0}, {'id': 'M012', 'value': 0.0}, {'id': 'M008', 'value': 0.0}], 'motion_area_sensors': [{'id': 'MA014', 'value': 1.0}, {'id': 'MA005', 'value': 0.0}, {'id': 'MA018', 'value': 0.0}, {'id': 'MA019', 'value': 0.0}], 'light_switch_sensors': [{'id': 'L002', 'value': 0.97}, {'id': 'L004', 'value': 1.0}, {'id': 'L001', 'value': 1.0}, {'id': 'L007', 'value': 0.0}, {'id': 'L003', 'value': 0.0}, {'id': 'L006', 'value': 1.0}, {'id': 'L005', 'value': 0.41}]}
}

class SmartEnvAgent:
    def __init__(self, logger, novelty_detection = None):

        self.log = logger
        self.novelty_detection = novelty_detection # can be None, CDD, or DDD

        # This variable can be set to true and the system will attempt to end training at the
        # completion of the current episode, or sooner if possible.
        self.end_training_early = True

        # This variable is checked only during the evaluation phase.  If set to True the system
        # will attempt to cleanly end the experiment at the conclusion of the current episode,
        # or sooner if possible.
        self.end_experiment_early = False

        # If you need values from the command line, you can get values from your custom options
        # here.  Set custom options in the _add_ta2_command_line_options() function.
        #options = self._get_command_line_options()
        #my_custom_value = options.custom_value
        #self.log.debug('Command line custom value is: {}'.format(my_custom_value))

        # Initialize any needed objects
        self.AL = ActivityManager(TESTBED_SAMPLE_FEATURE_VECTORS)
        self.concept_drift_detectors = {}
        self.data_drift_detectors = {}
        self.current_feature_vector = None
        self.new_data = False
        self.total_new_data = 0
        self.last_performance = None
        self.novelty_detected = False
        self.episode_number = None
        self.trial_number = None

        return

    def _add_ta2_command_line_options(self, parser: optparse.OptionParser):
        """If you do not want to use this function, you can remove it from TA2.py to clean up
        your code.  This is already defined in the parent class.

        This function allows you to easily add custom arguments to the command line parser.  To
        see what is already defined, please see the _add_command_line_options() function in the
        parent class found in options/TA2_logic.py.

        Parameters
        ----------
        parser : optparse.OptionParser
            This is the command line parser object, you can add custom entries to it here.

        Returns
        -------
        optparse.OptionParser
            The parser object that you have added additional options to.
        """

        # TODO: Maybe a novelty probability threshold, or concept drift method selection and parameters

        parser.add_option("--custom-value",
                          dest="custom_value",
                          help="Example for adding custom options to the command line parser.",
                          default="HelloWorld!")
        return parser

    def train_model(self):
        """Train your model here if needed.  If you don't need to train, just leave the function
        empty.  After this completes, the logic calls save_model() and reset_model() as needed
        throughout the rest of the experiment.
        """
        self.log.info('Training model with {}.'.format(self.AL.get_num_examples()))

        self.AL.train_model()
        self.new_data = False

        return

    def save_model(self, filename: str):
        """Saves the current model in memory to disk so it may be loaded back to memory again.

        Parameters
        ----------
        filename : str
            The filename to save the model to.
        """
        self.log.info('Save model to disk.')

        # Careful not to overwrite pre-trained model
        self.AL.save_model(filename)

        return

    def reset_model(self, filename: str):
        """Loads the model from disk to memory.

        Parameters
        ----------
        filename : str
            The filename where the model was stored.
        """
        self.log.info('Load model from disk.')

        self.AL.reset_model(filename)
        self.new_data = False

        # Initialize drift detectors
        if self.novelty_detection:
            for testbed_id in TESTBED_IDS:
                if self.novelty_detection == 'CDD':
                    self.concept_drift_detectors[testbed_id] = ConceptDriftDetector(self.log)
                if self.novelty_detection == 'DDD':
                    self.data_drift_detectors[testbed_id] = DataDriftDetector(self.log)
                    AL_testbed = self.AL.test_beds[testbed_id]
                    self.data_drift_detectors[testbed_id].fit(AL_testbed.feature_arr_data)

        return

    def experiment_start(self):
        """This function is called when this TA2 has connected to a TA1 and is ready to begin
        the experiment.
        """
        self.log.info('Experiment Start')
        return

    def trial_start(self, trial_number: int, novelty_description: dict):
        """This is called at the start of a trial with the current 0-based number.

        Parameters
        ----------
        trial_number : int
            This is the 0-based trial number in the novelty group.
        novelty_description : dict
            A dictionary that will have a description of the trial's novelty.
        """
        self.log.info('Trial Start: #{}  novelty_desc: {}'.format(trial_number,
                                                                  str(novelty_description)))
        self.trial_number = trial_number
        return

    def testing_start(self):
        """This is called after a trial has started but before we begin going through the
        episodes.
        """
        self.log.info('Testing Start')
        return

    def testing_episode_start(self, episode_number: int):
        """This is called at the start of each testing episode in a trial, you are provided the
        0-based episode number.

        Parameters
        ----------
        episode_number : int
            This is the 0-based episode number in the current trial.
        """
        self.log.info('Testing Episode Start: #{}'.format(episode_number))

        self.episode_number = episode_number

        return

    def testing_instance(self, feature_vector: dict, novelty_indicator: bool = None) -> dict:
        """Evaluate a testing instance.  Returns the predicted label or action, if you believe
        this episode is novel, and what novelty level you beleive it to be.

        Parameters
        ----------
        feature_vector : dict
            The dictionary containing the feature vector.  Domain specific feature vectors are
            defined on the github (https://github.com/holderlb/WSU-SAILON-NG).
        novelty_indicator : bool, optional
            An indicator about the "big red button".
                - True == novelty has been introduced.
                - False == novelty has not been introduced.
                - None == no information about novelty is being provided.

        Returns
        -------
        dict
            A dictionary of your label prediction of the format {'action': label}.  This is
                strictly enforced and the incorrect format will result in an exception being thrown.
        """
        #self.log.debug('Testing Instance: feature_vector={}, novelty_indicator={}'.format(
        #    feature_vector, novelty_indicator))
        
        self.current_feature_vector = feature_vector

        label_prediction = self.AL.predict(feature_vector)

        return label_prediction

    def testing_performance(self, performance: float, feedback: dict = None):
        """Provides the current performance on episode after each instance.

        Parameters
        ----------
        performance : float
            The normalized performance score.
        feedback : dict, optional
            A dictionary that may provide additional feedback on your prediction based on the
            budget set in the TA1. If there is no feedback, the object will be None.
        """
        #self.log.debug('Testing Performance: {}'.format(performance))
        #if feedback is not None:
        #    self.log.debug('Testing Feedback: {}'.format(str(feedback)))

        testbed_id = self.current_feature_vector['testbed_id']
        if self.novelty_detection:

            # Concept drift detection (CDD)
            if self.novelty_detection == 'CDD':
                if self.last_performance and (not self.novelty_detected):
                    error = 0
                    if performance < self.last_performance:
                        error = 1
                    self.novelty_detected = self.concept_drift_detectors[testbed_id].drift(error)
                    if self.novelty_detected:
                        self.log.info('Novelty (concept drift) detected at episode {} in trial {}'.format(self.episode_number, self.trial_number))
            
            # Data drift detection (DDD)
            if self.novelty_detection == 'DDD':
                if (not self.novelty_detected):
                    feature_arr = self.AL.test_beds[testbed_id].get_feature_array() # feature array from last prediction
                    #self.log.debug(str(feature_arr))
                    self.novelty_detected = self.data_drift_detectors[testbed_id].update(feature_arr)
                    if self.novelty_detected:
                        self.log.info('Novelty (data drift) detected at episode {} in trial {}'.format(self.episode_number, self.trial_number))

        self.last_performance = performance

        # If feedback contains correct activity, then store as training example.
        if feedback is not None:
            if 'action' in feedback:
                feature_label = {'action': feedback['action']}
                self.AL.test_beds[testbed_id].add_data(feature_label)
                self.new_data = True
                self.total_new_data += 1
    
        return

    def testing_episode_end(self, performance: float, feedback: dict = None) -> (float, float, int, dict): # type: ignore
        """Provides the final performance on the testing episode.

        Parameters
        ----------
        performance : float
            The final normalized performance score of the episode.
        feedback : dict, optional
            A dictionary that may provide additional feedback on your prediction based on the
            budget set in the TA1. If there is no feedback, the object will be None.

        Returns
        -------
        float, float, int, dict
            A float of the probability of there being novelty.
            A float of the probability threshold for this to evaluate as novelty detected.
            Integer representing the predicted novelty level.
            A JSON-valid dict characterizing the novelty.
        """
        self.log.info('Testing Episode End: performance={}, total new data = {}'.format(performance, self.total_new_data))
        if feedback is not None:
            self.log.debug('Testing Feedback: {}'.format(str(feedback)))

        # Retrain model on all available training examples
        if self.new_data:
            self.train_model()
            self.new_data = False

        # Check if novelty detected (TODO: determine novelty level and characterization)
        novelty_probability = 0.0
        novelty_threshold = 0.5
        novelty = 200 # i.e., no novelty
        novelty_characterization = dict()
        if self.novelty_detected:
            novelty_probability = 1.0
            novelty = 201 # i.e., novelty level 1 (placeholder)

        return novelty_probability, novelty_threshold, novelty, novelty_characterization

    def testing_end(self):
        """This is called after the last episode of a trial has completed, before trial_end().
        """
        self.log.info('Testing End')
        return

    def trial_end(self):
        """This is called at the end of each trial.
        """
        self.log.info('Trial End')
        return

    def experiment_end(self):
        """This is called when the experiment is done.
        """
        self.log.info('Experiment End')
        return

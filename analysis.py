#!/usr/bin/env python3
# ************************************************************************************************ #
# **                                                                                            ** #
# **    AIQ-SAIL-ON Analysis Core Logic                                                         ** #
# **                                                                                            ** #
# **        Robbie Stancil, 2021                                                                ** #
# **        Brian L Thomas, 2021                                                                ** #
# **        Larry Holder, 2021                                                                  ** #
# **                                                                                            ** #
# **  Tools by the AI Lab - Artificial Intelligence Quotient (AIQ) in the School of Electrical  ** #
# **  Engineering and Computer Science at Washington State University.                          ** #
# **                                                                                            ** #
# **  Copyright Washington State University, 2021                                               ** #
# **  Copyright Brian L. Thomas, 2021                                                           ** #
# **                                                                                            ** #
# **  All rights reserved                                                                       ** #
# **  Modification, distribution, and sale of this work is prohibited without permission from   ** #
# **  Washington State University.                                                              ** #
# **                                                                                            ** #
# **  Contact: Brian L. Thomas (bthomas1@wsu.edu)                                               ** #
# **  Contact: Larry Holder (holder@wsu.edu)                                                    ** #
# **  Contact: Diane J. Cook (djcook@wsu.edu)                                                   ** #
# ************************************************************************************************ #

import csv
import os
import optparse
import json
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from decimal import Decimal
import logging

NOVELTY_200 = 200
DIFFICULTY_EASY = 'easy'
DETECTION_KNOWN = 'known'
DETECTION_UNKNOWN = 'unknown'
DETECTION_TRANSLATION = dict({0: DETECTION_UNKNOWN, 1: DETECTION_KNOWN})
FLOAT_NEAR_ZERO = 0.00001 # used as placeholder for zero denominators

class Analysis():
    def __init__(self, options):
        self.experiment_id = options.experiment_id
        self.output_directory = options.output_directory
        self.printout=options.printout
        self.debug=options.debug
        self.logfile=options.logfile
        self.plottrials = options.plottrials
        self.plotamocs = options.plotamocs
        self.details = options.details
        self.ta1_team_name = options.ta1_team_name
        self.ta2_team_name = options.ta2_team_name
        self.domain_name = options.domain_name
        self.use_possible_novelties = options.use_novelty_list
        self.use_possible_difficulties = options.use_difficulty_list
        self.use_visibility_list = True
        self.possible_novelties = list([200, 201, 202, 203, 204, 205, 206, 207, 208])
        self.possible_difficulties = list(['easy', 'medium', 'hard'])
        self.detection_conditions = list(['known', 'unknown'])
        self.extra_per_trial_metrics = options.extras # flag to print extra per trial metrics (for internal use)

        # Local baseline results (LBH)
        self.baseline_results_df = None 
        baseline_results_file = options.baseline_results_file
        if baseline_results_file:
            print(f'Reading baseline results from local file: {baseline_results_file}')
            self.baseline_results_df = pd.read_csv(baseline_results_file)
        
        # Local agent results (LBH)
        self.agent_results_df = None 
        agent_results_file = options.agent_results_file
        if agent_results_file:
            print(f'Reading agent results from local file: {agent_results_file}')
            self.agent_results_df = pd.read_csv(agent_results_file)

        # Setup logging
        file_handler = logging.FileHandler(self.logfile)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.log = logging.getLogger('analysis')
        log_level = logging.WARNING
        if self.printout:
            log_level = logging.INFO
        if self.debug:
            log_level = logging.DEBUG
        self.log.setLevel(log_level)
        self.log.addHandler(file_handler)        

        return

    def analyze_experiment(self):
        # Dynamically build the lists for self.possible_novelties, self.possible_difficulties,
        # and self.detection_conditions from the experiment structure in the database.
        # Populate self.possible_novelties if the command line arg was not passed.
        if not self.use_possible_novelties:
            self.possible_novelties = self.get_experiment_novelties()
        # Populate self.possible_difficulties if the command line arg was not passed.
        if not self.use_possible_difficulties:
            self.possible_difficulties = self.get_experiment_difficulties()
        # Populate self.detection_conditions if the command line arg was not passed.
        if not self.use_visibility_list:
            self.detection_conditions = self.get_experiment_detection_conditions()

        # Get trial ids - These are stored in a dictionary with keys
        # corresponding to each combination of novelty_difficulty for each configuration
        ta2_experiment_trials = self.get_experiment_trials(self.agent_results_df)
        sota_experiment_trials = self.get_experiment_trials(self.baseline_results_df)
        ta2_known_trial_ids, ta2_unknown_trial_ids, sota_known_trial_ids, \
            sota_unknown_trial_ids = self.get_experiment_trial_ids(ta2_experiment_trials,
                                                                    sota_experiment_trials)

        # Get all data needed from episodes corresponding to all trial ids
        ta2_known_episode_data = self.get_episode_data(ta2_known_trial_ids, self.agent_results_df) 
        ta2_unknown_episode_data = self.get_episode_data(ta2_unknown_trial_ids, self.agent_results_df)
        sota_known_episode_data = self.get_episode_data(sota_known_trial_ids, self.baseline_results_df)
        sota_unknown_episode_data = self.get_episode_data(sota_unknown_trial_ids, self.baseline_results_df)

        # Get novelty introduction indices for each episode as well as CDT info
        ta2_known_novelty_introduced_indices, _, _, _, _, ta2_known_true_negative_counts, _ = \
            self.get_novelty_introduced_indices(ta2_known_episode_data)
        (ta2_unknown_novelty_introduced_indices,
            ta2_unknown_false_positive_counts,
            #  ta2_unknown_true_positive_counts,
            _,
            ta2_unknown_first_true_positive_indices,
            ta2_unknown_false_negative_counts,
            ta2_unknown_true_negative_counts,
            ta2_unknown_correctly_detected_trial) = self.get_novelty_introduced_indices(
            ta2_unknown_episode_data, True)

        # Generate plots of each trial
        path = os.path.join(self.output_directory,
                            'experiment_{0}'.format(self.experiment_id))
        if not os.path.exists(path):
            os.makedirs(path)

        if self.plottrials:
            self.generate_trial_plots(ta2_known_episode_data, sota_known_episode_data,
                                        ta2_known_novelty_introduced_indices, path, 'Known')
            self.generate_trial_plots(ta2_unknown_episode_data, sota_unknown_episode_data,
                                        ta2_unknown_novelty_introduced_indices, path, 'Unknown')
        
        if self.plotamocs:
            self.generate_amoc_plots(ta2_unknown_episode_data,
                                        ta2_unknown_novelty_introduced_indices, path, 'Unknown')
        
        # Compute metrics
        # Generate all needed combinations of the data
        ta2_unknown_episode_data = self.generate_remaining_groups(
            ta2_unknown_episode_data)
        sota_unknown_episode_data = self.generate_remaining_groups(
            sota_unknown_episode_data)
        ta2_unknown_correctly_detected_trial = self.generate_remaining_groups(
            ta2_unknown_correctly_detected_trial)
        ta2_unknown_false_negative_counts = self.generate_remaining_groups(
            ta2_unknown_false_negative_counts)
        ta2_unknown_false_positive_counts = self.generate_remaining_groups(
            ta2_unknown_false_positive_counts)
        ta2_unknown_true_negative_counts = self.generate_remaining_groups(
            ta2_unknown_true_negative_counts)
        ta2_unknown_novelty_introduced_indices = self.generate_remaining_groups(
            ta2_unknown_novelty_introduced_indices)
        ta2_unknown_first_true_positive_indices = self.generate_remaining_groups(
            ta2_unknown_first_true_positive_indices)

        ta2_known_episode_data = self.generate_remaining_groups(
            ta2_known_episode_data)
        sota_known_episode_data = self.generate_remaining_groups(
            sota_known_episode_data)
        ta2_known_true_negative_counts = self.generate_remaining_groups(
            ta2_known_true_negative_counts)
        ta2_known_novelty_introduced_indices = self.generate_remaining_groups(
            ta2_known_novelty_introduced_indices)

        ta2_all_episode_data = ta2_unknown_episode_data['all'] + \
            ta2_known_episode_data['all']
        sota_all_episode_data = sota_unknown_episode_data['all'] + \
            sota_known_episode_data['all']
        ta2_all_novelty_introduced_indices = ta2_unknown_novelty_introduced_indices[
            'all'] + ta2_known_novelty_introduced_indices[
            'all']

        self.generate_all_plot(ta2_all_episode_data, sota_all_episode_data,
                                ta2_all_novelty_introduced_indices, path)

        # Asymptotic performance value - number of episodes factored into
        # an agent's asymptotic task performance
        # Also use for AM4 (IPTI) initial performance just after novelty introduced
        # LBH: Use 10% of length of trial
        #m = 10
        m = int(0.1 * len(ta2_all_episode_data[0]))
        
        # Unknown case:
        unknown_m1 = dict() # FN_CDT
        unknown_m2 = dict() # CDT_%
        unknown_m2_1 = dict() # FP_%
        
        ### New Metric (TN_%) = #TN / N_pre
        unknown_m2_2 = dict()
        
        unknown_m3 = dict() # NRP system detection, based on asymptotic performance (last 10% of episodes)
        unknown_m4 = dict() # NRP given detection (n/a)
        unknown_m3_1 = dict() # NRP system detection, based on post-novelty performance (first 10% of novel episodes)
        unknown_m4_1 = dict() # NRP given detection (n/a)
        unknown_am1 = dict() # AM1: OPTIs (Overall Performance Task Improvement (PTI), system detection (unknown))
        unknown_am2 = dict() # AM2: APTIs (Asymptotic PTI, system detection (unknown))
        unknown_am3 = dict() # AM3: AMOC
        unknown_am4 = dict() # AM4: IPTIs (Initial PTI, system detection (unknown))
        unknown_nrm = dict() # NRM alpha
        unknown_nrm_beta = dict() # NRM beta/baseline
        
        # Requested metrics: average pre-novelty and post_novelty performance for baseline and TA2
        unknown_pre_sota = dict()
        unknown_pre_ta2 = dict()
        unknown_post_sota = dict()
        unknown_post_ta2 = dict()

        ta2_data = ta2_unknown_episode_data
        sota_data = sota_unknown_episode_data
        ta2_correctly_detected_trial = ta2_unknown_correctly_detected_trial
        ta2_false_negative_counts = ta2_unknown_false_negative_counts
        ta2_false_positive_counts = ta2_unknown_false_positive_counts
        ta2_true_negative_counts = ta2_unknown_true_negative_counts
        ta2_novelty_introduced_indices = ta2_unknown_novelty_introduced_indices
        ta2_first_true_positive_indices = ta2_unknown_first_true_positive_indices
        
        # Collect per trial metrics
        # Dictionary with key trial_id and value dictionary of trial info and metrics
        per_trial_metrics = dict()
        
        for configuration in ta2_data:
            
            detection_source = 'system'
            # Collect per trial metrics for specific configurations, i.e., specific novelty and difficulty
            specific_config = False
            config_split = configuration.split('_')
            if len(config_split) == 2:
                specific_config = True
                for trial, novelty_introduced in zip(ta2_data[configuration], ta2_novelty_introduced_indices[configuration]):
                    trial_dict = dict()
                    trial_dict['ta2'] = self.ta2_team_name
                    trial_dict['novelty_level'] = config_split[0]
                    trial_dict['difficulty'] = config_split[1]
                    trial_dict['detection_source'] = detection_source
                    trial_dict['trial_index'] = trial.iloc[0]['trial_index']
                    trial_dict['num_episodes'] = len(trial['trial_index'])
                    trial_dict['novelty_episode'] = novelty_introduced
                    self.add_characterization(trial_dict, trial)
                    per_trial_metrics[trial.iloc[0]['trial_id']] = trial_dict
                for trial, novelty_introduced in zip(sota_data[configuration], ta2_novelty_introduced_indices[configuration]):
                    trial_dict = dict()
                    trial_dict['ta2'] = 'Baseline'
                    trial_dict['novelty_level'] = config_split[0]
                    trial_dict['difficulty'] = config_split[1]
                    trial_dict['detection_source'] = detection_source
                    trial_dict['trial_index'] = trial.iloc[0]['trial_index']
                    trial_dict['num_episodes'] = len(trial['trial_index'])
                    trial_dict['novelty_episode'] = novelty_introduced
                    self.add_characterization(trial_dict, trial)
                    per_trial_metrics[trial.iloc[0]['trial_id']] = trial_dict
                    
            cdt_indices = np.where(
                ta2_correctly_detected_trial[configuration])[0]
            
            #LBH: with stats
            m1arr = np.array([0])
            if len(cdt_indices) > 0:
                m1arr = np.array(ta2_false_negative_counts[configuration])[cdt_indices]
            unknown_m1[configuration] = self.get_stats(m1arr)
            
            m2 = len(cdt_indices) / len(ta2_correctly_detected_trial[configuration])
            m2_1 = sum(np.array(ta2_false_positive_counts[configuration]) > 0) / \
                len(ta2_false_positive_counts[configuration])
            unknown_m2[configuration] = self.get_stats(np.array([m2]))
            unknown_m2_1[configuration] = self.get_stats(np.array([m2_1]))
                
            ### New Metric (%TN) = #TN / N_pre
            true_negative_counts = np.array(ta2_true_negative_counts[configuration])
            n_pre = np.array([ta2_novelty_introduced_indices[configuration][i]
                                if ta2_novelty_introduced_indices[configuration][i] > 0 else 1
                                for i in range(len(ta2_data[configuration]))])
            unknown_m2_2[configuration] = self.get_stats(true_negative_counts / n_pre)

            # M3 prep
            # Note that M3/M3.1 are not well-defined for all level 0 trials, since everything is pre-novelty
            # So, running this code on level 0 results may result in errors.
            sota_pre_novelty = [sota_data[configuration][i]['performance']
                                [:ta2_novelty_introduced_indices[configuration][i]].tolist()
                                for i in range(len(sota_data[configuration]))]
            ta2_post_novelty_m = [ta2_data[configuration][i]['performance'][-m:]
                                    .tolist() for i in
                                    range(len(ta2_data[configuration]))]
            sota_pre_novelty = self.to_float(sota_pre_novelty)
            ta2_post_novelty_m = self.to_float(ta2_post_novelty_m)

            # Preventing dividing by zero by setting p_pre_beta to be small if = 0 (This should
            # never happen but just in case)
            p_pre_beta = [float(np.mean(sota_pre_novelty[i])) if np.mean(sota_pre_novelty[i]) > 0
                            else FLOAT_NEAR_ZERO for i in range(len(sota_pre_novelty))]
            p_post_alpha_means = [float(np.mean(ta2_post_novelty_m[i]))
                            for i in range(len(ta2_post_novelty_m))]

            #unknown_m3[configuration] = np.mean(
            #    [p_post_alpha_means[i] / p_pre_beta[i] for i in range(len(p_post_alpha_means))])
            m3_values = [p_post_alpha_means[i] / p_pre_beta[i] for i in range(len(p_post_alpha_means))]
            unknown_m3[configuration] = self.get_stats(m3_values)
            
            # M3.1
            ta2_post_novelty_first_m = [ta2_data[configuration][i]['performance']
                                        [ta2_novelty_introduced_indices[configuration][i]:ta2_novelty_introduced_indices[configuration][i]+m]
                                    .tolist() for i in
                                    range(len(ta2_data[configuration]))]
            ta2_post_novelty_first_m = self.to_float(ta2_post_novelty_first_m)

            p_post_first_alpha_means = [float(np.mean(ta2_post_novelty_first_m[i]))
                            for i in range(len(ta2_post_novelty_first_m))]
            m3_1_values = [p_post_first_alpha_means[i] / p_pre_beta[i] for i in range(len(p_post_first_alpha_means))]
            unknown_m3_1[configuration] = self.get_stats(m3_1_values)
            
            if specific_config:
                for trial,false_negative_count,false_positive_count,m3,m3_1 in zip(ta2_data[configuration],
                                                                                ta2_false_negative_counts[configuration],
                                                                                ta2_false_positive_counts[configuration],
                                                                                m3_values,m3_1_values):
                    trial_dict = per_trial_metrics[trial.iloc[0]['trial_id']]
                    trial_num = trial_dict['trial_index']
                    if trial_num in cdt_indices:
                        trial_dict['m1'] = false_negative_count
                        trial_dict['m2'] = 1
                    else:
                        trial_dict['m1'] = 0
                        trial_dict['m2'] = 0
                    if false_positive_count > 0:
                        trial_dict['m2_1'] = 1
                    else: trial_dict['m2_1'] = 0
                    trial_dict['m3'] = m3
                    trial_dict['m3_1'] = m3_1
                for trial in sota_data[configuration]:
                    trial_dict = per_trial_metrics[trial.iloc[0]['trial_id']]
                    trial_num = trial_dict['trial_index']
                    trial_dict['m1'] = 0
                    trial_dict['m2'] = 0
                    trial_dict['m2_1'] = 0
                    trial_dict['m3'] = 0
                    trial_dict['m3_1'] = 0
            
            # M3 is for the unknown case, M4 is for the known case so set it to be blank
            # unknown_m4[configuration] = '-'
            unknown_m4[configuration] = '-,-,-,-,-,-' # LBH: with stats
            unknown_m4_1[configuration] = '-,-,-,-,-,-'

            sota_post_novelty = [sota_data[configuration][i]['performance']
                                    [ta2_novelty_introduced_indices[configuration][i]:].tolist()
                                    for i in range(len(sota_data[configuration]))]
            ta2_post_novelty = [ta2_data[configuration][i]['performance']
                                [ta2_novelty_introduced_indices[configuration][i]:].tolist()
                                for i in range(len(ta2_data[configuration]))]
            sota_post_novelty = self.to_float(sota_post_novelty)
            ta2_post_novelty = self.to_float(ta2_post_novelty)

            # AM1: OPTIs
            am1_values = [float(sum(ta2_post_novelty[i])) / (float(sum(sota_post_novelty[i])) + float(sum(ta2_post_novelty[i])))
                    if (float(sum(sota_post_novelty[i])) + float(sum(ta2_post_novelty[i]))) > 0 else 0
                    for i in range(len(sota_post_novelty))]
            unknown_am1[configuration] = self.get_stats(am1_values)

            sota_post_novelty_m = [sota_data[configuration][i]['performance'][-m:]
                                    .tolist() for i in range(len(sota_data[configuration]))]
            sota_post_novelty_m = self.to_float(sota_post_novelty_m)
            p_post_alpha = [sum(ta2_post_novelty_m[i])
                            for i in range(len(ta2_post_novelty_m))]
            p_post_beta = [sum(sota_post_novelty_m[i]) if sum(sota_post_novelty_m[i]) > 0
                            else FLOAT_NEAR_ZERO for i in range(len(sota_post_novelty_m))]

            # AM2: APTIs (old, un-normalized)
            #unknown_am2[configuration] = self.get_stats( #np.mean( # LBH: with stats
            #    [p_post_alpha[i] / p_post_beta[i] for i in range(len(p_post_alpha))])
            
            # AM2: APTIs (new, normalized)
            am2_values = [(float(p_post_alpha[i]) / (float(p_post_alpha[i]) + float(p_post_beta[i])))
                    if (float(p_post_alpha[i]) + float(p_post_beta[i])) > 0 else 0
                    for i in range(len(p_post_alpha))]
            unknown_am2[configuration] = self.get_stats(am2_values)
            
            # AM4: IPTIs
            sota_post_novelty_first_m = [sota_data[configuration][i]['performance']
                                        [ta2_novelty_introduced_indices[configuration][i]:ta2_novelty_introduced_indices[configuration][i]+m]
                                    .tolist() for i in
                                    range(len(sota_data[configuration]))]
            p_post_alpha_first_m = [sum(ta2_post_novelty_first_m[i])
                            for i in range(len(ta2_post_novelty_first_m))]
            p_post_beta_first_m = [sum(sota_post_novelty_first_m[i]) if sum(sota_post_novelty_first_m[i]) > 0
                            else FLOAT_NEAR_ZERO for i in range(len(sota_post_novelty_first_m))]
            am4_values = [(float(p_post_alpha_first_m[i]) / (float(p_post_alpha_first_m[i]) + float(p_post_beta_first_m[i])))
                    if (float(p_post_alpha_first_m[i]) + float(p_post_beta_first_m[i])) > 0 else 0
                    for i in range(len(p_post_alpha_first_m))]
            unknown_am4[configuration] = self.get_stats(am4_values)
            
            if specific_config:
                for trial,am1,am2,am4 in zip(ta2_data[configuration],am1_values,am2_values,am4_values):
                    trial_dict = per_trial_metrics[trial.iloc[0]['trial_id']]
                    trial_dict['am1'] = am1
                    trial_dict['am2'] = am2
                    trial_dict['am4'] = am4
                for trial in sota_data[configuration]:
                    trial_dict = per_trial_metrics[trial.iloc[0]['trial_id']]
                    trial_dict['am1'] = 0
                    trial_dict['am2'] = 0
                    trial_dict['am4'] = 0
            
            # am3 (AM3:AU-AMOC) version in TA3 slides from 11/17/2020
            # Get AMOC points and compute area under curve for each trial
            ta2_am3s = []
            num_trials = len(ta2_data[configuration])
            for trial_index in range(num_trials):
                amoc_points = self.AMOC(ta2_data[configuration][trial_index],
                                        ta2_novelty_introduced_indices[configuration][trial_index])
                xs = [fs[0] for fs in amoc_points]
                ys = [fs[1] for fs in amoc_points]
                ta2_am3s.append(np.trapz(y=ys, x=xs)) # trapz = area under (x,y) curve
            unknown_am3[configuration] = self.get_stats(ta2_am3s)
            
            # nrm (NRM alpha) version in TA3 slides from 05/29/2021               
            ta2_pre_novelty = [ta2_data[configuration][i]['performance']
                                [:ta2_novelty_introduced_indices[configuration][i]].tolist()
                                for i in range(len(ta2_data[configuration]))]
            ta2_post_means = [float(np.mean(ta2_post_novelty[i]))
                                if len(ta2_post_novelty[i]) > 0 else 0
                                for i in range(len(ta2_data[configuration]))]
            ta2_pre_means = [float(np.mean(ta2_pre_novelty[i]))
                                if len(ta2_pre_novelty[i]) > 0 else 0
                                for i in range(len(ta2_data[configuration]))]
            ta2_abs_diffs = np.abs(np.subtract(ta2_post_means, ta2_pre_means))
            ta2_pre_stds = np.array([float(np.std(ta2_pre_novelty[i])) if np.std(ta2_pre_novelty[i]) > 0
                            else FLOAT_NEAR_ZERO for i in range(len(ta2_data[configuration]))])
            nrms = ta2_abs_diffs / ta2_pre_stds
            nrm = sum(nrms < 2) / len(nrms)
            unknown_nrm[configuration] = self.get_stats(np.array([nrm]))
            
            if specific_config:
                for trial,nrm in zip(ta2_data[configuration],nrms):
                    trial_dict = per_trial_metrics[trial.iloc[0]['trial_id']]
                    if nrm < 2:
                        trial_dict['nrm'] = 1
                    else:
                        trial_dict['nrm'] = 0
            
            # NRM beta/baseline
            sota_post_means = [np.mean(sota_post_novelty[i])
                                if len(sota_post_novelty[i]) > 0 else 0
                                for i in range(len(sota_data[configuration]))]
            sota_post_means = self.to_float(sota_post_means)
            sota_pre_means = [np.mean(sota_pre_novelty[i])
                                if len(sota_pre_novelty[i]) > 0 else 0
                                for i in range(len(sota_data[configuration]))]
            sota_pre_means = self.to_float(sota_pre_means)
            sota_abs_diffs = np.abs(np.subtract(sota_post_means, sota_pre_means))
            sota_pre_stds = np.array([np.std(sota_pre_novelty[i]) if np.std(sota_pre_novelty[i]) > 0
                            else FLOAT_NEAR_ZERO for i in range(len(sota_data[configuration]))])
            sota_abs_diffs = sota_abs_diffs.astype(float)
            sota_pre_stds = sota_pre_stds.astype(float)
            nrms_beta = sota_abs_diffs / sota_pre_stds
            nrm_beta = sum(nrms_beta) / len(nrms_beta)
            unknown_nrm_beta[configuration] = self.get_stats(np.array([nrm_beta]))
            
            if specific_config:
                for trial,nrm_beta in zip(sota_data[configuration],nrms_beta):
                    trial_dict = per_trial_metrics[trial.iloc[0]['trial_id']]
                    if nrm_beta < 2:
                        trial_dict['nrm'] = 1
                    else:
                        trial_dict['nrm'] = 0
        
            # Extra metrics
            unknown_pre_sota[configuration] = self.get_stats(sota_pre_means)
            unknown_pre_ta2[configuration] = self.get_stats(ta2_pre_means)
            unknown_post_sota[configuration] = self.get_stats(sota_post_means)
            unknown_post_ta2[configuration] = self.get_stats(ta2_post_means)
            
            if specific_config and self.extra_per_trial_metrics:
                sota_am3s = []
                num_trials = len(sota_data[configuration])
                for trial_index in range(num_trials):
                    amoc_points = self.AMOC(sota_data[configuration][trial_index],
                                        ta2_novelty_introduced_indices[configuration][trial_index]) # yes, ta2 (avoids computing for sota, which should be the same)
                    xs = [fs[0] for fs in amoc_points]
                    ys = [fs[1] for fs in amoc_points]
                    sota_am3s.append(np.trapz(y=ys, x=xs)) # trapz = area under (x,y) curve
                sota_post_novelty_first_m = [sota_data[configuration][i]['performance']
                                                [ta2_novelty_introduced_indices[configuration][i]:ta2_novelty_introduced_indices[configuration][i]+m] # yes, ta2 (avoids computing for sota, which should be the same)
                                                .tolist() for i in
                                                range(len(sota_data[configuration]))]
                p_post_first_beta_means = [np.mean(sota_post_novelty_first_m[i])
                                            for i in range(len(sota_post_novelty_first_m))]
                p_post_beta_means = [np.mean(sota_post_novelty_m[i])
                                        for i in range(len(sota_post_novelty_m))]
                for trial,au_amoc,pre_perf_all,post_perf_all,post_perf_first_m,post_perf_last_m in \
                        zip(ta2_data[configuration],
                            ta2_am3s,
                            ta2_pre_means,
                            ta2_post_means,
                            p_post_first_alpha_means,
                            p_post_alpha_means):
                    trial_dict = per_trial_metrics[trial.iloc[0]['trial_id']]
                    trial_dict['au_amoc'] = au_amoc
                    trial_dict['pre_perf_all'] = pre_perf_all
                    trial_dict['post_perf_all'] = post_perf_all
                    trial_dict['post_perf_first_m'] = post_perf_first_m
                    trial_dict['post_perf_last_m'] = post_perf_last_m
                for trial,au_amoc,pre_perf_all,post_perf_all,post_perf_first_m,post_perf_last_m in \
                        zip(sota_data[configuration],
                            sota_am3s,
                            sota_pre_means,
                            sota_post_means,
                            p_post_first_beta_means,
                            p_post_beta_means):
                    trial_dict = per_trial_metrics[trial.iloc[0]['trial_id']]
                    trial_dict['au_amoc'] = au_amoc
                    trial_dict['pre_perf_all'] = pre_perf_all
                    trial_dict['post_perf_all'] = post_perf_all
                    trial_dict['post_perf_first_m'] = post_perf_first_m
                    trial_dict['post_perf_last_m'] = post_perf_last_m

        # known case:
        known_m1 = dict() # FN_CDT (n/a)
        known_m2 = dict() # CDT_% (n/a)
        known_m2_1 = dict() # FP_% (n/a)
        
        ### New Metric (TN_%) = TN / N_pre
        known_m2_2 = dict()
        
        known_m3 = dict() # NRP system detection (n/a)
        known_m4 = dict() # NRP given detection, based on asymptotic performance (last 10% of episodes)
        known_m3_1 = dict() # NRP system detection (n/a)
        known_m4_1 = dict() # NRP given detection, based on post-novelty performance (first 10% of novel episodes)
        known_am1 = dict() # AM1: OPTIg (Overall Performance Task Improvement (PTI), given detection (known))
        known_am2 = dict() # AM2: APTIg (Asymptotic PTI, given detection (known))
        known_am3 = dict() # AM3: AMOC
        known_am4 = dict() # AM4: IPTIg (Initial PTI, given detection (known))
        known_nrm = dict() # NRM alpha
        known_nrm_beta = dict() # NRM beta/baseline
        
        # Requested metrics: average pre-novelty and post-novelty performance for baseline and TA2
        known_pre_sota = dict()
        known_pre_ta2 = dict()
        known_post_sota = dict()
        known_post_ta2 = dict()

        ta2_data = ta2_known_episode_data
        sota_data = sota_known_episode_data
        ta2_true_negative_counts = ta2_known_true_negative_counts
        ta2_novelty_introduced_indices = ta2_known_novelty_introduced_indices
        for configuration in ta2_data:
            
            detection_source = 'given'
            # Collect per trial metrics for specific configurations, i.e., specific novelty and difficulty
            specific_config = False
            config_split = configuration.split('_')
            if len(config_split) == 2:
                specific_config = True
                for trial, novelty_introduced in zip(ta2_data[configuration], ta2_novelty_introduced_indices[configuration]):
                    trial_dict = dict()
                    trial_dict['ta2'] = self.ta2_team_name
                    trial_dict['novelty_level'] = config_split[0]
                    trial_dict['difficulty'] = config_split[1]
                    trial_dict['detection_source'] = detection_source
                    trial_dict['trial_index'] = trial['trial_index'][0]
                    trial_dict['num_episodes'] = len(trial['trial_index'])
                    trial_dict['novelty_episode'] = novelty_introduced
                    self.add_characterization(trial_dict, trial)
                    per_trial_metrics[trial.iloc[0]['trial_id']] = trial_dict
                for trial, novelty_introduced in zip(sota_data[configuration], ta2_novelty_introduced_indices[configuration]):
                    trial_dict = dict()
                    trial_dict['ta2'] = 'Baseline'
                    trial_dict['novelty_level'] = config_split[0]
                    trial_dict['difficulty'] = config_split[1]
                    trial_dict['detection_source'] = detection_source
                    trial_dict['trial_index'] = trial['trial_index'][0]
                    trial_dict['num_episodes'] = len(trial['trial_index'])
                    trial_dict['novelty_episode'] = novelty_introduced
                    self.add_characterization(trial_dict, trial)
                    per_trial_metrics[trial.iloc[0]['trial_id']] = trial_dict
            
            #known_m1[configuration] = '-'
            #known_m2[configuration] = '-'
            #known_m2_1[configuration] = '-'

            # M4 is for the known case, M3 is for the unknown case so set it to be blank
            #known_m3[configuration] = '-'
            
            # LBH: with stats
            known_m1[configuration] = '-,-,-,-,-,-'
            known_m2[configuration] = '-,-,-,-,-,-'
            known_m2_1[configuration] = '-,-,-,-,-,-'
            known_m3[configuration] = '-,-,-,-,-,-'
            known_m3_1[configuration] = '-,-,-,-,-,-'
            
            ### New Metric (%TN) = #TN / N_pre
            true_negative_counts = np.array(ta2_true_negative_counts[configuration])
            n_pre = np.array([ta2_novelty_introduced_indices[configuration][i]
                                if ta2_novelty_introduced_indices[configuration][i] > 0 else 1
                                for i in range(len(ta2_data[configuration]))])
            known_m2_2[configuration] = self.get_stats(true_negative_counts / n_pre)

            # M4 prep
            sota_pre_novelty = [sota_data[configuration][i]['performance']
                                [:ta2_novelty_introduced_indices[configuration][i]].tolist()
                                for i in range(len(sota_data[configuration]))]
            ta2_post_novelty_m = [ta2_data[configuration][i]['performance'][-m:]
                                    .tolist() for i in
                                    range(len(ta2_data[configuration]))]

            # Preventing dividing by zero by setting p_pre_beta to be small if = 0 (This should
            # never happen but just in case)
            p_pre_beta = [float(np.mean(sota_pre_novelty[i])) if np.mean(sota_pre_novelty[i]) > 0
                            else FLOAT_NEAR_ZERO for i in range(len(sota_pre_novelty))]
            p_post_alpha_means = [float(np.mean(ta2_post_novelty_m[i]))
                            for i in range(len(ta2_post_novelty_m))]
            
            m4_values = [p_post_alpha_means[i] / p_pre_beta[i]
                            for i in range(len(p_post_alpha_means))]
            known_m4[configuration] = self.get_stats(m4_values)
            
            # M4.1
            ta2_post_novelty_first_m = [ta2_data[configuration][i]['performance']
                                        [ta2_novelty_introduced_indices[configuration][i]:ta2_novelty_introduced_indices[configuration][i]+m]
                                    .tolist() for i in
                                    range(len(ta2_data[configuration]))]
            p_post_first_alpha_means = [float(np.mean(ta2_post_novelty_first_m[i]))
                            for i in range(len(ta2_post_novelty_first_m))]
            m4_1_values = [p_post_first_alpha_means[i] / p_pre_beta[i] for i in range(len(p_post_first_alpha_means))]
            known_m4_1[configuration] = self.get_stats(m4_1_values)

            if specific_config:
                for trial,m4,m4_1 in zip(ta2_data[configuration],m4_values,m4_1_values):
                    trial_dict = per_trial_metrics[trial.iloc[0]['trial_id']]
                    trial_dict['m4'] = m4
                    trial_dict['m4_1'] = m4_1
                for trial in sota_data[configuration]:
                    trial_dict = per_trial_metrics[trial.iloc[0]['trial_id']]
                    trial_dict['m4'] = 0
                    trial_dict['m4_1'] = 0

            sota_post_novelty = [sota_data[configuration][i]['performance']
                                    [ta2_novelty_introduced_indices[configuration][i]:].tolist()
                                    for i in range(len(sota_data[configuration]))]
            ta2_post_novelty = [ta2_data[configuration][i]['performance']
                                [ta2_novelty_introduced_indices[configuration][i]:].tolist()
                                for i in range(len(ta2_data[configuration]))]
            
            # AM1: OPTIg
            am1_values = [sum(ta2_post_novelty[i]) / (sum(sota_post_novelty[i]) + sum(ta2_post_novelty[i]))
                    if (sum(sota_post_novelty[i]) + sum(ta2_post_novelty[i])) > 0 else 0
                    for i in range(len(sota_post_novelty))]
            known_am1[configuration] = self.get_stats(am1_values)

            sota_post_novelty_m = [sota_data[configuration][i]['performance'][-m:]
                                    .tolist() for i in range(len(sota_data[configuration]))]
            p_post_alpha = [float(sum(ta2_post_novelty_m[i]))
                            for i in range(len(ta2_post_novelty_m))]
            p_post_beta = [float(sum(sota_post_novelty_m[i])) if sum(sota_post_novelty_m[i]) > 0
                            else FLOAT_NEAR_ZERO for i in range(len(sota_post_novelty_m))]

            # AM2: APTIg (old, un-normalized)
            #known_am2[configuration] = self.get_stats( #np.mean( # LBH: with stats
            #    [p_post_alpha[i] / p_post_beta[i] for i in range(len(p_post_alpha))])
            
            # AM2: APTIg (new, normalized)
            am2_values = [(p_post_alpha[i] / (p_post_alpha[i] + p_post_beta[i]))
                    if (p_post_alpha[i] + p_post_beta[i]) > 0 else 0
                    for i in range(len(p_post_alpha))]
            known_am2[configuration] = self.get_stats(am2_values)
            
            # AM4: IPTIg
            sota_post_novelty_first_m = [sota_data[configuration][i]['performance']
                                        [ta2_novelty_introduced_indices[configuration][i]:ta2_novelty_introduced_indices[configuration][i]+m]
                                    .tolist() for i in
                                    range(len(sota_data[configuration]))]
            p_post_alpha_first_m = [sum(ta2_post_novelty_first_m[i])
                            for i in range(len(ta2_post_novelty_first_m))]
            p_post_beta_first_m = [sum(sota_post_novelty_first_m[i]) if sum(sota_post_novelty_first_m[i]) > 0
                            else FLOAT_NEAR_ZERO for i in range(len(sota_post_novelty_first_m))]
            am4_values = [(float(p_post_alpha_first_m[i]) / (float(p_post_alpha_first_m[i]) + float(p_post_beta_first_m[i])))
                    if (float(p_post_alpha_first_m[i]) + float(p_post_beta_first_m[i])) > 0 else 0
                    for i in range(len(p_post_alpha_first_m))]
            known_am4[configuration] = self.get_stats(am4_values)
            
            if specific_config:
                for trial,am1,am2,am4 in zip(ta2_data[configuration],am1_values,am2_values,am4_values):
                    trial_dict = per_trial_metrics[trial.iloc[0]['trial_id']]
                    trial_dict['am1'] = am1
                    trial_dict['am2'] = am2
                    trial_dict['am4'] = am4
                for trial in sota_data[configuration]:
                    trial_dict = per_trial_metrics[trial.iloc[0]['trial_id']]
                    trial_dict['am1'] = 0
                    trial_dict['am2'] = 0
                    trial_dict['am4'] = 0
            
            # AM3: AU-AMOC
            # known_am3[configuration] = ['-', '-']
            #known_am3[configuration] = '-'
            
            # LBH: with stats
            known_am3[configuration] = '-,-,-,-,-,-'
            
            # nrm (NRM alpha) version in TA3 slides from 05/29/2021               
            ta2_pre_novelty = [ta2_data[configuration][i]['performance']
                                [:ta2_novelty_introduced_indices[configuration][i]].tolist()
                                for i in range(len(ta2_data[configuration]))]
            ta2_post_means = [float(np.mean(ta2_post_novelty[i]))
                                if len(ta2_post_novelty[i]) > 0 else 0
                                for i in range(len(ta2_data[configuration]))]
            ta2_pre_means = [float(np.mean(ta2_pre_novelty[i]))
                                if len(ta2_pre_novelty[i]) > 0 else 0
                                for i in range(len(ta2_data[configuration]))]
            ta2_abs_diffs = np.abs(np.subtract(ta2_post_means, ta2_pre_means))
            ta2_pre_stds = np.array([float(np.std(ta2_pre_novelty[i])) if np.std(ta2_pre_novelty[i]) > 0
                            else FLOAT_NEAR_ZERO for i in range(len(ta2_data[configuration]))])
            nrms = ta2_abs_diffs / ta2_pre_stds
            nrm = sum(nrms < 2) / len(nrms)
            known_nrm[configuration] = self.get_stats(np.array([nrm]))
            
            if specific_config:
                for trial,nrm in zip(ta2_data[configuration],nrms):
                    trial_dict = per_trial_metrics[trial.iloc[0]['trial_id']]
                    if nrm < 2:
                        trial_dict['nrm'] = 1
                    else:
                        trial_dict['nrm'] = 0
            
            # NRM beta/baseline              
            sota_post_means = [np.mean(sota_post_novelty[i])
                                if len(sota_post_novelty[i]) > 0 else 0
                                for i in range(len(sota_data[configuration]))]
            sota_pre_means = [np.mean(sota_pre_novelty[i])
                                if len(sota_pre_novelty[i]) > 0 else 0
                                for i in range(len(sota_data[configuration]))]
            sota_abs_diffs = np.abs(np.subtract(sota_post_means, sota_pre_means))
            sota_pre_stds = [np.std(sota_pre_novelty[i]) if np.std(sota_pre_novelty[i]) > 0
                            else FLOAT_NEAR_ZERO for i in range(len(sota_data[configuration]))]
            nrms_beta = sota_abs_diffs / sota_pre_stds
            nrm_beta = sum(nrms_beta < 2) / len(nrms_beta)
            known_nrm_beta[configuration] = self.get_stats(np.array([nrm_beta]))
            
            if specific_config:
                for trial,nrm_beta in zip(sota_data[configuration],nrms_beta):
                    trial_dict = per_trial_metrics[trial.iloc[0]['trial_id']]
                    if nrm_beta < 2:
                        trial_dict['nrm'] = 1
                    else:
                        trial_dict['nrm'] = 0
                        
            # Extra metrics
            known_pre_sota[configuration] = self.get_stats(sota_pre_means)
            known_pre_ta2[configuration] = self.get_stats(ta2_pre_means)
            known_post_sota[configuration] = self.get_stats(sota_post_means)
            known_post_ta2[configuration] = self.get_stats(ta2_post_means)
            
            if specific_config and self.extra_per_trial_metrics:
                sota_am3s = []
                num_trials = len(sota_data[configuration])
                for trial_index in range(num_trials):
                    amoc_points = self.AMOC(sota_data[configuration][trial_index],
                                        ta2_novelty_introduced_indices[configuration][trial_index]) # yes, ta2 (avoids computing for sota, which should be the same)
                    xs = [fs[0] for fs in amoc_points]
                    ys = [fs[1] for fs in amoc_points]
                    sota_am3s.append(np.trapz(y=ys, x=xs)) # trapz = area under (x,y) curve
                sota_post_novelty_first_m = [sota_data[configuration][i]['performance']
                                                [ta2_novelty_introduced_indices[configuration][i]:ta2_novelty_introduced_indices[configuration][i]+m] # yes, ta2 (avoids computing for sota, which should be the same)
                                                .tolist() for i in
                                                range(len(sota_data[configuration]))]
                p_post_first_beta_means = [np.mean(sota_post_novelty_first_m[i])
                                            for i in range(len(sota_post_novelty_first_m))]
                p_post_beta_means = [np.mean(sota_post_novelty_m[i])
                                        for i in range(len(sota_post_novelty_m))]
                for trial,au_amoc,pre_perf_all,post_perf_all,post_perf_first_m,post_perf_last_m in \
                        zip(ta2_data[configuration],
                            ta2_am3s,
                            ta2_pre_means,
                            ta2_post_means,
                            p_post_first_alpha_means,
                            p_post_alpha_means):
                    trial_dict = per_trial_metrics[trial.iloc[0]['trial_id']]
                    trial_dict['au_amoc'] = au_amoc
                    trial_dict['pre_perf_all'] = pre_perf_all
                    trial_dict['post_perf_all'] = post_perf_all
                    trial_dict['post_perf_first_m'] = post_perf_first_m
                    trial_dict['post_perf_last_m'] = post_perf_last_m
                for trial,au_amoc,pre_perf_all,post_perf_all,post_perf_first_m,post_perf_last_m in \
                        zip(sota_data[configuration],
                            sota_am3s,
                            sota_pre_means,
                            sota_post_means,
                            p_post_first_beta_means,
                            p_post_beta_means):
                    trial_dict = per_trial_metrics[trial.iloc[0]['trial_id']]
                    trial_dict['au_amoc'] = au_amoc
                    trial_dict['pre_perf_all'] = pre_perf_all
                    trial_dict['post_perf_all'] = post_perf_all
                    trial_dict['post_perf_first_m'] = post_perf_first_m
                    trial_dict['post_perf_last_m'] = post_perf_last_m

        # Creates plots for each combination of the data
        self.generate_average_plots(ta2_known_episode_data, sota_known_episode_data,
                                    ta2_known_novelty_introduced_indices, path, 'Known')
        self.generate_average_plots(ta2_unknown_episode_data, sota_unknown_episode_data,
                                    ta2_unknown_novelty_introduced_indices, path, 'Unknown')

        # Saves all of the metrics for each combination of the data
        for configuration in unknown_m1:
            filename = 'metrics_unknown_{0}.csv'.format(
                self.configuration_to_natural_words(configuration).lower())
            metricsfilename = os.path.join(path, filename)
            metricsfile = open(metricsfilename, 'w')
            # LBH: with stats
            metricsfile.write('Measure,Min,Max,Mean,Median,Norm-Median,StdDev\n')
            metricsfile.write('M1,' +
                                str(unknown_m1[configuration]['min']) + ',' +
                                str(unknown_m1[configuration]['max']) + ',' +
                                str(unknown_m1[configuration]['mean']) + ',' +
                                str(unknown_m1[configuration]['median']) + ',' +
                                str(unknown_m1[configuration]['norm_median']) + ',' +
                                str(unknown_m1[configuration]['stddev']) + '\n')
            metricsfile.write('M2,' +
                                str(unknown_m2[configuration]['min']) + ',' +
                                str(unknown_m2[configuration]['max']) + ',' +
                                str(unknown_m2[configuration]['mean']) + ',' +
                                str(unknown_m2[configuration]['median']) + ',' +
                                str(unknown_m2[configuration]['norm_median']) + ',' +
                                str(unknown_m2[configuration]['stddev']) + '\n')
            metricsfile.write('M2.1,' +
                                str(unknown_m2_1[configuration]['min']) + ',' +
                                str(unknown_m2_1[configuration]['max']) + ',' +
                                str(unknown_m2_1[configuration]['mean']) + ',' +
                                str(unknown_m2_1[configuration]['median']) + ',' +
                                str(unknown_m2_1[configuration]['norm_median']) + ',' +
                                str(unknown_m2_1[configuration]['stddev']) + '\n')
            metricsfile.write('M3,' +
                                str(unknown_m3[configuration]['min']) + ',' +
                                str(unknown_m3[configuration]['max']) + ',' +
                                str(unknown_m3[configuration]['mean']) + ',' +
                                str(unknown_m3[configuration]['median']) + ',' +
                                str(unknown_m3[configuration]['norm_median']) + ',' +
                                str(unknown_m3[configuration]['stddev']) + '\n')
            metricsfile.write('M3.1,' +
                                str(unknown_m3_1[configuration]['min']) + ',' +
                                str(unknown_m3_1[configuration]['max']) + ',' +
                                str(unknown_m3_1[configuration]['mean']) + ',' +
                                str(unknown_m3_1[configuration]['median']) + ',' +
                                str(unknown_m3_1[configuration]['norm_median']) + ',' +
                                str(unknown_m3_1[configuration]['stddev']) + '\n')
            metricsfile.write('M4,' + str(unknown_m4[configuration]) + '\n')
            metricsfile.write('M4.1,' + str(unknown_m4_1[configuration]) + '\n')
            metricsfile.write('OPTI,' +
                                str(unknown_am1[configuration]['min']) + ',' +
                                str(unknown_am1[configuration]['max']) + ',' +
                                str(unknown_am1[configuration]['mean']) + ',' +
                                str(unknown_am1[configuration]['median']) + ',' +
                                str(unknown_am1[configuration]['norm_median']) + ',' +
                                str(unknown_am1[configuration]['stddev']) + '\n')
            metricsfile.write('IPTI,' +
                                str(unknown_am4[configuration]['min']) + ',' +
                                str(unknown_am4[configuration]['max']) + ',' +
                                str(unknown_am4[configuration]['mean']) + ',' +
                                str(unknown_am4[configuration]['median']) + ',' +
                                str(unknown_am4[configuration]['norm_median']) + ',' +
                                str(unknown_am4[configuration]['stddev']) + '\n')
            metricsfile.write('APTI,' +
                                str(unknown_am2[configuration]['min']) + ',' +
                                str(unknown_am2[configuration]['max']) + ',' +
                                str(unknown_am2[configuration]['mean']) + ',' +
                                str(unknown_am2[configuration]['median']) + ',' +
                                str(unknown_am2[configuration]['norm_median']) + ',' +
                                str(unknown_am2[configuration]['stddev']) + '\n')
            #metricsfile.write('AM3,0,1,' + str(unknown_am3[configuration]) + ',1,1,0\n')
            metricsfile.write('AMOC,' +
                                str(unknown_am3[configuration]['min']) + ',' +
                                str(unknown_am3[configuration]['max']) + ',' +
                                str(unknown_am3[configuration]['mean']) + ',' +
                                str(unknown_am3[configuration]['median']) + ',' +
                                str(unknown_am3[configuration]['norm_median']) + ',' +
                                str(unknown_am3[configuration]['stddev']) + '\n')
            metricsfile.write('NRM,' +
                                str(unknown_nrm[configuration]['min']) + ',' +
                                str(unknown_nrm[configuration]['max']) + ',' +
                                str(unknown_nrm[configuration]['mean']) + ',' +
                                str(unknown_nrm[configuration]['median']) + ',' +
                                str(unknown_nrm[configuration]['norm_median']) + ',' +
                                str(unknown_nrm[configuration]['stddev']) + '\n')
            metricsfile.write('NRM_beta,' +
                                str(unknown_nrm_beta[configuration]['min']) + ',' +
                                str(unknown_nrm_beta[configuration]['max']) + ',' +
                                str(unknown_nrm_beta[configuration]['mean']) + ',' +
                                str(unknown_nrm_beta[configuration]['median']) + ',' +
                                str(unknown_nrm_beta[configuration]['norm_median']) + ',' +
                                str(unknown_nrm_beta[configuration]['stddev']) + '\n')
            metricsfile.write('M2.2,' +
                                str(unknown_m2_2[configuration]['min']) + ',' +
                                str(unknown_m2_2[configuration]['max']) + ',' +
                                str(unknown_m2_2[configuration]['mean']) + ',' +
                                str(unknown_m2_2[configuration]['median']) + ',' +
                                str(unknown_m2_2[configuration]['norm_median']) + ',' +
                                str(unknown_m2_2[configuration]['stddev']) + '\n')
            metricsfile.write('PRE_SOTA,' +
                                str(unknown_pre_sota[configuration]['min']) + ',' +
                                str(unknown_pre_sota[configuration]['max']) + ',' +
                                str(unknown_pre_sota[configuration]['mean']) + ',' +
                                str(unknown_pre_sota[configuration]['median']) + ',' +
                                str(unknown_pre_sota[configuration]['norm_median']) + ',' +
                                str(unknown_pre_sota[configuration]['stddev']) + '\n')
            metricsfile.write('PRE_TA2,' +
                                str(unknown_pre_ta2[configuration]['min']) + ',' +
                                str(unknown_pre_ta2[configuration]['max']) + ',' +
                                str(unknown_pre_ta2[configuration]['mean']) + ',' +
                                str(unknown_pre_ta2[configuration]['median']) + ',' +
                                str(unknown_pre_ta2[configuration]['norm_median']) + ',' +
                                str(unknown_pre_ta2[configuration]['stddev']) + '\n')
            metricsfile.write('POST_SOTA,' +
                                str(unknown_post_sota[configuration]['min']) + ',' +
                                str(unknown_post_sota[configuration]['max']) + ',' +
                                str(unknown_post_sota[configuration]['mean']) + ',' +
                                str(unknown_post_sota[configuration]['median']) + ',' +
                                str(unknown_post_sota[configuration]['norm_median']) + ',' +
                                str(unknown_post_sota[configuration]['stddev']) + '\n')
            metricsfile.write('POST_TA2,' +
                                str(unknown_post_ta2[configuration]['min']) + ',' +
                                str(unknown_post_ta2[configuration]['max']) + ',' +
                                str(unknown_post_ta2[configuration]['mean']) + ',' +
                                str(unknown_post_ta2[configuration]['median']) + ',' +
                                str(unknown_post_ta2[configuration]['norm_median']) + ',' +
                                str(unknown_post_ta2[configuration]['stddev']) + '\n')
            metricsfile.close()
            
        for configuration in known_m1:
            filename = 'metrics_known_{0}.csv'.format(
                self.configuration_to_natural_words(configuration).lower())
            metricsfilename = os.path.join(path, filename)
            metricsfile = open(metricsfilename, 'w')
            # LBH: with stats
            metricsfile.write('Measure,Min,Max,Mean,Median,Norm-Median,StdDev\n')
            metricsfile.write('M1,' + str(known_m1[configuration]) + '\n')
            metricsfile.write('M2,' + str(known_m2[configuration]) + '\n')
            metricsfile.write('M2.1,' + str(known_m2_1[configuration]) + '\n')
            metricsfile.write('M3,' + str(known_m3[configuration]) + '\n')
            metricsfile.write('M3.1,' + str(known_m3_1[configuration]) + '\n')
            metricsfile.write('M4,' +
                                str(known_m4[configuration]['min']) + ',' +
                                str(known_m4[configuration]['max']) + ',' +
                                str(known_m4[configuration]['mean']) + ',' +
                                str(known_m4[configuration]['median']) + ',' +
                                str(known_m4[configuration]['norm_median']) + ',' +
                                str(known_m4[configuration]['stddev']) + '\n')
            metricsfile.write('M4.1,' +
                                str(known_m4_1[configuration]['min']) + ',' +
                                str(known_m4_1[configuration]['max']) + ',' +
                                str(known_m4_1[configuration]['mean']) + ',' +
                                str(known_m4_1[configuration]['median']) + ',' +
                                str(known_m4_1[configuration]['norm_median']) + ',' +
                                str(known_m4_1[configuration]['stddev']) + '\n')
            metricsfile.write('OPTI,' +
                                str(known_am1[configuration]['min']) + ',' +
                                str(known_am1[configuration]['max']) + ',' +
                                str(known_am1[configuration]['mean']) + ',' +
                                str(known_am1[configuration]['median']) + ',' +
                                str(known_am1[configuration]['norm_median']) + ',' +
                                str(known_am1[configuration]['stddev']) + '\n')
            metricsfile.write('IPTI,' +
                                str(known_am4[configuration]['min']) + ',' +
                                str(known_am4[configuration]['max']) + ',' +
                                str(known_am4[configuration]['mean']) + ',' +
                                str(known_am4[configuration]['median']) + ',' +
                                str(known_am4[configuration]['norm_median']) + ',' +
                                str(known_am4[configuration]['stddev']) + '\n')
            metricsfile.write('APTI,' +
                                str(known_am2[configuration]['min']) + ',' +
                                str(known_am2[configuration]['max']) + ',' +
                                str(known_am2[configuration]['mean']) + ',' +
                                str(known_am2[configuration]['median']) + ',' +
                                str(known_am2[configuration]['norm_median']) + ',' +
                                str(known_am2[configuration]['stddev']) + '\n')
            metricsfile.write('AMOC,' + str(known_am3[configuration]) + '\n')
            metricsfile.write('NRM,' +
                                str(known_nrm[configuration]['min']) + ',' +
                                str(known_nrm[configuration]['max']) + ',' +
                                str(known_nrm[configuration]['mean']) + ',' +
                                str(known_nrm[configuration]['median']) + ',' +
                                str(known_nrm[configuration]['norm_median']) + ',' +
                                str(known_nrm[configuration]['stddev']) + '\n')
            metricsfile.write('NRM_beta,' +
                                str(known_nrm_beta[configuration]['min']) + ',' +
                                str(known_nrm_beta[configuration]['max']) + ',' +
                                str(known_nrm_beta[configuration]['mean']) + ',' +
                                str(known_nrm_beta[configuration]['median']) + ',' +
                                str(known_nrm_beta[configuration]['norm_median']) + ',' +
                                str(known_nrm_beta[configuration]['stddev']) + '\n')
            metricsfile.write('M2.2,' +
                                str(known_m2_2[configuration]['min']) + ',' +
                                str(known_m2_2[configuration]['max']) + ',' +
                                str(known_m2_2[configuration]['mean']) + ',' +
                                str(known_m2_2[configuration]['median']) + ',' +
                                str(known_m2_2[configuration]['norm_median']) + ',' +
                                str(known_m2_2[configuration]['stddev']) + '\n')
            metricsfile.write('PRE_SOTA,' +
                                str(known_pre_sota[configuration]['min']) + ',' +
                                str(known_pre_sota[configuration]['max']) + ',' +
                                str(known_pre_sota[configuration]['mean']) + ',' +
                                str(known_pre_sota[configuration]['median']) + ',' +
                                str(known_pre_sota[configuration]['norm_median']) + ',' +
                                str(known_pre_sota[configuration]['stddev']) + '\n')
            metricsfile.write('PRE_TA2,' +
                                str(known_pre_ta2[configuration]['min']) + ',' +
                                str(known_pre_ta2[configuration]['max']) + ',' +
                                str(known_pre_ta2[configuration]['mean']) + ',' +
                                str(known_pre_ta2[configuration]['median']) + ',' +
                                str(known_pre_ta2[configuration]['norm_median']) + ',' +
                                str(known_pre_ta2[configuration]['stddev']) + '\n')
            metricsfile.write('POST_SOTA,' +
                                str(known_post_sota[configuration]['min']) + ',' +
                                str(known_post_sota[configuration]['max']) + ',' +
                                str(known_post_sota[configuration]['mean']) + ',' +
                                str(known_post_sota[configuration]['median']) + ',' +
                                str(known_post_sota[configuration]['norm_median']) + ',' +
                                str(known_post_sota[configuration]['stddev']) + '\n')
            metricsfile.write('POST_TA2,' +
                                str(known_post_ta2[configuration]['min']) + ',' +
                                str(known_post_ta2[configuration]['max']) + ',' +
                                str(known_post_ta2[configuration]['mean']) + ',' +
                                str(known_post_ta2[configuration]['median']) + ',' +
                                str(known_post_ta2[configuration]['norm_median']) + ',' +
                                str(known_post_ta2[configuration]['stddev']) + '\n')
            metricsfile.close()

        # Run combine on the results.
        self.run_combine(path)
        
        # Write per trial metrics, if requested
        if self.details:
            #self.write_per_trial_metrics(per_trial_metrics, path)
            self.write_per_trial_metrics_ta3(per_trial_metrics, path)

        # Remove any zip file that was already there.
        cmd = 'rm -f {}.zip'.format(path)
        p = subprocess.Popen(cmd,
                                shell=True,
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                close_fds=True)
        output = p.stdout.read()
        self.log.info(output)

        # Compress the results and remove the source.
        cmd = 'zip -m -j -r {}.zip {}'.format(path, path)
        p = subprocess.Popen(cmd,
                                shell=True,
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                close_fds=True)
        output = p.stdout.read()
        self.log.info(output)
        
        # Remove path, which is now empty.
        cmd = 'rmdir {}'.format(path)
        p = subprocess.Popen(cmd,
                                shell=True,
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                close_fds=True)
        output = p.stdout.read()
        self.log.info(output)

        return
    
    def add_characterization(self, trial_dict, trial):
        # Get characterization for last episode in trial and add components to trial dict
        num_episodes = len(trial['trial_index'])
        nov_char_dict = trial.iloc[num_episodes-1]['novelty_characterization']
        if type(nov_char_dict) is str:
            nov_char_dict = json.loads(nov_char_dict)
        if 'novelty_characterization' in nov_char_dict: # special case for PARC and Rutgers
            nov_char_dict = nov_char_dict['novelty_characterization']
        trial_dict['p0'] = 0.0
        trial_dict['p1'] = 0.0
        trial_dict['p2'] = 0.0
        trial_dict['p3'] = 0.0
        trial_dict['p4'] = 0.0
        trial_dict['p5'] = 0.0
        trial_dict['p6'] = 0.0
        trial_dict['p7'] = 0.0
        trial_dict['p8'] = 0.0
        trial_dict['entity'] = None
        trial_dict['attribute'] = None
        trial_dict['change'] = None
        nov_char_dict_level = None
        if 'level' in nov_char_dict:
            nov_char_dict_level = nov_char_dict['level']
        if 'levels' in nov_char_dict:
            nov_char_dict_level = nov_char_dict['levels']
        if nov_char_dict_level:
            for level_prob in nov_char_dict_level:
                level_number = -1
                prob = 0.0
                if 'level_number' in level_prob: # official key
                    level_number = level_prob['level_number']
                if 'Level' in level_prob: # informal key
                    level_number = level_prob['Level']
                if 'level' in level_prob: # informal key
                    level_number = level_prob['level']
                if (level_number > -1) and (level_number < 9):
                    key = 'p' + str(level_number)
                    if 'probability' in level_prob: # official key
                        prob = level_prob['probability']
                    if 'Prob' in level_prob: # informal key
                        prob = level_prob['Prob']
                    if 'prob' in level_prob: # another informal key
                        prob = level_prob['prob']
                    if (prob >= 0.0) and (prob <= 1.0):
                        trial_dict[key] = prob
        if 'entity' in nov_char_dict:
            trial_dict['entity'] = str(nov_char_dict['entity'])
        if 'attribute' in nov_char_dict:
            trial_dict['attribute'] = str(nov_char_dict['attribute'])
        if 'change' in nov_char_dict:
            trial_dict['change'] = str(nov_char_dict['change'])
        return
    
    def write_per_trial_metrics(self, per_trial_metrics, path):
        """Write metrics to file for each trial in experiment."""
        filename = 'metrics_per_trial.csv'
        metricsfilename = os.path.join(path, filename)
        metricsfile = open(metricsfilename, 'w')
        metricsfile.write('ta1,ta2_or_baseline,domain,detection_source,' + 
                          'trial_id,number_of_instances,instance_of_distribution_change,' + 
                          'novelty_hierarchy_level,difficulty_category_level,' + 
                          'm1_fn_cdt,m2_cdt,m21_fp,m3_nrp_sys_det,m4_nrp_given_det,optis,iptis,aptis,optig,iptig,aptig,nrm,m31,m41')
        if self.extra_per_trial_metrics:
            metricsfile.write(',au_amoc,pre_perf_all,post_perf_all,post_perf_first_m,post_perf_last_m')
        metricsfile.write('\n')
        for (trial_id,trial) in per_trial_metrics.items():
            linestr = self.ta1_team_name
            linestr += ',' + trial['ta2']
            linestr += ',' + self.domain_name
            linestr += ',' + trial['detection_source']
            linestr += ',' + str(trial['trial_index'] + 1)
            linestr += ',' + str(trial['num_episodes'])
            linestr += ',' + str(trial['novelty_episode'] + 1)
            linestr += ',' + str(trial['novelty_level'])
            linestr += ',' + str(trial['difficulty'])
            linestr += ',' + (str(trial['m1']) if trial['detection_source'] == 'system' else '')
            linestr += ',' + (str(trial['m2']) if trial['detection_source'] == 'system' else '')
            linestr += ',' + (str(trial['m2_1']) if trial['detection_source'] == 'system' else '')
            linestr += ',' + (str(trial['m3']) if trial['detection_source'] == 'system' else '')
            linestr += ',' + (str(trial['m4']) if trial['detection_source'] == 'given' else '')
            if trial['detection_source'] == 'system':
                linestr += ',' + str(trial['am1']) # OPTIs
                linestr += ',' + str(trial['am4']) # IPTIs
                linestr += ',' + str(trial['am2']) # APTIs
                linestr += ',,,' # OPTIg, IPTIg, APTIg not relevant
            else: # given detection
                linestr += ',,,' # OPTIs, IPTIs, APTIs not relevant
                linestr += ',' + str(trial['am1']) # OPTIg
                linestr += ',' + str(trial['am4']) # IPTIg
                linestr += ',' + str(trial['am2']) # APTIg
            linestr += ',' + str(trial['nrm'])
            # New M3.1 and M4.1 metrics
            linestr += ',' + (str(trial['m3_1']) if trial['detection_source'] == 'system' else '')
            linestr += ',' + (str(trial['m4_1']) if trial['detection_source'] == 'given' else '')
            if self.extra_per_trial_metrics:
                linestr += ',' + (str(trial['au_amoc']) if trial['detection_source'] == 'system' else '')
                linestr += ',' + str(trial['pre_perf_all'])
                linestr += ',' + str(trial['post_perf_all'])
                linestr += ',' + str(trial['post_perf_first_m'])
                linestr += ',' + str(trial['post_perf_last_m'])
            linestr += '\n'
            metricsfile.write(linestr)
        metricsfile.close()
        return
    
    def write_per_trial_metrics_ta3(self, per_trial_metrics, path):
        """Write metrics to file for each trial in experiment.
        This version follows the requested format from TA3.
        The TA2 results and Baseline results are written to separate files."""
        # Write TA2 metrics
        filename = 'metrics_per_trial.csv'
        metricsfilename = os.path.join(path, filename)
        metricsfile = open(metricsfilename, 'w')
        metricsfile.write('ta1,ta2_or_baseline,domain,trial_id,number_of_instances,instance_of_distribution_change,' + 
                          'novelty_hierarchy_level,difficulty_category_level,m1_fn_cdt,m2_cdt,m21_fp,' + 
                          'm3_nrp_sys_det,m31,m4_nrp_given_det,m41,optis,iptis,aptis,optig,iptig,aptig,nrm' +
                          ',p0,p1,p2,p3,p4,p5,p6,p7,p8')
        metricsfile.write('\n')
        for (trial_id,trial) in per_trial_metrics.items():
            if trial['ta2'] != 'Baseline':
                linestr = self.ta1_team_name
                linestr += ',' + trial['ta2']
                linestr += ',' + self.domain_name
                linestr += ',' + str(trial['trial_index'] + 1)
                linestr += ',' + str(trial['num_episodes'])
                linestr += ',' + str(trial['novelty_episode'] + 1)
                # Report novelty level as single digit
                novelty_level = int(trial['novelty_level'])
                if (novelty_level in [100,101,102,103,104,105,106,107,108]):
                    novelty_level -= 100
                if (novelty_level in [200,201,202,203,204,205,206,207,208]):
                    novelty_level -= 200
                linestr += ',' + str(novelty_level)
                linestr += ',' + str(trial['difficulty'])

                cdt = False
                if trial['detection_source'] == 'system':
                    cdt = (True if trial['m2'] == 1 else False) # trial['m2'] undefined for 'given' detection
                linestr += ',' + (str(trial['m1']) if (cdt and (trial['detection_source'] == 'system')) else '')
                linestr += ',' + (str(trial['m2']) if trial['detection_source'] == 'system' else '')
                linestr += ',' + (str(trial['m2_1']) if trial['detection_source'] == 'system' else '')
                linestr += ',' + (str(trial['m3']) if trial['detection_source'] == 'system' else '')
                linestr += ',' + (str(trial['m3_1']) if trial['detection_source'] == 'system' else '')
                linestr += ',' + (str(trial['m4']) if trial['detection_source'] == 'given' else '')
                linestr += ',' + (str(trial['m4_1']) if trial['detection_source'] == 'given' else '')
                if trial['detection_source'] == 'system':
                    linestr += ',' + str(trial['am1']) # OPTIs
                    linestr += ',' + str(trial['am4']) # IPTIs
                    linestr += ',' + str(trial['am2']) # APTIs
                    linestr += ',,,' # OPTIg, IPTIg, APTIg not relevant
                else: # given detection
                    linestr += ',,,' # OPTIs, IPTIs, APTIs not relevant
                    linestr += ',' + str(trial['am1']) # OPTIg
                    linestr += ',' + str(trial['am4']) # IPTIg
                    linestr += ',' + str(trial['am2']) # APTIg
                # TA2 NRM now reporteed as corresponding Baseline's NRM, so need to look it up
                nrm = self.get_baseline_nrm(trial, per_trial_metrics)
                linestr += ',' + str(nrm)
                
                # Novelty Level Predictions
                linestr += ',' + str(trial['p0'])
                linestr += ',' + str(trial['p1'])
                linestr += ',' + str(trial['p2'])
                linestr += ',' + str(trial['p3'])
                linestr += ',' + str(trial['p4'])
                linestr += ',' + str(trial['p5'])
                linestr += ',' + str(trial['p6'])
                linestr += ',' + str(trial['p7'])
                linestr += ',' + str(trial['p8'])
                linestr += '\n'
                metricsfile.write(linestr)
        metricsfile.close()
        return
    
    def get_baseline_nrm(self, ta2_trial, per_trial_metrics):
        for (trial_id,trial) in per_trial_metrics.items():
            if (trial['ta2'] == 'Baseline' and
                trial['novelty_level'] == ta2_trial['novelty_level'] and
                trial['difficulty'] == ta2_trial['difficulty'] and
                trial['detection_source'] == ta2_trial['detection_source'] and
                trial['trial_index'] == ta2_trial['trial_index']):
                return trial['nrm']
        self.log.error("get_baseline_nrm: can't find Baseline trial corresponding to TA2 trial.")
    
    def AMOC(self, trial, novelty_introduced_index):
        def alarm_score (tau, e, Npost):
            if (Npost <= 0) or (e < tau):
                return 0
            return ((Npost - (e - tau)) / Npost)
        def alarm_prob(a):
            return a[1]
        S = 0 # cumulative score
        F = 0 # false alarm rate
        R = [[0,0]]
        H = -1
        tau = novelty_introduced_index # N_pre + 1
        N_post = len(trial) - tau + 1
        alarms = []
        for episode_index in range(len(trial)):
            nov_prob = trial.iloc[episode_index]['novelty_probability']
            if nov_prob > 0.0:
                alarms.append([episode_index, nov_prob])
        alarms.sort(reverse=True, key=alarm_prob)
        for alarm in alarms:
            episode_index = alarm[0]
            if episode_index < tau:
                F = F + 1 # false alarm penalty f = 1
            else:
                if H < 0: # H undefined
                    S = S + alarm_score(tau, episode_index, N_post)
                    H = episode_index
                else:
                    if episode_index < H:
                        S = S - alarm_score(tau, H, N_post)
                        S = S + alarm_score(tau, episode_index, N_post)
                        H = episode_index
            R.append([F,S])
        S_total = S
        F_total = F
        for FS in R:
            if (F_total > 0):
                FS[0] = FS[0] / F_total
            if (S_total > 0):
                FS[1] = FS[1] / S_total
        S_last = R[-1][1]
        R.append([1,S_last]) # in case no false alarms
        return R
    
    def get_stats(self, arr):
        """Return dictionary of statistics: min, max, mean, median, norm_median, stddev."""
        stats = dict()
        stats['min'] = np.amin(arr)
        stats['max'] = np.amax(arr)
        stats['mean'] = np.mean(arr)
        stats['median'] = np.median(arr)
        normalizer = stats['max'] - stats['min']
        if normalizer == 0:
            normalizer = 1
        stats['norm_median'] = (stats['median'] - stats['min']) / normalizer
        stats['stddev'] = np.std(arr)
        return stats
    
    def to_float(self, data):
        """Converts all non-list elements to float."""
        if isinstance(data, list):
            return [self.to_float(item) for item in data]
        elif isinstance(data, Decimal):
            return float(data)
        elif isinstance(data, float):
            return float(data)
        else:
            return data

    def get_experiment_novelties(self):
        """Get the list of novelties used in the experiment."""
        self.log.debug('get_experiment_novelties(experiment_id={})'.format(self.experiment_id))
        novelties = list([NOVELTY_200])
        unique_novelties = self.agent_results_df['novelty_level'].unique().tolist()
        if unique_novelties:
            novelties = unique_novelties
        return novelties

    def get_experiment_difficulties(self):
        """Get the list of difficulties used in the experiment."""
        self.log.debug('get_experiment_difficulties(experiment_id={})'.format(self.experiment_id))
        difficulties = list([DIFFICULTY_EASY])
        unique_difficulties = self.agent_results_df['novelty_difficulty'].unique().tolist()
        if unique_difficulties:
            difficulties = unique_difficulties
        return difficulties

    def get_experiment_detection_conditions(self):
        """Get the detection conditions used in the experiment."""
        self.log.debug('get_experiment_detection_conditions(experiment_id={})'.format(self.experiment_id))
        detection_conditions = list([DETECTION_KNOWN, DETECTION_UNKNOWN])
        unique_detection_conditions_numeric = self.agent_results_df['novelty_visibility'].unique().tolist()
        if unique_detection_conditions_numeric:
            detection_conditions = [DETECTION_TRANSLATION[i] for i in unique_detection_conditions_numeric]
        return detection_conditions

    def get_experiment_trials(self, results_df):
        """Uses the provided experiment id to find trial ids that correspond
        to it. Separates cases where novelty was hidden from cases where
        novelty was given"""
        self.log.debug(
            'get_experiment_trials(experiment_id={})'.format(self.experiment_id))
        
        trials = results_df.drop_duplicates(subset=['trial_index', 'trial_id', 'novelty_level',
                                                         'novelty_difficulty','novelty_visibility'])
        return trials

    def get_experiment_trial_ids(self, ta2_experiment_trials, sota_experiment_trials):
        """Extracts each combination of configurations from the list
        of experiment trials. These are stored in a dictionary with keys
        corresponding to each combination of novelty_difficulty"""
        self.log.debug('get_experiment_trial_ids()')

        ta2_known_trial_ids = {}
        ta2_unknown_trial_ids = {}
        sota_known_trial_ids = {}
        sota_unknown_trial_ids = {}
        for novelty in self.possible_novelties:
            for difficulty in self.possible_difficulties:
                ta2_known_trial_ids[str(novelty) + '_' + difficulty] = ta2_experiment_trials.query(
                    'novelty_visibility == 1 & novelty_level == {0} & novelty_difficulty == \'{1}\''
                    .format(novelty, difficulty))['trial_id']
                ta2_unknown_trial_ids[str(novelty) + '_' + difficulty] = ta2_experiment_trials.query(
                    'novelty_visibility == 0 & novelty_level == {0} & novelty_difficulty == \'{1}\''
                    .format(novelty, difficulty))['trial_id']
                sota_known_trial_ids[str(novelty) + '_' + difficulty] = sota_experiment_trials.query(
                    'novelty_visibility == 1 & novelty_level == {0} & novelty_difficulty == \'{1}\''
                    .format(novelty, difficulty))['trial_id']
                sota_unknown_trial_ids[str(novelty) + '_' + difficulty] = sota_experiment_trials.query(
                    'novelty_visibility == 0 & novelty_level == {0} & novelty_difficulty == \'{1}\''
                    .format(novelty, difficulty))['trial_id']
        return ta2_known_trial_ids, ta2_unknown_trial_ids, sota_known_trial_ids, \
            sota_unknown_trial_ids

    def get_episode_data(self, experiment_trials, results_df):
        """Gets all of the necessary data from episodes corresponding to given trials and results."""
        self.log.debug('get_episode_data()')
        episodes = dict()
        for configuration in experiment_trials:
            episodes[configuration] = []
            for trial_id in experiment_trials[configuration]:
                results = results_df[results_df['trial_id'] == trial_id]
                results = results.copy()
                results['novelty_detected'] = (results['novelty_probability'] >= results['novelty_threshold'])
                results['novelty_characterization'] = json.dumps({'source': 'local'})
                episodes[configuration].append(results)
            episodes[configuration].sort(key=lambda i: i.iloc[0]['trial_index'])
        return episodes

    def get_novelty_introduced_indices(self, episode_data, hidden=False):
        """Gets episode indices when novelty was introduced"""
        self.log.debug(
            'get_novelty_introduced_indices(hidden={})'.format(hidden))
        novelty_introduced_indices = dict()
        false_positive_counts = dict()
        true_positive_counts = dict()
        first_true_positive_indices = dict()
        false_negative_counts = dict()
        true_negative_counts = dict()
        correctly_detected_trial = dict()
        for configuration in episode_data:
            novelty_introduced_indices[configuration] = []
            false_positive_counts[configuration] = []
            true_positive_counts[configuration] = []
            first_true_positive_indices[configuration] = []
            false_negative_counts[configuration] = []
            true_negative_counts[configuration] = []
            correctly_detected_trial[configuration] = []
            for trial in episode_data[configuration]:
                novel_episodes = trial[trial['novelty_initiated']]
                novelty_introduced = len(trial) # i.e., novelty never introduced
                if len(novel_episodes) > 0:
                    novelty_introduced = trial[trial['novelty_initiated']].index[0]
                novelty_introduced_indices[configuration].append(
                    novelty_introduced)
                true_negative_count = len(trial.iloc[0:novelty_introduced].query('not novelty_detected'))
                true_negative_counts[configuration].append(
                    true_negative_count)
                if hidden:
                    false_positive_count = len(
                        trial.iloc[0:novelty_introduced].query('novelty_detected'))
                    true_positive_count = len(
                        trial.iloc[novelty_introduced:].query('novelty_detected'))
                    false_positive_counts[configuration].append(
                        false_positive_count)
                    true_positive_counts[configuration].append(
                        true_positive_count)
                    try:
                        first_true_positive_index = trial.iloc[novelty_introduced:].query('novelty_detected').index[0]
                    except IndexError:
                        # Novelty never detected
                        first_true_positive_index = len(trial)
                    first_true_positive_indices[configuration].append(first_true_positive_index)
                    # New approach to FN counts and M1, i.e., M1 = #FN before first FP
                    false_negative_counts[configuration].append(first_true_positive_index - novelty_introduced)
                    correctly_detected_trial[configuration].append(
                        (false_positive_count == 0) and (true_positive_count > 0))
        return (novelty_introduced_indices, false_positive_counts, true_positive_counts,
                first_true_positive_indices, false_negative_counts, true_negative_counts, correctly_detected_trial)

    def generate_trial_plots(self, ta2_episode_data, sota_episode_data, novelty_introduced_indices, path, condition):
        """Generates a TA2/SOTA performance plot for each trial."""
        self.log.debug('generate_trial_plots()')
        plt.figure()

        for configuration in ta2_episode_data:
            for trial_index in range(len(ta2_episode_data[configuration])):
                
                # TODO: These lists of episodes may need to be modified given possibility of empty episodes
                ta2_data = ta2_episode_data[configuration][trial_index]
                sota_data = sota_episode_data[configuration][trial_index]
                novelty_introduced_index = novelty_introduced_indices[configuration][trial_index]

                plt.ylim(0.0, 1.05) # LBH: changed upper limit from 1.0 to 1.05 to clearly see 1.0 performance
                plt.ylabel('Performance')
                plt.xlabel('Episodes')

                # TODO: These x values may need to be modified given possibility of empty episodes
                ta2_x_values = list(range(len(ta2_data)))
                sota_x_values = list(range(len(sota_data)))
                plt.plot(ta2_x_values, ta2_data['performance'], label='Agent', color='black')
                plt.plot(sota_x_values,
                         sota_data['performance'], label='Baseline', color='gold')
                plt.axvline(novelty_introduced_index,
                            0,
                            1,
                            label='Novelty Introduced',
                            color='gray',
                            linestyle=':')
                plt.legend()
                plt.xticks([]) # remove episode numbers to hide novelty introduction episode number

                natural_config = self.configuration_to_natural_words(
                    configuration)
                label = 'Perf: {0} {1} Trial {2} (ID {3}'.format(
                    condition, natural_config, ta2_data['trial_index'][0], ta2_data['trial_id'][0])
                plt.title(label)

                filename = 'perf_{0}_{1}_trial_{2}_id_{3}.png'.format(
                    condition.lower(), natural_config.lower(), ta2_data['trial_index'][0], ta2_data['trial_id'][0])
                plt.savefig(os.path.join(path, filename), bbox_inches='tight')

                plt.cla()
        return
    
    def generate_amoc_plots(self, ta2_episode_data, novelty_introduced_indices, path, condition):
        """Generates an AMOC plot for each trial."""
        self.log.debug('generate_amoc_plots()')
        plt.figure()
        for configuration in ta2_episode_data:
            for trial_index in range(len(ta2_episode_data[configuration])):
                ta2_data = ta2_episode_data[configuration][trial_index]
                novelty_introduced_index = novelty_introduced_indices[configuration][trial_index]
                amoc_points = self.AMOC(ta2_data, novelty_introduced_index)
                xs = [fs[0] for fs in amoc_points]
                ys = [fs[1] for fs in amoc_points]
        
                #plt.xlim(0.0, 1.0)
                #plt.ylim(0.0, 1.1)
                plt.xlabel('False Alarm Rate')
                plt.ylabel('Average Score')
                plt.plot(xs, ys, color='blue')

                natural_config = self.configuration_to_natural_words(configuration)
                label = 'AMOC: {0} {1} Trial {2} (ID {3})'.format(
                    condition, natural_config, ta2_data['trial_index'][0], ta2_data['trial_id'][0])
                plt.title(label)

                filename = 'amoc_{0}_{1}_trial_{2}_id_{3}.png'.format(
                    condition.lower(), natural_config.lower(), ta2_data['trial_index'][0], ta2_data['trial_id'][0])
                plt.savefig(os.path.join(path, filename), bbox_inches='tight')

                plt.cla()
        return
    
    def generate_remaining_groups(self, data):
        """Creates the remaining groups needed to generate all of the metrics"""
        self.log.debug('generate_remaining_groups()')
        # Create groups for each novelty level including all difficulty levels
        for novelty_level in self.possible_novelties:
            tmp_data = []
            for difficulty_level in self.possible_difficulties:
                index = str(novelty_level) + '_' + difficulty_level
                tmp_data += data[index]
            data[str(novelty_level)] = tmp_data
        # Create groups for each difficulty level including all novelty levels
        for difficulty_level in self.possible_difficulties:
            tmp_data = []
            for novelty_level in self.possible_novelties:
                index = str(novelty_level) + '_' + difficulty_level
                tmp_data += data[index]
            data[difficulty_level] = tmp_data
        # Create group for all data
        tmp_data = []
        for novelty_level in self.possible_novelties:
            for difficulty_level in self.possible_difficulties:
                index = str(novelty_level) + '_' + difficulty_level
                tmp_data += data[index]
        data['all'] = tmp_data
        return data
    
    # New version that aligns curves at max BRB episode (DMKD version)
    def generate_average_plots(self, ta2_episode_data, sota_episode_data,
                               novelty_introduced_indices, path, condition):
        """Generates a plot for each combination of the data conditions"""
        self.log.debug('generate_average_plots()')
        plt.figure()

        for configuration in ta2_episode_data:
            ta2_data = ta2_episode_data[configuration]
            sota_data = sota_episode_data[configuration]
            novelty_introduced_index = np.mean(
                novelty_introduced_indices[configuration])
            novelty_introduced_index_max = np.max(
                novelty_introduced_indices[configuration])
            ta2_performance = [x['performance'] for x in ta2_data]
            ta2_performance_shifted = []
            for x,n in zip(ta2_performance,novelty_introduced_indices[configuration]):
                xnp = list(x.to_numpy().astype(float))
                #print("xnp = " + str(xnp))
                shift = novelty_introduced_index_max - n
                #print("shift = " + str(shift))
                shifted_array = xnp
                if shift > 0:
                    prefix_array = list(np.repeat(xnp[0], shift))
                    #print("prefix_array = " + str(prefix_array))
                    shifted_array = prefix_array + xnp
                #print("shifted_array = " + str(shifted_array))
                trunc_array = shifted_array[:len(xnp)]
                #print("trunc_arry = " + str(trunc_array))
                ta2_performance_shifted.append(np.array(trunc_array))
            sota_performance = [x['performance'] for x in sota_data]
            sota_performance_shifted = []
            for x,n in zip(sota_performance,novelty_introduced_indices[configuration]):
                xnp = list(x.to_numpy().astype(float))
                shift = novelty_introduced_index_max - n
                shifted_array = xnp
                if shift > 0:
                    prefix_array = list(np.repeat(xnp[0], shift))
                    shifted_array = prefix_array + xnp
                trunc_array = shifted_array[:len(xnp)]
                sota_performance_shifted.append(np.array(trunc_array))
            ta2_average_performance = np.mean(ta2_performance_shifted, axis=0)
            sota_average_performance = np.mean(sota_performance_shifted, axis=0)

            plt.ylim(0.0, 1.05) # LBH: changed upper limit from 1.0 to 1.05 to clearly see 1.0 performance
            plt.ylabel('Performance')
            plt.xlabel('Episodes')

            ta2_x_values = np.array(range(len(ta2_average_performance)))
            sota_x_values = np.array(range(len(sota_average_performance)))
            plt.plot(ta2_x_values,
                     self.smooth(ta2_average_performance),
                     label='Agent',
                     color='black')
            plt.plot(sota_x_values,
                     self.smooth(sota_average_performance),
                     label='Baseline',
                     color='gold')
            plt.axvline(novelty_introduced_index_max,
                        0,
                        1,
                        label='Novelty Introduced',
                        color='gray',
                        linestyle=':')
            plt.legend()
            plt.xticks([])

            config = self.configuration_to_natural_words(configuration)
            label = 'Averages for: {0} {1}'.format(condition, config)
            #plt.title(label) # DMKD: no title

            filename = 'averages_{0}_{1}.png'.format(
                condition.lower(), config.lower())
            plt.savefig(os.path.join(path, filename), bbox_inches='tight', dpi=300) # DMKD: increase DPI to 300

            plt.cla()
        return

    def generate_all_plot_old(self, ta2_data, sota_data, novelty_introduced_indices, path):
        """Generates a plot for all of the data"""
        self.log.debug('generate_all_plots()')
        plt.figure()
        novelty_introduced_index = np.mean(novelty_introduced_indices)
        ta2_average_performance = np.mean(
            [x['performance'] for x in ta2_data], axis=0)
        sota_average_performance = np.mean(
            [x['performance'] for x in sota_data], axis=0)

        plt.ylim(0.0, 1.05) # LBH: changed upper limit from 1.0 to 1.05 to clearly see 1.0 performance
        plt.ylabel('Performance')
        plt.xlabel('Episodes')
        
        ta2_average_performance = ta2_average_performance[~pd.isnull(
            ta2_average_performance)]
        sota_average_performance = sota_average_performance[~pd.isnull(
            sota_average_performance)]

        ta2_x_values = np.array(range(len(ta2_average_performance)))
        sota_x_values = np.array(range(len(sota_average_performance)))

        plt.plot(ta2_x_values,
                 self.smooth(ta2_average_performance),
                 label='Agent',
                 color='black')
        plt.plot(sota_x_values,
                 self.smooth(sota_average_performance),
                 label='Baseline',
                 color='gold')
        plt.axvline(novelty_introduced_index,
                    0,
                    1,
                    label='Novelty Introduced',
                    color='gray',
                    linestyle=':')
        plt.legend()

        label = 'Averages for all data'
        plt.title(label)

        filename = 'averages_all.png'
        plt.savefig(os.path.join(path, filename), bbox_inches='tight')

        plt.cla()
        return
    
    # New version that aligns plot at max BRB episode
    def generate_all_plot(self, ta2_data, sota_data, novelty_introduced_indices, path):
        """Generates a plot for all of the data"""
        self.log.debug('generate_all_plots()')
        plt.figure()
        
        # ----- begin: same code inside "for configuration" loop in generate_average_plots
        novelty_introduced_index = np.mean(
                novelty_introduced_indices)
        novelty_introduced_index_max = np.max(
                novelty_introduced_indices)
        ta2_performance = [x['performance'] for x in ta2_data]
        ta2_performance_shifted = []
        for x,n in zip(ta2_performance,novelty_introduced_indices):
            xnp = list(x.to_numpy().astype(float))
            #print("xnp = " + str(xnp))
            shift = novelty_introduced_index_max - n
            #print("shift = " + str(shift))
            shifted_array = xnp
            if shift > 0:
                prefix_array = list(np.repeat(xnp[0], shift))
                #print("prefix_array = " + str(prefix_array))
                shifted_array = prefix_array + xnp
            #print("shifted_array = " + str(shifted_array))
            trunc_array = shifted_array[:len(xnp)]
            #print("trunc_arry = " + str(trunc_array))
            ta2_performance_shifted.append(np.array(trunc_array))
        sota_performance = [x['performance'] for x in sota_data]
        sota_performance_shifted = []
        for x,n in zip(sota_performance,novelty_introduced_indices):
            xnp = list(x.to_numpy().astype(float))
            shift = novelty_introduced_index_max - n
            shifted_array = xnp
            if shift > 0:
                prefix_array = list(np.repeat(xnp[0], shift))
                shifted_array = prefix_array + xnp
            trunc_array = shifted_array[:len(xnp)]
            sota_performance_shifted.append(np.array(trunc_array))
        ta2_average_performance = np.mean(ta2_performance_shifted, axis=0)
        sota_average_performance = np.mean(sota_performance_shifted, axis=0)

        plt.ylim(0.0, 1.05) # LBH: changed upper limit from 1.0 to 1.05 to clearly see 1.0 performance
        plt.ylabel('Performance')
        plt.xlabel('Episodes')

        ta2_x_values = np.array(range(len(ta2_average_performance)))
        sota_x_values = np.array(range(len(sota_average_performance)))

        plt.plot(ta2_x_values,
                 self.smooth(ta2_average_performance),
                 label='Agent',
                 color='black')
        plt.plot(sota_x_values,
                 self.smooth(sota_average_performance),
                 label='Baseline',
                 color='gold')
        plt.axvline(novelty_introduced_index_max,
                    0,
                    1,
                    label='Novelty Introduced',
                    color='gray',
                    linestyle=':')
        plt.legend()
        plt.xticks([])
        # ----- end: same code inside "for configuration" loop in generate_average_plots

        label = 'Averages for all data'
        plt.title(label)

        filename = 'averages_all.png'
        plt.savefig(os.path.join(path, filename), bbox_inches='tight')

        plt.cla()
        return

    def configuration_to_natural_words(self, configuration):
        self.log.debug(
            'configuration_to_natural_words(configuration={})'.format(configuration))
        if configuration == 'all':
            return 'All_Trials'
        if configuration in self.possible_novelties:
            return 'Novelty_' + configuration
        if configuration in self.possible_difficulties:
            return 'Difficulty_' + configuration.title()
        # Check if configuration of the form <novelty>_<difficulty>
        config = configuration.split('_')
        if len(config) == 2:
            novelty = config[0]
            difficulty = config[1]
            if (novelty in self.possible_novelties) and (difficulty in self.possible_difficulties):
                return 'Novelty_' + novelty + '_Difficulty_' + difficulty.title()
        return 'Case_Not_Found'

    @staticmethod
    def smooth(x):
        return x
        #return list(savgol_filter(x, 5, 3, mode='nearest'))
        # LBH: plots pretty smooth even without this

    def run_combine(self, path: str):
        self.log.info('run_combine(path={})'.format(path))
        measures = ['M1', 'M2', 'M2.1', 'M3', 'M3.1', 'M4', 'M4.1', 'OPTI', 'IPTI', 'APTI', 'AMOC',
                    'NRM', 'NRM_beta', 'M2.2', 'PRE_SOTA', 'PRE_TA2', 'POST_SOTA', 'POST_TA2']
        metrics = dict()
        metrics_novelty = dict()
        metrics_difficulty = dict()
        metrics_all = dict()

        # Collect metrics from individual CSV files
        for detection_condition in self.detection_conditions:
            if detection_condition not in metrics:
                metrics[detection_condition] = dict()
            for novelty_level in self.possible_novelties:
                if novelty_level not in metrics[detection_condition]:
                    metrics[detection_condition][novelty_level] = dict()
                for difficulty_level in self.possible_difficulties:
                    if difficulty_level not in metrics[detection_condition][novelty_level]:
                        metrics[detection_condition][novelty_level][difficulty_level] = dict()
                    csv_file_name = 'metrics_' + detection_condition + '_novelty_' + novelty_level \
                                    + '_difficulty_' + difficulty_level + '.csv'
                    csv_file_name = os.path.join(path, csv_file_name)
                    with open(csv_file_name, mode='r') as csv_file:
                        csv_reader = csv.DictReader(csv_file)
                        for row in csv_reader:
                            metrics[detection_condition][novelty_level][difficulty_level][
                                row['Measure']] = row
            # Collect metrics by novelty (for all difficulties)
            if detection_condition not in metrics_novelty:
                metrics_novelty[detection_condition] = dict()
            for novelty_level in self.possible_novelties:
                if novelty_level not in metrics_novelty[detection_condition]:
                    metrics_novelty[detection_condition][novelty_level] = dict()
                csv_file_name = 'metrics_' + detection_condition + '_novelty_' \
                                + novelty_level + '.csv'
                csv_file_name = os.path.join(path, csv_file_name)
                with open(csv_file_name, mode='r') as csv_file:
                    csv_reader = csv.DictReader(csv_file)
                    for row in csv_reader:
                        metrics_novelty[detection_condition][novelty_level][row['Measure']] = row
            # Collect metrics by difficulty (for all novelties)
            if detection_condition not in metrics_difficulty:
                metrics_difficulty[detection_condition] = dict()
            for difficulty_level in self.possible_difficulties:
                if difficulty_level not in metrics_difficulty[detection_condition]:
                    metrics_difficulty[detection_condition][difficulty_level] = dict()
                csv_file_name = 'metrics_' + detection_condition + '_difficulty_' \
                                + difficulty_level + '.csv'
                csv_file_name = os.path.join(path, csv_file_name)
                with open(csv_file_name, mode='r') as csv_file:
                    csv_reader = csv.DictReader(csv_file)
                    for row in csv_reader:
                        metrics_difficulty[detection_condition][difficulty_level][
                            row['Measure']] = row
            # Collect metrics for all novelties and difficulties
            if detection_condition not in metrics_all:
                metrics_all[detection_condition] = dict()
            csv_file_name = 'metrics_' + detection_condition + '_all_trials.csv'
            csv_file_name = os.path.join(path, csv_file_name)
            with open(csv_file_name, mode='r') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                for row in csv_reader:
                    metrics_all[detection_condition][row['Measure']] = row

        # Write combined metrics to CSV files
        for detection_condition in self.detection_conditions:
            csv_file_name = 'metrics_' + detection_condition + '.csv'
            csv_file_name = os.path.join(path, csv_file_name)
            with open(csv_file_name, mode='w') as csv_file:
                csv_file.write('Measure,Min,Max,Mean,Median,Norm-Median,StdDev\n')
                for measure in measures:
                    if measure == 'AM1':
                        csv_file.write('\nAdditional Metrics\n' + measure + '\n')
                    else:
                        csv_file.write('\n' + measure + '\n')
                    for novelty_level in self.possible_novelties:
                        for difficulty_level in self.possible_difficulties:
                            # print(str(metrics[detection_condition][novelty_level][
                            # difficulty_level]))
                            line_str = 'Level ' + novelty_level + ' - ' + difficulty_level
                            line_str += ',' + metrics[detection_condition][novelty_level][
                                difficulty_level][measure]['Min']
                            line_str += ',' + metrics[detection_condition][novelty_level][
                                difficulty_level][measure]['Max']
                            line_str += ',' + metrics[detection_condition][novelty_level][
                                difficulty_level][measure]['Mean']
                            line_str += ',' + metrics[detection_condition][novelty_level][
                                difficulty_level][measure]['Median']
                            line_str += ',' + metrics[detection_condition][novelty_level][
                                difficulty_level][measure]['Norm-Median']
                            line_str += ',' + metrics[detection_condition][novelty_level][
                                difficulty_level][measure]['StdDev']
                            csv_file.write(line_str + '\n')
                    for novelty_level in self.possible_novelties:
                        line_str = 'Level ' + novelty_level + ' - easy/medium/hard'
                        line_str += ',' + \
                                    metrics_novelty[detection_condition][novelty_level][measure][
                                        'Min']
                        line_str += ',' + \
                                    metrics_novelty[detection_condition][novelty_level][measure][
                                        'Max']
                        line_str += ',' + \
                                    metrics_novelty[detection_condition][novelty_level][measure][
                                        'Mean']
                        line_str += ',' + \
                                    metrics_novelty[detection_condition][novelty_level][measure][
                                        'Median']
                        line_str += ',' + \
                                    metrics_novelty[detection_condition][novelty_level][measure][
                                        'Norm-Median']
                        line_str += ',' + \
                                    metrics_novelty[detection_condition][novelty_level][measure][
                                        'StdDev']
                        csv_file.write(line_str + '\n')
                    for difficulty_level in self.possible_difficulties:
                        line_str = difficulty_level + ' - levels all'
                        line_str += ',' + metrics_difficulty[detection_condition][difficulty_level][
                            measure]['Min']
                        line_str += ',' + metrics_difficulty[detection_condition][difficulty_level][
                            measure]['Max']
                        line_str += ',' + metrics_difficulty[detection_condition][difficulty_level][
                            measure]['Mean']
                        line_str += ',' + metrics_difficulty[detection_condition][difficulty_level][
                            measure]['Median']
                        line_str += ',' + metrics_difficulty[detection_condition][difficulty_level][
                            measure]['Norm-Median']
                        line_str += ',' + metrics_difficulty[detection_condition][difficulty_level][
                            measure]['StdDev']
                        csv_file.write(line_str + '\n')
                    line_str = 'All - Levels all - easy/medium/hard'
                    line_str += ',' + metrics_all[detection_condition][measure]['Min']
                    line_str += ',' + metrics_all[detection_condition][measure]['Max']
                    line_str += ',' + metrics_all[detection_condition][measure]['Mean']
                    line_str += ',' + metrics_all[detection_condition][measure]['Median']
                    line_str += ',' + metrics_all[detection_condition][measure]['Norm-Median']
                    line_str += ',' + metrics_all[detection_condition][measure]['StdDev']
                    csv_file.write(line_str + '\n')
        return


if __name__ == '__main__':
    parser = optparse.OptionParser(usage="usage: %prog [options]")
    # Main arguments
    parser.add_option("--logfile",
                      dest="logfile",
                      help="Filename if you want to write the log to disk.")
    parser.add_option("--experimentid",
                      dest="experiment_id")
    parser.add_option("--baseline",
                      dest="baseline_results_file")
    parser.add_option("--agent",
                      dest="agent_results_file")
    # Optional arguments
    parser.add_option("--outputdirectory",
                      dest="output_directory",
                      default="./")
    parser.add_option("--debug",
                      dest="debug",
                      action="store_true",
                      help="Set logging level to DEBUG from INFO.",
                      default=False)
    parser.add_option("--printout",
                      dest="printout",
                      action="store_true",
                      help="Print output to the screen at given logging level.",
                      default=False)
    parser.add_option("--plottrials",
                      dest="plottrials",
                      action="store_true",
                      help="Output plot images for individual trial performance.",
                      default=False)
    parser.add_option("--plotamocs",
                      dest="plotamocs",
                      action="store_true",
                      help="Output plot images for individual trial AMOC curves.",
                      default=False)
    parser.add_option("--details",
                      dest="details",
                      action="store_true",
                      help="Print details about each episode and write per-trial metrics.",
                      default=False)
    parser.add_option("--extras",
                      dest="extras",
                      action="store_true",
                      help="Print extra metrics when writing per-trial metrics.",
                      default=False)
    parser.add_option("--TA1",
                      dest="ta1_team_name",
                      help="Name of the baseline agent.",
                      default="Baseline")
    parser.add_option("--TA2",
                      dest="ta2_team_name",
                      help="Name of the target agent.",
                      default="Agent")
    parser.add_option("--domain",
                      dest="domain_name",
                      help="Name of the domain (e.g., SmartEnv).",
                      default="SmartEnv")
    parser.add_option("--usenoveltylist",
                      dest="use_novelty_list",
                      action="store_true",
                      help="Use the manually defined novelty list in the class __init__().",
                      default=False)
    parser.add_option("--usedifficultylist",
                      dest="use_difficulty_list",
                      action="store_true",
                      help="Use the manually defined difficulty list in the class __init__().",
                      default=False)
    parser.add_option("--usevisibilitylist",
                      dest="use_visibility_list",
                      action="store_true",
                      help="Use the manually defined visibility list in the class __init__().",
                      default=False)
    (options, args) = parser.parse_args()

    agent = Analysis(options)
    agent.analyze_experiment()

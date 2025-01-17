# run-exp.py
#
# Run agent locally with file-based trial/episode data.
#
# The --noveltydetection option can be "CDD" (concept drift detection)
# or "DDD" (data drift detection). If given, novelty detection is done
# from the beginning of a trial. If not given, then no novelty
# detection is done.
#
# If the --learning option is given, then the agent will re-train after
# an episode is complete. If --noveltydetection is given, then re-training
# begins only after the episode in which novelty is detected. If no
# --noveltydetection is given, the re-training begins with the first
# episode. If the --learning option is not given, then no re-training
# is done.

import os
import sys
import gzip
import shutil
import argparse
import sqlite3
import json
import random
import logging
from SmartEnvAgent import SmartEnvAgent

DEBUG = False
BUDGET = 0.5
NOVELTY_LEVELS = [200, 201, 202, 203, 204, 205, 206, 207, 208]
NOVELTY_DIFFICULTIES = ['easy', 'medium', 'hard']
TRAINING_FILE_PREFIX = 'smartenv'
MODEL_FILE_PREFIX = 'model'

def run_experiment(args, logger): # serial
    experiment_id = args.experiment_id
    experiment_path = args.experiment_path
    output_file = args.output_file
    trial_start = args.trial_start
    trial_end = args.trial_end
    novelty_detection = args.novelty_detection
    learning = args.learning
    #agent.experiment_start() # skip to allow parallelism
    trials = get_trials(experiment_id, experiment_path)
    trials = get_testing_trials(trials, trial_start, trial_end)
    if trials:
        # Output CSV header
        with open(output_file, 'w') as out_file:
            out_file.write('trial_index,trial_id,novelty_level,novelty_difficulty,novelty_visibility,episode_index,performance,novelty_initiated,novelty_probability,novelty_threshold\n')
        for trial in trials:
            agent = SmartEnvAgent(logger, novelty_detection)
            run_trial(agent, trial, experiment_id, experiment_path, output_file, novelty_detection, learning)
    #agent.experiment_end() # skip to allow parallelism
    return

def run_trial(agent, trial, experiment_id, experiment_path, output_file, novelty_detection, learning):
    """novelty_detection is None, CDD or DDD. learning is True or False."""
    if agent.end_experiment_early:
        return
    trial_index = trial['trial']
    trial_id = trial['experiment_trial_id']
    novelty_level = trial['novelty']
    novelty_visibility = trial['novelty_visibility']
    novelty_difficulty = trial['difficulty']
    training_data_path = os.path.join(experiment_path, TRAINING_FILE_PREFIX)
    agent.reset_model(training_data_path)
    novelty_description = {'trial_id': trial_id, 'novelty': novelty_level, 'visibility': novelty_visibility, 'difficulty': novelty_difficulty}
    agent.trial_start(trial_index, novelty_description)
    novelty_detected = False
    agent.testing_start()
    trial_episodes = get_episodes(trial_id, experiment_id, experiment_path)
    for episode in trial_episodes:
        if agent.end_experiment_early:
            break
        episode_index = episode['episode_index']
        novelty_initiated = (episode['novelty_initiated'] == 1)
        num_correct = 0
        num_instances = 0
        agent.testing_episode_start(episode_index)
        for instance in get_instances(episode_index, trial_id, experiment_id, experiment_path):
            if agent.end_experiment_early:
                break
            num_instances += 1
            #num_instances_all += 1
            feature_label, feature_vector = instance

            if DEBUG:
                print('------------------------------')
                print(f'EPISODE {episode_index}, INSTANCE {num_instances - 1}')
                print(f'  feature_vector: {feature_vector}\n  label: {feature_label}')

            novelty_indicator = None
            if novelty_visibility == 1:
                novelty_indicator = novelty_initiated
            prediction = agent.testing_instance(feature_vector, novelty_indicator)
            # Evaluate prediction, update performance and feedback dict
            if prediction == feature_label:
                num_correct += 1
            performance = num_correct / num_instances
            # if novelty_detection=CDD|DDD and learning=T, then feedback only after novelty_detected, but keep going
            # if novelty_detection=CDD|DDD and learning=F, then no feedback ever and stop after novelty
            # if novelty_detection=None and learning=T, then always feedback and keep going
            # if novelty_detection=None and learning=F, then no feedback ever, but keep going
            feedback = None
            if learning:
                if (novelty_detection is None) or novelty_detected:
                    if random.random() < BUDGET:
                        feedback = feature_label
            agent.testing_performance(performance, feedback)
        # Compute episode performance and feedback
        feedback = None
        novelty_probability, novelty_threshold, novelty, novelty_characterization = agent.testing_episode_end(performance, feedback)
        # Process novelty response
        if (novelty_probability >= novelty_threshold):
            novelty_detected = True
        # Output episode info
        with open(output_file, 'a') as out_file:
            out_file.write(f'{trial_index},{trial_id},{novelty_level},{novelty_difficulty},{novelty_visibility},{episode_index},{performance},{novelty_initiated},{novelty_probability},{novelty_threshold}\n')
        if (not learning) and novelty_detection and novelty_initiated and novelty_detected:
            break # stop early
        # Save model at the end of each episode, overwriting previous model (uncomment, if desired)
        #if learning:
        #    model_path = os.path.join(experiment_path, MODEL_FILE_PREFIX)
        #    agent.save_model(model_path)
        if DEBUG:
            break

    agent.testing_end()
    agent.trial_end()
    return

def get_trials(experiment_id, experiment_path):
    db_file = os.path.join(experiment_path, experiment_id, 'experiment_trial_trial_episode.db')
    trials = get_table_as_dicts(db_file, 'experiment_trial')
    return trials

def get_testing_trials(trials, trial_start, trial_end, visible=False):
    """Return trials whose trial index is between trial_start and trial_end.
    If visible=True, then include trials with visibility=1; otherwise only
    include trials with visibility=0."""
    testing_trials = []
    for trial in trials:
        if (trial['trial'] >= trial_start) and (trial['trial'] <= trial_end) and (trial['novelty'] in NOVELTY_LEVELS) and (trial['difficulty'] in NOVELTY_DIFFICULTIES):
            if visible or (trial['novelty_visibility'] == 0):
                testing_trials.append(trial)
    return testing_trials

def get_episodes(trial_id, experiment_id, experiment_path):
    # First, gunzip the database
    gz_file = os.path.join(experiment_path, experiment_id, trial_id, 'experiment_trial_trial_episode.db.gz')
    temp_db_file = os.path.join(experiment_path, experiment_id, trial_id, 'temp_database.db')
    with gzip.open(gz_file, 'rb') as f_in:
        with open(temp_db_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    episodes = get_table_as_dicts(temp_db_file, 'trial_episode')
    os.remove(temp_db_file)
    return episodes

def get_instances(episode_index, trial_id, experiment_id, experiment_path):
    episode_filename = 'ep_' + str(episode_index) + '.csv'
    episode_file = os.path.join(experiment_path, experiment_id, trial_id, episode_filename)
    instances = []
    with open(episode_file, 'r') as file:
        for index, line in enumerate(file):
            if index == 0: # skip first line
                continue
            line = line.replace("'", '"')
            line = line.replace('None', 'null')
            instance = json.loads(line)
            instances.append(instance)
    return instances

def get_table_as_dicts(db_file, table_name):
    conn = sqlite3.connect(db_file)
    # Configure the connection to return rows as dictionaries
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    # Execute the query to retrieve all rows from the table
    cursor.execute(f"SELECT * FROM {table_name}")
    # Fetch all rows and convert them to dictionaries
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experimentid', dest='experiment_id', type=str, required=True)
    parser.add_argument('--experimentpath', dest='experiment_path', type=str, required=True)
    parser.add_argument('--output', dest='output_file', type=str, required=True)
    parser.add_argument('--trialstart', dest='trial_start', type=int, default=0)
    parser.add_argument('--trialend', dest='trial_end', type=int, default=29)
    parser.add_argument('--logfile', dest='log_file', type=str, default='log.txt')
    parser.add_argument('--noveltydetection', dest='novelty_detection', type=str, default=None)
    parser.add_argument('--learning', dest='learning', action='store_true')
    args = parser.parse_args()
    return args

def get_logger(log_file):
    logger = logging.getLogger('TA2')
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger

def main():
    args = parse_arguments()
    if args.novelty_detection is not None:
        if args.novelty_detection not in ['CDD', 'DDD']:
            sys.exit('Error: invalid value for noveltydetection')
    logger = get_logger(args.log_file)
    run_experiment(args, logger)
    return

if __name__ == "__main__":
    main()


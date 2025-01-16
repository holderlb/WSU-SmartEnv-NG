# ActivityLearner.py

import numpy as np
from datetime import datetime, timedelta, timezone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import joblib
import random
import csv

DEBUG = False
MAX_EXAMPLES = 100000
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = None
RF_MIN_SAMPLES_SPLIT = 20

# Util for managing multiple testbeds.
class ActivityManager:

    def __init__(self, testbed_sample_feature_vectors = {}):
        """For each testbed_id:feature_vector pair in the input, create and initialize an ActivityLearner."""
        self.current_test_bed = None
        self.test_beds = dict()
        for testbed_id, feature_vector in testbed_sample_feature_vectors.items():
            self.test_beds[testbed_id] = ActivityLearner(feature_vector)
        return

    # Check we are on correct test bed
    def selector(self, feature_vector):
        # Get id
        test_bed_id = feature_vector['testbed_id']
        # Update current id
        if self.current_test_bed is None:
            print('Last test bed: ', self.current_test_bed)
            self.current_test_bed = test_bed_id
            print('Setting current testbed: ', self.current_test_bed)
        elif test_bed_id != self.current_test_bed:
            print('Last test bed: ', self.current_test_bed)
            self.current_test_bed = test_bed_id
            print('Updating current testbed: ', self.current_test_bed)
        # Make new AL if not yet made
        if self.current_test_bed not in self.test_beds:
            print('Creating testbed: ', self.current_test_bed)
            self.test_beds[self.current_test_bed] = ActivityLearner(feature_vector)
            print('Current test beds:', self.test_beds.keys())
        return

    def predict(self, feature_vector):
        self.selector(feature_vector)
        return self.test_beds[self.current_test_bed].predict(feature_vector)

    def train_model(self):
        for testbed_id in self.test_beds.keys():
            self.test_beds[testbed_id].train()
        return
    
    def save_model(self, filename):
        """Save models, classes and datasets for each testbed."""
        for testbed_id in self.test_beds.keys():
            self.test_beds[testbed_id].save(filename + '-' + testbed_id)
        return
    
    def reset_model(self, filename):
        """Create new model, load training data, and train, for each testbed."""
        for testbed_id in self.test_beds.keys():
            self.test_beds[testbed_id].reset_model(filename + '-' + testbed_id)
        return

    def is_trained(self, feature_vector):
        self.selector(feature_vector)
        return self.test_beds[self.current_test_bed].is_trained()
    
    def get_num_examples(self):
        # Return list of (testbed_id, #training_examples).
        result = []
        for testbed_id in self.test_beds.keys():
            num_egs = len(self.test_beds[testbed_id].data)
            result.append((testbed_id, num_egs))
        return result
    
class ActivityLearner:

    def __init__(self, sample_feature_vector = None):
        self.util = ALUtil()
        self.data = list()
        self.feature_arr_data = list()
        self.last_sensor_list = None
        self._is_trained = False # Is the model trained on all currently-stored data
        self._is_inited = False
        self.model = None
        self.activitynames = ["wash_dishes", "relax", "personal_hygiene", "bed_toilet_transition", "cook", "sleep",
                              "take_medicine", "leave_home", "work", "enter_home", "eat"]
        self.currentSecondsOfDay = 0
        self.currentTimestamp = 0
        self.dayOfWeek = 0
        self.dominant = 0
        self.sensornames = []
        self.sensortimes = []
        self.data = []
        self.dstype = []
        self.numwin = 0
        self.prevwin1 = 0
        self.prevwin2 = 0
        self.wincnt = 0
        self.IgnoreOther = 0
        self.ClusterOther = 0
        self.NoOverlap = 0  # Do not allow windows to overlap
        self.Mode = "TRAIN"  # TRAIN, TEST, CV, PARTITION, ANNOTATE, WRITE
        self.NumActivities = 0
        self.NumSensors = 0
        self.NumSetFeatures = 14
        self.SecondsInADay = 86400
        self.MaxWindow = 1
        self.weightinc = 0.01
        self.windata = np.zeros((self.MaxWindow, 3), dtype=int)
        self.NumFeatures = None
        if sample_feature_vector:
            self.data_init(sample_feature_vector)
        self.last_formatted_data = None # set during call to predict, used for add_data
        self.last_feature_arr = None # set during call to predict, used for get_feature_arr
        return
    
    def get_new_model(self):
        model = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, bootstrap=True, criterion="entropy",
                                         min_samples_split=RF_MIN_SAMPLES_SPLIT, max_depth=RF_MAX_DEPTH,
                                         class_weight='balanced')
        return model
    
    def save(self, filename):
        # Save data
        with open(filename + '-data.csv', 'w') as file:
            file.write('datetime,sensor,value,activity\n')
            for d in self.data:
                # Format date properly
                dtstr = d[0].strftime("%Y-%m-%d %H:%M:%S") + f".{d[0].microsecond:06d}"
                file.write(f'{dtstr},{d[1]},{d[2]},{d[3]}\n')
        # Save model
        joblib.dump(self.model, filename + '-model.joblib')
        return
    
    def reset_model(self, filename):
        self.reset_data()
        # Load data
        with open(filename + '-data.csv', 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                dt = datetime.strptime(row['datetime'], "%Y-%m-%d %H:%M:%S.%f")
                dt_with_tz = dt.replace(tzinfo=timezone.utc)
                d = [dt_with_tz, row['sensor'], float(row['value']), row['activity']]
                self.data.append(d)
        self.data = self.data[-MAX_EXAMPLES:]
        # Train new model
        self.train() # also sets self.feature_arr_data
        return
    
    def reset_init(self):
        self._is_inited = False
        return

    def is_trained(self):
        return self._is_trained

    def data_init(self, feature_vector):
        flat_vector = self.flatten_feature_vector(feature_vector)
        self.last_sensor_list = flat_vector
        # Make sure its all zeroed
        for key in self.last_sensor_list.keys():
            self.last_sensor_list[key] = 0.0
            self.sensornames.append(key)
            self.sensortimes.append(0)
            self.dstype.append('n')
        # Update sensor number here
        self.NumSensors = len(self.sensortimes)
        self._is_inited = True
        return

    def add_data(self, feature_label):
        # Updates last formatted data with given feature label and adds to self.data.
        activity = feature_label['action']
        self.last_formatted_data[3] = activity
        self.data.append(self.last_formatted_data)
        self.data = self.data[-MAX_EXAMPLES:] # keep only most recent examples
        self._is_trained = False
        if DEBUG:
            print(f'AL:ADD_DATA:\n  formatted_data: {self.last_formatted_data}')
        return

    def reset_data(self):
        self.data = list()
        self.feature_arr_data = list()
        self._is_trained = False
        return

    def train(self):
        if not self._is_trained:
            prepared_data, prepared_labels = self.read_data(self.data, flag=1)
            self.feature_arr_data = prepared_data
            self.train_model(prepared_data, prepared_labels)
            self._is_trained = True
        return

    def predict(self, feature_vector):
        flat_vector = self.flatten_feature_vector(feature_vector)
        found_sensor = self.triggered_sensor(flat_vector)
        formatted_data = self.data_formatter(found_sensor, feature_vector, {'action': 'eat'}) # dummy action
        data, _ = self.read_data([formatted_data], flag=0)
        self.last_feature_arr = data[0]
        activity_number = self.test_model(data)[0]
        predicted_activity = self.activitynames[activity_number]
        formatted_data[3] = predicted_activity
        self.last_formatted_data = formatted_data
        prediction = {'action': predicted_activity}
        if DEBUG:
            print(f'AL:PREDICT:\n  formatted_data: {formatted_data}\n  prediction: {prediction}')
        return prediction
    
    def get_feature_array(self):
        return self.last_feature_arr

    # ---- Formatting functions -------
    # Data comes in like door_sensor{}, get rid of that and flatten
    # noinspection PyMethodMayBeStatic
    def flatten_feature_vector(self, feature_vector):
        sensor_types = ["door_sensors", "light_switch_sensors",
                        "motion_sensors", "motion_area_sensors"]

        flattened_vector = []
        for typed in sensor_types:
            flattened_vector = flattened_vector + feature_vector[typed]

        # Data now looks like [{'id': 'MA100', 'value': 0.0}, ...
        flattened_list = dict()
        for element in flattened_vector:
            flattened_list[element['id']] = element['value']

        return flattened_list

    # Find the triggered sensor value from the last input
    def triggered_sensor(self, flat_vector):
        # We don't care what this, but it has to be an actual sensor incase of error
        triggered_sensor_name = list(flat_vector.keys())[0]

        # Find the differing sensor value and save the name
        for name in flat_vector.keys():
            if flat_vector[name] != self.last_sensor_list[name]:
                triggered_sensor_name = name
                break

        # Update last checked list here
        self.last_sensor_list = flat_vector
        return triggered_sensor_name

    # noinspection PyMethodMayBeStatic
    def data_formatter(self, found_sensor, feature_vector, feature_label):
        # Raw data looks like
        # feature_vector={'time_stamp': 1603247296.6152952, 'cart_position': 0.004881350392732478,
        # 'pole_angular_velocity': 0.004488318299689688}  feature_label={'action': 'left'}

        #date_time = datetime.utcfromtimestamp(feature_vector['time_stamp']) # deprecated
        date_time = datetime.fromtimestamp(feature_vector['time_stamp'], tz=timezone.utc)
        senor_name = found_sensor
        flat_vector = self.flatten_feature_vector(feature_vector)
        sensor_value = flat_vector[found_sensor]
        activity_label = feature_label['action']

        return [date_time, senor_name, sensor_value, activity_label]

    def test_model(self, data):
        if self.model:
            predictions = self.model.predict(data)
        else:
            print("AL:TEST_MODEL: WARNING: Random prediction.")
            predictions = [random.randint(0, len(self.activitynames)-1)]
        if DEBUG:
            print(f'AL:TEST_MODEL:\n  read_data: {data}\n  predictions: {predictions}')
        return predictions

    def train_model(self, data, labels):
        self.model = self.get_new_model()
        self.model.fit(data, labels)
        return

    def read_data(self, sensor_data_set, flag):
        data = []
        labels = []

        if flag == 0:
            first_event_flag = 0
        else:
            first_event_flag = 1

        self.currentTimestamp = None

        for sensor_data in sensor_data_set:

            # Chris new ordering here
            dt = sensor_data[0]
            sensorid = sensor_data[1]
            sensorstatus = sensor_data[2]
            alabel = sensor_data[3]

            if self.currentTimestamp is None:
                self.currentTimestamp = dt

            self.currentSecondsOfDay = self.util.compute_seconds(dt)
            self.dayOfWeek = dt.weekday()
            previous_timestamp = self.currentTimestamp
            self.currentTimestamp = dt

            snum1 = self.util.find_sensor(self.sensornames, sensorid)
            timediff = self.currentTimestamp - previous_timestamp

            # reset the sensor times and the window
            if first_event_flag == 1 or timediff.days < 0 or timediff.days > 1:
                for i in range(self.NumSensors):
                    self.sensortimes[i] = self.currentTimestamp - timedelta(days=1)
                first_event_flag = 0

            if sensorstatus == "ON" and self.dstype[snum1] == 'n':
                self.dstype[snum1] = 'm'

            self.sensortimes[snum1] = self.currentTimestamp  # last time sensor fired

            self.NumFeatures = self.NumSetFeatures + (2 * self.NumSensors)
            tempdata = np.zeros(self.NumFeatures)

            end = 0
            if alabel != "Other_Activity" or self.IgnoreOther == 0:
                end = self.compute_feature(sensorid, tempdata)

            if end == 1:  # End of window reached, add feature vector
                if alabel == "Other_Activity" and self.ClusterOther == 1:
                    labels.append(-1)
                else:
                    labels.append(self.util.find_activity(self.Mode, self.activitynames, self.NumActivities, alabel))

                data.append(tempdata)

        return data, labels

    # Compute the feature vector for each window-size sequence of sensor events.
    # 0: time of the last sensor event in window (hour)
    # 1: time of the last sensor event in window (seconds)
    # 2: day of the week for the last sensor event in window
    # 3: window size in time duration
    # 4: time since last sensor event
    # 5: dominant sensor for previous window
    # 6: dominant sensor two windows back
    # 7: last sensor event in window
    # 8: last sensor location in window
    # 9: last motion sensor location in window
    # 10: complexity of window (entropy calculated from sensor counts)
    # 11: change in activity level between two halves of window
    # 12: number of transitions between areas in window
    # 13: number of distinct sensors in window
    # 14 - NumSensors+13: counts for each sensor
    # NumSensors+14 - 2*NumSensors+13: time since sensor last fired (<= SECSINDAY)

    def compute_feature(self, sensorid1, tempdata):
        lastlocation = -1
        lastmotionlocation = -1
        complexity = 0
        maxcount = 0
        numtransitions = 0
        numdistinctsensors = 0

        self.windata[self.wincnt][0] = self.util.find_sensor(self.sensornames, sensorid1)
        self.windata[self.wincnt][1] = self.currentSecondsOfDay
        self.windata[self.wincnt][2] = self.dayOfWeek

        if self.wincnt < (self.MaxWindow - 1):  # not reached end of window
            self.wincnt += 1
            return 0
        else:  # reached end of window
            wsize = self.MaxWindow
            scount = np.zeros(self.NumSensors, dtype=int)

            # Determine the dominant sensor for this window
            # count the number of transitions between areas in this window
            for i in range(self.MaxWindow - 1, self.MaxWindow - (wsize + 1), -1):
                scount[self.windata[i][0]] += 1
                id = self.windata[i][0]

                if lastlocation == -1:
                    lastlocation = id
                if (lastmotionlocation == -1) and (self.dstype[id] == 'm'):
                    lastmotionlocation = id
                if i < self.MaxWindow - 1:  # check for transition
                    id2 = self.windata[i + 1][0]
                    if id != id2:
                        if (self.dstype[id] == 'm') and (self.dstype[id2] == 'm'):
                            numtransitions += 1

            for i in range(self.NumSensors):
                if scount[i] > 1:
                    ent = float(scount[i]) / float(wsize)
                    ent *= np.log2(ent)
                    complexity -= float(ent)
                    numdistinctsensors += 1

            if np.mod(self.numwin, self.MaxWindow) == 0:
                self.prevwin2 = self.prevwin1
            self.prevwin1 = self.dominant
            self.dominant = 0
            for i in range(self.NumSensors):
                if scount[i] > maxcount:
                    maxcount = scount[i]
                    self.dominant = i

            # Attribute 0..2: time of last sensor event in window
            tempdata[0] = self.windata[self.MaxWindow - 1][1] / 3600  # hour of day
            tempdata[1] = self.windata[self.MaxWindow - 1][1]  # seconds of day
            tempdata[2] = self.windata[self.MaxWindow - 1][2]  # day of week

            # Attribute 3: time duration of window in seconds
            time1 = self.windata[self.MaxWindow - 1][1]  # most recent sensor event
            time2 = self.windata[self.MaxWindow - wsize][1]  # first sensor event in window
            if time1 < time2:
                duration = time1 + (self.SecondsInADay - time2)
            else:
                duration = time1 - time2
            tempdata[3] = duration  # window duration

            timehalf = self.windata[int(self.MaxWindow - (wsize / 2))][1]  # halfway point
            if time1 < time2:
                duration = time1 + (self.SecondsInADay - time2)
            else:
                duration = time1 - time2
            if timehalf < time2:
                halfduration = timehalf + (self.SecondsInADay - time2)
            else:
                halfduration = timehalf - time2
            if duration == 0.0:
                activitychange = 0.0
            else:
                activitychange = float(halfduration) / float(duration)

            # Attribute 4: time since last sensor event
            time2 = self.windata[self.MaxWindow - 2][1]
            if time1 < time2:
                duration = time1 + (self.SecondsInADay - time2)
            else:
                duration = time1 - time2
            tempdata[4] = duration

            # Attribute 5..6: dominant sensors from previous windows
            tempdata[5] = self.prevwin1
            tempdata[6] = self.prevwin2

            # Attribute 7: last sensor id in window
            tempdata[7] = self.util.find_sensor(self.sensornames, sensorid1)

            # Attribute 8: last location in window
            tempdata[8] = lastlocation

            # Attribute 9: last motion location in window
            tempdata[9] = lastmotionlocation

            # Attribute 10: complexity (entropy of sensor counts)
            tempdata[10] = complexity

            # Attribute 11: activity change (activity change between window halves)
            tempdata[11] = activitychange

            # Attribute 12: number of transitions between areas in window
            tempdata[12] = numtransitions

            # Attribute 13: number of distinct sensors in window
            # tempdata[13] = numdistinctsensors
            tempdata[13] = 0

            # Attributes NumSetFeatures..(NumSensors+(NumSetFeatures-1))
            weight = 1
            for i in range(self.MaxWindow - 1, self.MaxWindow - (wsize + 1), -1):
                tempdata[self.windata[i][0] + self.NumSetFeatures] += 1 * weight
                weight += self.weightinc

            # Attributes NumSensors+NumSetFeatures..(2*NumSensors+(NumSetFeatures-1))
            # time since each sensor fired
            for i in range(self.NumSensors):
                difftime = self.currentTimestamp - self.sensortimes[i]

                # There is a large gap in time or shift backward in time
                if difftime.total_seconds() < 0 or (difftime.days > 0):
                    tempdata[self.NumSetFeatures + self.NumSensors + i] = self.SecondsInADay
                else:
                    tempdata[self.NumSetFeatures + self.NumSensors + i] = difftime.total_seconds()

            for i in range(self.MaxWindow - 1):
                self.windata[i][0] = self.windata[i + 1][0]
                self.windata[i][1] = self.windata[i + 1][1]
                self.windata[i][2] = self.windata[i + 1][2]
                self.numwin += 1
            if self.NoOverlap == 1 and self.Mode != "ANNOTATE":  # windows should not overlap
                self.wincnt = 0

            return 1


# Supporting utility functions that don't a(e)ffect logic
class ALUtil:

    def __init__(self):

        return

    # Return the index of a specific activity name in the list of activities
    # If activity is not found in the list, add the new name
    @staticmethod
    def find_activity(mode, activity_names, num_activities, aname):
        try:
            i = activity_names.index(aname)
            return i
        except Exception as e:
            if mode == "TEST":
                print("Could not find activity ", aname)
                return -1
            else:
                activity_names.append(aname)
            num_activities += 1
            return num_activities - 1

    # Return the index of a specific sensor name in the list of sensors
    @staticmethod
    def find_sensor(sensor_names, sensor_name):
        try:
            i = sensor_names.index(sensor_name)
            return i
        except Exception as e:
            print("Could not find sensor ", sensor_name)
            return -1

    # Compute the number of seconds past midnight for the current datetime
    @staticmethod
    def compute_seconds(dt):
        seconds = dt - dt.replace(hour=0, minute=0, second=0)
        return int(seconds.total_seconds())



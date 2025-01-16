# WSU-SmartEnv-NG
Standalone novelty generator for WSU's Smart Environment domain.

## Introduction

This repository provides a framework for testing novelty-aware agents in the Smart Environment
domain, developed as part of the DARPA SAIL-ON program. For more information about the domain,
see the WSU-SAILON-NG repository [here](https://github.com/holderlb/WSU-SAILON-NG). That repository
is based on a client-server arrangement between the agent and WSU's server. This was
done in order to not expose any information about the novelties before evaluation. Here, a
local version of the agent is implemented, and data is available (see
below) so that evaluation can be done locally.

The agent is evaluated on a set of trails for each type of novelty. Each trial provides a sequence of
episodes, where each episode is one day in the life of the smart environment (i.e., smart home).
At some point in the sequence of episodes, a novelty is introduced and persists for the
remainder of the trial. There are 8 levels of novelty (again, see the repo referenced above), including
trials with no novelty (level 0). Furthermore, each novelty level is categorized as easy, medium, or hard
difficulty based on the likely impact of the novelty on the agent's performance. The agent is evaluated on
30 trials for each novelty level/difficulty, and each trial consists of 200 episodes.

The agent is evaluated based on its activity recognition performance and its ability to detect
novelty. The agent provided here uses the SciKit Learn Random Forest model for the classifier and
provides two different novelty detectors based on traditional concept and data drift detectors
as implemented in the [Frouros](https://github.com/IFCA-Advanced-Computing/frouros) package, which
is included here as submodule.

## Run Experiment

We recommend you run the WSU-SmartEnv in a standalone conda environment. The `smartenv.yml` file can be
used to create an appropriate environment using the command:

```
conda env create -f smartenv.yml
```

Next, the datasets referenced below stored locally. Let's assume you expand these
files into the `./data` folder, creating subfolders `smartenv_012345` and `smartenv_0678`,
and training data files `smartenv-12bc911e4b7e-data.csv`, `smartenv-4cdb124c42f4-data.csv`,
and `smartenv-72c576abcdda-data.csv`, one for each smart environment testbed ID.


Then, you can run the evaluation using the following sample command:

```
python run-exp.py --experimentid "smartenv-012345" --experimentpath "./data" --output "results.csv"
--logfile "log.txt" --trialstart 0 --trialend 29
```

This command will run the SmartEnvAgent on novelty levels 0-5 using all 30 trials for each difficulty level
(easy, medium, hard). The `--trialstart` and `--trialend` options allow you to run small experiments with
fewer trials or run subsets of trials in parallel, since each trial is independent of other trials.

A log of the run will be appended to `log.txt`, and the results are written in CSV format to `results.csv`.
If you would like the limit the novelty levels or difficulty levels used in the experiment, you can modify
the `NOVELTY_LEVELS` and `NOVELTY_DIFFICULTIES` lists at the top of `run-exp.py` accordingly. Note that the
novelty levels are prefaced with "20" for legacy reasons.

## Run Analysis

The `analyze.py` script is used to analyze the results of the experiment. The analyze script requires results
from a baseline agent, as well as the agent being evaluated. 
## Data

Data for the various novelty levels are available at the following links due to their size.

* Novelty Levels 0-5: Trials and episodes used to evaluate the agent on novelty levels 0-5.
The data is available [here](https://ailab.wsu.edu/AIQ/smartenv/smartenv_012345.zip) (19GB).

* Novelty Levels 0,6-8: Trials and episodes used to evaluate the agent on novelty levels 0, 6, 7 and 8.
The level 0 trials are different than the above level 0 trials, but are added here so that level 0
trials can be included along with evaluation of level 6-8, if desired.
The data is available [here](https://ailab.wsu.edu/AIQ/smartenv/smartenv_0678.zip) (11GB).

* Training data: Training data for each of the three smart environment testbeds. These three CSV files should be
put in the same location as the unzipped folders above.
The data is available [here](https://ailab.wsu.edu/AIQ/smartenv/smartenv_train.zip) (1.5MB)

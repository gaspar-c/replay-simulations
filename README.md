# replay-simulations

This repository contains code to reproduce [Fig. 4](https://www.pnas.org/doi/10.1073/pnas.2312281120#fig04) of the manuscript 
['Structure and function of the hippocampal CA3 module' by Sammons et al. (2024), PNAS](https://www.pnas.org/doi/10.1073/pnas.2312281120).

There are 4 python scripts titled `get_panel*.py`. Each will reproduce the corresponding panels **B**, **C**<sup>(*)</sup>,
**D** and **E** of the manuscript figure. Since each of the 4 scripts can take many hours to run, an additional
`get_test.py` script runs a small test version of the network, to rapidly check that the code can be successfully 
compiled and run.

## How to run

The code was tested with Python 3.8.18. The necessary packages and dependencies can be installed with pip or anaconda 
from the `requirements.txt` file.

If using anaconda, it is necessary to add the `conda-forge` channel before installing `brian2`:
````
conda config --add channels conda-forge
````

Run Scripts:
- `get_test.py`, performs smaller and faster simulation, to check that everything runs without errors
- `get_panelB.py`, generates **Fig. 4B**
- `get_panelC.py`, generates **Fig. 4C**<sup>(*)</sup>
- `get_panelD.py`, generates **Fig. 4D**
- `get_panelE.py`, generates **Fig. 4E**

(*) By default, `get_panelC.py` generates the left part of panel **C**, for N = 50,000. 
To generate the right part of the panel, the script must be changed such that N = 100,000.

## Contents

Each `get*` script consists of a group of *tests* that can be run in parallel, provided more than one CPU core exists. 
A *test* is a simulation of a given spiking network to determine whether the replay of an embedded sequence can occur. It consists of the following steps:
- A pseudo-random EI spiking network is created with an embedded sequence of 10 assemblies;
- To balance the activity, the network runs for 10 seconds with an active STDP mechanism in the I-to-E synapses;
- After the balancing, STDP is turned off and the first assembly in the sequence is stimulated 5 times, in intervals of 1 second. After each stimulation, we record whether the replay succeeded or failed.

Each of them will produce a folder in the `outputs/` directory with the test results, containing the following files:
- `0_group_log.log`, a log file of the entire test group;
- `0_group_results.txt`, a file with the summarised test results for each stimulation of each network in the test group;
- `#.log`, the log file for test #;
- `#_[start-end].png`, a figure with a snapshot of test # in the time interval `start-end`.

Additionally, for panels **C**-**E** a figure `panel*_results.png` with the summarised test group results, 
corresponding to the figure in the manuscript, is created outside the group folder.

These scripts use the following python code under the `code_files/` directory:
 - `run_tests.py`, process and run group of tests;
 - `parameters.py`, simulation parameters;
 - `network.py`, create and setup spiking network;
 - `tests.py`, manipulate and test spiking network;
 - `plots.py`, plot simulations;
 - `plot_group_results.py`, plot summary results of test group;
 - `aux_functions.py`,  auxiliary functions;
 - `detect_peaks.py`, supporting function to detect peaks [1].


## References
 [1]: Marcos Duarte, https://github.com/demotu/BMC
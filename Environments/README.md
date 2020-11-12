# Environments

The code has been tested on several computers and their respective environment are provided in the yml-files. All computers use Ubuntu.

If you are unsuccessful using one environment, try another. When trying to use the environments on other computers than these four, we have had problems in some instances, and environment 3 and 4 have had the highest success rate (probably because computer 3 and 4 are newer and therefore may have later versions of some packages).

### Computer 1

environment_1.yml

Run `conda activate eeg1` to active it.

Uses tensorflow-gpu==1.10.1

### Computer 2

environment_2.yml

Run `conda activate eeg2` to active it.

Uses tensorflow-gpu==1.13.1. The scripts have to be modified by changing the imports from `tensorflow.keras` to `keras`.

### Computer 3

environment_3.yml

Run `conda activate eeg3` to active it.

Uses tensorflow-gpu==2.1.0.

### Computer 4

environment_4.yml

Run `conda activate eeg4` to active it.

Uses tensorflow-gpu==2.0.0.

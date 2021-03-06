# GN3
This network recreates any single missing electrode:

<p align="center">
<img src="https://github.com/Svanteberg/Virtual-EEG-electrodes/blob/master/images/movie_gn3.gif" width="50%">
</p>

## Simple version
The file `gn3.py` contain a simple version for training GN3. It will train for 1,000 epochs, printing the MAE for 2,100 (21 * 100) examples from the training and validation dataset at each epoch. When training is done, it will generate EEG from 5040 examples (21 * 240, repeated consecutive electrode order, new EEG for every example) from the test dataset and save:

- MAE of training data for all epochs
- MAE of validation data for all epochs
- Overall MAE for the 5,040 examples of test data
- The 5,040 generated and original examples
- The resulting network model

All will be saved in a folder that is created automatically and named according to the date and time when starting the script: date_time-gn3. The folder will be located in a folder named 'results', that is also created if it is not already existing. 

## GUI version

A tkinter based GUI version is available, `gn3_gui.py`, where training progression, MAE of individual electrodes, and EEG examples are plotted intermittently.

<p align="center">
<img src="https://github.com/Svanteberg/Virtual-EEG-electrodes/blob/master/images/gn3_gui_screenshot.png" width="100%">
</p>

In the EEG examples, the recreated signals are in black with the corresponding original signals in blue, and the input signals are in red.

## Visualizing data
A simple EEG viewer is provided in the ``plot_eeg_gn3.py`` file for visualizing the saved EEG examples.

<p align="center">
<img src="https://github.com/Svanteberg/Virtual-EEG-electrodes/blob/master/images/eeg_gui_gn3_results_screenshot.png" width="100%">
</p>

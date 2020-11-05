# GN1
This network upsamples from 4 to 21 electrodes (17 recreated as output from the network):

<p align="center">
<img src="https://github.com/Svanteberg/Virtual-EEG-electrodes/blob/master/images/10-20_4-17.png" width="50%">
</p>

## Simple version
The file `gn1.py` contain a simple version for training GN1. It will train for 1,000 epochs, printing the MAE for 1,000 examples from the training and validation dataset at each epoch. When training is done, it will generate EEG from 5000 examples from the test dataset and save:

- MAE of training data for all epochs
- MAE of validation data for all epochs
- Overall MAE for the 5,000 examples of test data
- The 5,000 generated and original examples
- The resulting network model

## GUI version

A tkinter based GUI version is available, `gn1_gui.py`, where training progression, MAE of individual electrodes, and EEG examples are plotted intermittently.

<p align="center">
<img src="https://github.com/Svanteberg/Virtual-EEG-electrodes/blob/master/images/gn1_gui_screenshoot.png" width="100%">
</p>

In the EEG examples, the recreated signals are in black with the corresponding original signals in blue, and the input signals are in red.

## Generating data

The script `gn1_generate_from_model.py` can be used to generate data from a trained model. The script have to be modified to have the right path for the model.

<p align="center">
<img src="https://github.com/Svanteberg/Virtual-EEG-electrodes/blob/master/images/change_path.png" width="100%">
</p>

The results will be saved in the directory named 'results' as `gn1_new_eeg_from_model.npy`.

The script `gn1_generate_from_pretrained.py` can be used to generate data from a pre-trained model. The script loads weights into the model. The file containing the weights `gn1_weights.h5`have to be downloaded and put in a directory named 'models'. The results will be saved in the directory named 'results' as `gn1_new_eeg_from_pretrained.npy`.

## Visualizing data
A simple EEG viewer is provided in the ``plot_eeg_gn1.py`` file for visualizing the saved EEG examples.

<p align="center">
<img src="https://github.com/Svanteberg/Virtual-EEG-electrodes/blob/master/images/eeg_gui_results_screenshot.png" width="100%">
</p>

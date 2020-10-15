This network upsamples from 4 to 21 electrodes (17 recreated as output from the network):

<p align="center">
<img src="https://github.com/Svanteberg/Virtual-EEG-electrodes/blob/master/images/10-20_4-17.png" width="50%">
</p>


The file `gn1.py` contain a simple version for training GN1. It will train for 1,000 epochs, printing the MAE for 1,000 examples from the training and validation dataset at each epoch. When training is done, it will generate EEG from 5000 examples from the test dataset and save:

- MAE of training data for all epochs
- MAE of validation data for all epochs
- Overall MAE for the 5,000 examples of test data
- The 5,000 generated and original examples
- The resulting network model

A GUI version is available, `gn1_gui.py`, where training progression, MAE of individual electrodes, and EEG examples are plotted intermittently.

<p align="center">
<img src="https://github.com/Svanteberg/Virtual-EEG-electrodes/blob/master/images/gn1_gui_screenshoot.png" width="100%">
</p>

A simple EEG viewer is provided in the ``plot_eeg.py`` file for visualizing the saved EEG examples.

This network upsamples from 4 to 21 electrodes (17 recreated as output from the network):

<p align="center">
<img src="https://github.com/Svanteberg/Virtual-EEG-electrodes/blob/master/images/10-20_4-17.png" width="50%">
</p>


The file `GN1.py` contain a simple version for training GN1. It will train for 1000 epochs, printing the MAE for training and validation data at each epoch. When training is done, it will generate EEG from 5000 examples from the test dataset and save: MAE of training and validation data for all epochs, the overall MAE for the 5000 examples of test data, the 5000 generated and original examples, and the resulting network model.

A GUI version, where results and EEG examples are plotted intermittently, will also be available.

A simple EEG viewer is provided in the ``plot_eeg.py`` file for plotting the saved EEG examples.

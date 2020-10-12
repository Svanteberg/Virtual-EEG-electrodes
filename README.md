# Virtual EEG-electrodes (under construction)

## Introduction

In this study, deep neural networks based on convolutional layers were used to process EEG data. Two networks were trained to upsample the data and one network was trained to recreate single channels.

Electrodes are placed according to the international 10-20 system (Jasper, Electroencephalogr. Clin. Neurophysiol. 1958, 10:367-380). Here, a full EEG montage is defined  as consisting of 21 electrodes positioned (skalp positions approximated by a two-dimensional grid):

<p align="center">
<img src="https://github.com/Svanteberg/Virtual-EEG-electrodes/blob/master/images/10-20.png" width="40%">
</p>

### The networks

The *networks* were of *generative* character and are here referred to as GN1, GN2, and GN3.

#### GN1

This network upsampled from 4 to 21 electrodes (17 recreated as output from the network). Here, the electrode density is very low, the distances are large and the problem is hence ill posed.

<p align="center">
<img src="https://github.com/Svanteberg/Virtual-EEG-electrodes/blob/master/images/10-20_4-17.png" width="50%">
</p>

#### GN2

This network upsampled from 14 to 21 electrodes (7 recreated as output from the network). For this case, the electrode density is higher, the electrodes have an even distribution and the recreated values lie within a field of known values (in reality, the density decreases in radial direction due to the spherical geometry). The conditions for finding a solution for the problem is thus more favorable.

<p align="center">
<img src="https://github.com/Svanteberg/Virtual-EEG-electrodes/blob/master/images/10-20_14-7.png" width="50%">
</p>

#### GN3

This network recreates the value of any one blocked channel. The signal of the blocked channel is replaced by low amplitude white noise. In addition to recreating the signal, the network therefore also has to learn to detect which channel is missing.

<p align="center">
<img src="https://github.com/Svanteberg/Virtual-EEG-electrodes/blob/master/images/movie_gn3.gif" width="50%">
</p>

An example of the training progression for GN1 of the first 0 to 200 examples is given below. The original signal is in red and the recreated is in blue.

<img src="https://github.com/Svanteberg/Virtual-EEG-electrodes/blob/master/images/movie.gif" width="110%">

## Data

The EEG data from the published data base created at the Temple University Hospital (TUH), Philadelphia (Obeid & Picone, Frontiers of neuroscience 2016, 10:1-5) was used for this study. The TUH EEG Corpus (v1.1.0) with average reference was used (downloaded during 17-21 January 2019).

The Python library ‘pyEDFlib’ (Nahrstaedt & Lee-Messer, https://github.com/holgern/pyedflib) was used to extract EEG data. A total of 11,163 recordings (roughly 5,144 hours, from 1,385 subjects) with duration > 300 seconds and sampled at 256 Hz was extracted from the data set. The data was bandpass filtered between 0.3 Hz and 40 Hz using second-degree Butterworth filters. A 60 Hz notch filter was used to remove residual AC-noise. Filtering was applied with zero phase shift.

### Data organisation

The data was organized with each subject having a folder containing one or more of their respective EEG recordings. All EEGs in each folder were divided into numpy files of 10 s epochs and numbered in consecutive order. 

<p align="center">
<img src="https://github.com/Svanteberg/Virtual-EEG-electrodes/blob/master/images/data_architecture.png" width="75%">
</p>

Two lists mapping the numpy files to the subjects and EEG recordings were created. A subject list:

```
    [subject id 0, subject id 2, ..., sibject id n]
```

An index list for the numpy files:

```
    (subject id 0 ->) [[[[start EEG 1,end EEG 1],[start EEG 2,end EEG 2],...,[start EEG p,end EEG p]],
    (subject id 1 ->) [[start EEG 1,end EEG 1],[start EEG 2,end EEG 2],...,[start EEG q,end EEG q]],
                        .
                        .
                        .
    (subject id n ->) [[start EEG 1,end EEG 1],[start EEG 2,end EEG 2],...,[start EEG r,end EEG r]]]]
```

e.g.

```
    indices = [[[0,121],[122,205]],
                [[0,93],[94,303],[304,511],[512,789]],
                ...,
                [[0,64],[65,247],[248,388],[389,601]]]
```

so that each row in the index list corresponds to a subject and the numbers in each brachet correspond to the start and end of an EEG recording.

The data was split in a 80, 10 and 10 percent distribution for training, validation and testing. The distribution was with regard to the number of subjects to keep the data sets disjoint.

## Network architecture

## Training schedule

For each epoch of training, the training order of the subjects were randomized

<p align="center">
<img src="https://github.com/Svanteberg/Virtual-EEG-electrodes/blob/master/images/subjects_rand_ord.png" width="100%">
</p>

and the network was trained with one example from each subject. For each subject and training epoch, the EEG recordings were given a random order.

<p align="center">
<img src="https://github.com/Svanteberg/Virtual-EEG-electrodes/blob/master/images/eeg_rand_ord.png" width="65%">
</p>

A start position in the first EEG was randomly chosen by first randomly choosing a numpy file and then a random start position within the file. This way of drawing examples resulted in 10 s intervals overlapping two 10 s numpy files. These two files were loaded and concatenated, the example could then be extracted.

<p align="center">
<img src="https://github.com/Svanteberg/Virtual-EEG-electrodes/blob/master/images/file_concat.png" width="35%">
</p>

If the amplitude was between -500 and 500 µV, the example was accepted and used for training. If not, a new starting position in the recording was randomly chosen and the new example was checked for amplitude. This was repeated up to 100 times. If all 100 examples of that recording were rejected, the same procedure was performed for the next recording, and so on. If all examples of all recordings of a subject were rejected, no training took place that epoch for that specific subject.


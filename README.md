# Virtual EEG-electrodes (under construction)

## Introduction

This is a supplement to https://biorxiv.org/cgi/content/short/2020.04.20.049916v1. In this study, deep neural networks based on convolutional layers were used to process EEG data. Two networks were trained to upsample the data and one network was trained to recreate single channels.

Electrodes are placed according to the international 10-20 system (Jasper, Electroencephalogr. Clin. Neurophysiol. 1958, 10:367-380). Here, a full EEG montage is defined  as consisting of 21 electrodes positioned (skalp positions approximated by a two-dimensional grid):

<p align="center">
<img src="https://github.com/Svanteberg/Virtual-EEG-electrodes/blob/master/images/10-20.png" width="50%">
</p>

### The networks



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

This network recreates the value of any one blocked channel. The signal of the blocked channel is replaced by low amplitude white noise. The network therefore also has to learn to detect which channel is missing, in addition to recreating the signal.

<p align="center">
<img src="https://github.com/Svanteberg/Virtual-EEG-electrodes/blob/master/images/movie_gn3.gif" width="50%">
</p>

An example of the training progression for GN1 of the first 0 to 200 examples is given below. The original signal is in red and the recreated is in blue.

<img src="https://github.com/Svanteberg/Virtual-EEG-electrodes/blob/master/images/movie_gn1.gif" width="110%">

## Data

The EEG data from the published data base created at the Temple University Hospital (TUH), Philadelphia (Obeid & Picone, Frontiers of neuroscience 2016, 10:1-5) was used for this study. The TUH EEG Corpus (v1.1.0) with average reference was used (downloaded during 17-21 January 2019).

The Python library ‘pyEDFlib’ (Nahrstaedt & Lee-Messer, https://github.com/holgern/pyedflib) was used to extract EEG data. A total of 11,163 recordings (roughly 5,144 hours, from 1,385 subjects) with duration > 300 seconds and sampled at 256 Hz was extracted from the data set. The data was bandpass filtered between 0.3 Hz and 40 Hz using second-degree Butterworth filters. A 60 Hz notch filter was used to remove residual AC-noise. Filtering was applied with zero phase shift.

### Data organisation

The data was organized with each subject having a folder containing one or more of their respective EEG recordings. All EEGs in each folder were divided into numpy files of 10 s epochs and numbered. 

<p align="center">
<img src="https://github.com/Svanteberg/Virtual-EEG-electrodes/blob/master/images/data_architecture.png" width="75%">
</p>

A list mapping the numpy files to the subjects and EEG recordings was created.


```
    [[subject 1 id,[[start EEG 1,end EEG 1],[start EEG 2,end EEG 2],...,[start EEG p,end EEG p]]],
    [subject 2 id,[[start EEG 1,end EEG 1],[start EEG 2,end EEG 2],...,[start EEG q,end EEG q]]],
    ...,
    [subject n id,[[start EEG 1,end EEG 1],[start EEG 2,end EEG 2],...,[start EEG r,end EEG r]]]]
```

e.g.

```
    [[0,[[0,121],[122,205]]],
    [1,[[0,93],[94,303],[304,511],[512,789]]],
    ...,
    [1385,[[0,64],[65,247],[248,388],[389,601]]]]
```

The data was split in a 80, 10 and 10 percent distribution for training, validation and testing. The distribution was with regard to the number of subjects to keep the data sets disjoint.

## Network architecture

### Imports

```
from tensorflow.keras.layers import LeakyReLU, Layer, Input, Conv2D, Activation, add, BatchNormalization, Conv2DTranspose, Reshape, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import mean
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow as tf

import numpy as np
import random
```

### Temporal encoder block

```
    def conv(k,n,x):
        for i in range(n):
            x = Conv2D(filters=16*k*2**i,kernel_size=(1,3),strides=(1,2),padding='same')(x)
            x = LeakyReLU(alpha=0.2)(x)
        return x
```
### Spatial block

```
        x = Conv2D(1024,kernel_size=(4,1),strides=1,padding='valid')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2DTranspose(filters=256,kernel_size=(17,1),strides=1,padding='valid')(x)
        x = LeakyReLU(alpha=0.2)(x)
```

### Temporal decoder block

```
    def deconv(k,n,x):
        for i in range(n):
            x = Conv2DTranspose(filters=16*k*2**(n-i),kernel_size=(1,3),strides=(1,2),padding='same')(x)
            if i != n-1:
                x = LeakyReLU(alpha=0.2)(x)
        return x
```

### Generator network

```
    def analyzer_model():
        input_eeg = Input(shape=(4,2560,1))

        x = self.conv(1,4,input_eeg)

        x = Conv2D(1024,kernel_size=(4,1),strides=1,padding='valid')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2DTranspose(filters=256,kernel_size=(17,1),strides=1,padding='valid')(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = self.deconv(1,4,x)
        x = Conv2D(1,kernel_size=(1,1),strides=1)(x)
```

The network was initialized as:

```
    generator = analyzer_model()
    generator.compile(loss='mae',optimizer=Adam(1e-4, 0.5, 0.999))
```

## Training schedule

For each epoch of training, the training order of the subjects were randomized

<p align="center">
<img src="https://github.com/Svanteberg/Virtual-EEG-electrodes/blob/master/images/subjects_rand_ord.png" width="50%">
</p>

and the network was trained with one example from each subject. For each subject and training epoch, the EEG recordings were given a random order. A start position in the first EEG were randomly chosen. This way of drawing examples resulted in 10 s intervals overlapping two 10 s numpy files. These two files were loaded and concatenated, the example could then be extracted. If the amplitude was between -500 and 500 µV, the example was accepted and used for training. If not, a new starting position in the recording was randomly chosen and the new example was checked for amplitude. This was repeated up to 100 times. If all 100 examples of that recording was rejected, the same procedure was performed for the next recording, and so on. If all examples of all recordings of a subject were rejected, no training took place that epoch for that specific subject.

```
    def train(self):
        for epoch in range(0,epochs):
            index = random.sample(range(len(train_subjects)),len(train_subjects))
            for loop_index in range(0,len(train_subjects)):
                # train generator
                real_eeg,norm = self.generate_eeg(True,index[loop_index])
                if norm != 0: 
                    generator.train_on_batch(x=real_eeg[:,input_A,:,:],y=real_eeg[:,output_B,:,:])

```

```
    def generate_eeg(self,num):
        # initialize out data
        eeg = np.zeros((21,2560))
        # choose recording from subject (if multiple)
        recording = random.sample(range(len(indices)),len(indices))
        # choose epoch of recording, skip first and last couple of epochs to avoid artefacts
        epoch = random.randint(indices[recording[0]][0]+4,indices[recording[0]][1]-4)
        # load two epochs and concatenate
        epoch_1 = np.load('data/256/'+subject+'/eeg_'+str(epoch)+'.npy')
        epoch_2 = np.load('data/256/'+subject+'/eeg_'+str(epoch+1)+'.npy')
        data = np.concatenate((epoch_1,epoch_2),axis=1)
        # check i amplitude > threshold, if so draw new data
        try_count = 0
        recording_count = 0
        while np.max(np.abs(data)) > self.threshold and recording_count < len(recording)-1:
            epoch = random.randint(indices[recording[recording_count]][0]+4,indices[recording[recording_count]][1]-4)
            epoch_1 = np.load('data/256/'+subject+'/eeg_'+str(epoch)+'.npy')
            epoch_2 = np.load('data/256/'+subject+'/eeg_'+str(epoch+1)+'.npy')
            data = np.concatenate((epoch_1,epoch_2),axis=1)
            try_count += 1
            if try_count > 100:
                recording_count += 1
                try_count = 0
        if np.max(np.abs(data)) > self.threshold:
            eeg = 'Amplitude error'
            norm = 0
        else:
            # choose starting position in the two epochs to create one epoch
            position = random.randint(0,2559)
            eeg = data[:,position:(position+2560)]
            eeg = eeg[np.newaxis,:,:,np.newaxis]
            eeg,norm = normalize_std(eeg)
        return eeg,norm
```

```
    def normalize_std(self,data):
        # Normalizes the amplitude to a std of 1 per example
        norm_data = np.zeros(np.shape(data))
        norm = np.zeros(np.shape(data)[0])
        for i in np.arange(np.shape(data)[0]):
            norm[i] = np.std(np.reshape(data[i],(np.shape(data)[1]*np.shape(data)[2],1)))
            norm_data[i] = data[i]/norm[i]
        return norm_data,norm


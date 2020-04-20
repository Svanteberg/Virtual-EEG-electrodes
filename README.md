# Virtual EEG-electrodes

## Introduction

## Data

The EEG data from the published data base created at the Temple University Hospital (TUH), Philadelphia (Obeid & Picone, Frontiers of neuroscience 2016, 10:1-5) was used for this study. The TUH EEG Corpus (v1.1.0) with average reference was used (downloaded during 17-21 January 2019).

Electrodes are placed according to the international 10-20 system (Jasper, Electroencephalogr. Clin. Neurophysiol. 1958, 10:367-380).

<p align="center">
<img src="https://github.com/Svanteberg/Virtual-EEG-electrodes/blob/master/images/10-20.png" width="50%">
</p>

A total of 1,385 (♂/♀: 751/634) subjects with 11,163 (♂/♀: 5,648/5,515) recordings, corresponding to 5,144 hours, was extracted from the data set based on the criterion described below. Ages varied from 18 to 95 years with a mean age of 51±18 years. All original recordings were in European Data Format (EDF), unfiltered and sampled at 256 Hz. An arbitrary lower limit of recording length of 300 seconds was used to ensure that each recording would offer some variance during training. 

All data was high pass filtered at 0.3 Hz and low pass filtered at 40 Hz using second-degree Butterworth filters; the bandwidth was chosen according to our clinical preference. A 60 Hz notch filter was used to remove residual AC-noise. Filtering was applied with zero phase shift.

## Network architecture

### Temporal encoder block

```
    def conv(self,k,n,x):
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
    def deconv(self,k,n,x):
        for i in range(n):
            x = Conv2DTranspose(filters=16*k*2**(n-i),kernel_size=(1,3),strides=(1,2),padding='same')(x)
            if i != n-1:
                x = LeakyReLU(alpha=0.2)(x)
        return x
```

### Generator network

```
    def analyzer_model(self):
        input_eeg = Input(shape=(4,2560,1))

        x = self.conv(1,4,input_eeg)

        x = Conv2D(1024,kernel_size=(4,1),strides=1,padding='valid')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2DTranspose(filters=256,kernel_size=(17,1),strides=1,padding='valid')(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = self.deconv(1,4,x)
        x = Conv2D(1,kernel_size=(1,1),strides=1)(x)
```


## Training schedule

```
    def generate_eeg(self,train_data,num):
        # train or validate
        if train_data:
            subject = self.train_subjects[num]
            indices = self.train_indices[num]
        else:
            subject = self.val_subjects[num]
            indices = self.val_indices[num]
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
            


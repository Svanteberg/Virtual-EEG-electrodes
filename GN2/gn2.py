# tensorflow imports
from tensorflow.keras.layers import LeakyReLU, Input, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# general imports
import numpy as np
import random
import sys
import pickle
import time
import os

# set random seeds

np.random.seed(seed = 12345)
random.seed(12345)

# defining object for network, training, and data handling

"""
The object creates and trains a convolutional neural network for upsampling EEG data (21 electrodes, 10 s duration, and 256 Hz sample rate).

Input electrodes: FP1, F7 , T5, Fp2, F8, T6, C3, O1, C4, O2, A1, A2, FZ, PZ
Output electrodes: T3, T4, F3, P3, F4, P4, CZ

The results are saved in a directory: results -> date + start time + 'gn2'; e.g., results/20201012-114948-gn2/.
If the directory does not exist, it will be created. The following will be saved:

- training MAE for each epoch based on 1000 examples (mae_train.npy)
- valdiation MAE for each epoch based on 1000 examples (mae_val.npy)
- test MAE based on 5000 examples (mae_test.npy)
- the 5000 examples used for testing (test_eeg.npy)
- model (gn1_model.h5)

Note that all three datasets are used, so if you would like to test different architectures/hyperparameters before using the test dataset, 
set self.use_test_set = False

------------------------------ DESCRIPTION OF EEG DATA FILE ORGANIZATION ------------------------------

*** The script is dependent on the data being organized in the right way. Since this probably will be the main hurdle to get the script working, 
it may be a better alternative to create your own organization and modify the script accordingly. ***

The order of the electrodes must be: FP1, F7 ,T3, T5, Fp2, F8, T4, T6, F3, C3, P3, O1, F4, C4, P4, O2, A1, A2, FZ, CZ, PZ.

Each example have the size (21,2560).

Each subject should have their own directory (subject_id below) containing EEG data. Here organized as:

    data -> 256 -> subject_id -> EEG data (numpy data files)

At the initialization, two files are loaded that map the data (and these will have to be created for your data):

    'data/256/subject_folders' is loaded into self.subjects; this contain all the subject id:s. 
    'data/256/indices.npy' is loaded into self.file_indices; this contain info for the EEG recordings of each subject

(The subjects are then randomized by the script to training, validation, and testing according to a distribution 0.8, 0.1, and 0.1.)

All EEG recordings of a subject should be divided into 10 s sections and saved in the subject's directory with a consecutive number:

    'eeg_0.npy', 'eeg_1.npy', ..., 'eeg_n.npy'

and the self.file_indices (= 'data/256/indices.npy') file should contain the information regarding the 10 s sections for each subject and recording, e.g.,

    self.file_indices[0][0] = [0,120]    -> recording 1 of subject 0 is in the files 'eeg_0.npy' to 'eeg_120.npy
    self.file_indices[0][1] = [121,242]  -> recording 2 of subject 0 is in the files 'eeg_121.npy' to 'eeg_242.npy
    .
    .
    .
    self.file_indices[99][0] = [0,89]    -> recording 1 of subject 99 is in the files 'eeg_0.npy' to 'eeg_89.npy
    self.file_indices[99][1] = [90,311]  -> recording 2 of subject 99 is in the files 'eeg_90.npy' to 'eeg_311.npy

"""

class generator_2():

    def __init__(self):

        # input & output shapes
        self.shape_in = (14, 2560, 1)
        self.shape_out = (7, 2560, 1)
        # electrodes used for input & output
        self.input_electrodes = [0, 1, 3, 4, 5, 7, 9, 11, 13, 15, 16, 17, 18, 20]
        self.output_electrodes = [2, 6, 8, 10, 12, 14, 19]
        # network parameters
        self.layers = 4
        self.strides = 2
        # training parameters
        self.epochs = 1000
        self.use_test_set = True             # set to 'False' if not using test dataset
        # data generation parameters
        self.threshold = 500                # max absolute amplitude (µV) of data
        self.use_data_normalization = True  # if true, data is normalized to a standard deviation of one
        # load subject data
        with open('data/256/subject_folders', 'rb') as fp:
            self.subjects = pickle.load(fp)
        self.file_indices = np.load('data/256/indices.npy', allow_pickle=True)
        # distribute subjects into training, validation and test set
        self.subject_distribution = (0.8, 0.9, 1)
        self.train_subjects = []
        self.val_subjects = []
        self.test_subjects = []
        self.train_indices = []
        self.val_indices = []
        self.test_indices = []
        for i in range(len(self.subjects)):
            p = random.random()
            if p < self.subject_distribution[0]:
                self.train_subjects.append(self.subjects[i])
                self.train_indices.append(self.file_indices[i])
            elif p < self.subject_distribution[1]:
                self.val_subjects.append(self.subjects[i])
                self.val_indices.append(self.file_indices[i])
            else:
                self.test_subjects.append(self.subjects[i])
                self.test_indices.append(self.file_indices[i])
        # create generator network
        self.generator = self.generator_model()
        self.generator.compile(loss='mae',optimizer=Adam(1e-4, 0.5, 0.999))
        print(self.generator.summary())

#==================== generator network functions ===========================================

    # building blocks

    def conv(self,x):
        # convolutional block
        for i in range(self.layers):
            x = Conv2D(filters = 32*2**i, kernel_size = (1, 3), strides = (1, self.strides), padding = 'same')(x)
            x = LeakyReLU(alpha = 0.2)(x)
        return x

    def deconv(self,x):
        # deconvolutional block
        for i in range(self.layers):
            x = Conv2DTranspose(filters = 32*2**(self.layers-i-1), kernel_size = (1, 3), strides = (1, self.strides), padding = 'same')(x)
            if i != self.layers-1:
                x = LeakyReLU(alpha = 0.2)(x)
        return x

    # model

    def generator_model(self):
        input_eeg = Input(shape = self.shape_in)
        # temporal encoder
        x = self.conv(input_eeg)
        # spatial analysis
        x = Conv2D(1024, kernel_size = (self.shape_in[0], 1), strides = 1, padding = 'valid')(x)
        x = LeakyReLU(alpha = 0.2)(x)
        x = Conv2DTranspose(filters = 256, kernel_size = (self.shape_out[0], 1), strides = 1, padding = 'valid')(x)
        x = LeakyReLU(alpha = 0.2)(x)
        # temporal decoder
        x = self.deconv(x)
        # merging all filters
        x = Conv2D(1,kernel_size = (1, 1), strides = 1)(x)
        return Model(inputs = input_eeg, outputs = x, name = 'generator')

#==================================training loop function=============================================

    def train(self):
        self.directory = os.path.join('results',time.strftime('%Y%m%d-%H%M%S', time.localtime()) + '-' + sys.argv[0][:-3])
        # create directory for saving results
        if not os.path.exists(self.directory):
            print(self.directory)
            os.makedirs(self.directory)
        self.mae_train = np.zeros(self.epochs)
        self.mae_val = np.zeros(self.epochs)
        print('************************ Training started ***************************')
        for epoch in range(0, self.epochs):
            # generate random order of subjects
            index = random.sample(range(len(self.train_subjects)), len(self.train_subjects))
            # training loop
            for loop_index in range(0, len(self.train_subjects)):
                # train generator
                real_eeg, norm = self.generate_eeg(0, index[loop_index])
                if norm != 0: 
                    self.generator.train_on_batch(x = real_eeg[:, self.input_electrodes, :, :],
                                                    y = real_eeg[:, self.output_electrodes, :, :])
                if loop_index == len(self.train_subjects) - 1:
                    self.mae_train[epoch], self.mae_val[epoch] = self.training_test()
                    print('--- Epoch:',epoch+1,'--- Training MAE:',self.mae_train[epoch],'µV --- Validation MAE:',self.mae_val[epoch],'µV ---')
        # (test and) save
        if self.use_test_set: # perform evaluation with test data
            mae_test, test_eeg = self.final_test()
            print('-------------------- Final testing MAE:',mae_test,'µV -------------------------')
            print('*********************** Saving results ****************************')
            # save model
            self.generator.save(self.directory + '/gn2_model.h5')
            # save training/validation/test losses
            np.save(self.directory + '/mae_train.npy',self.mae_train)
            np.save(self.directory + '/mae_val.npy',self.mae_val)
            np.save(self.directory + '/mae_test.npy',mae_test)
            # save eegs
            np.save(self.directory + '/test_eeg.npy',test_eeg)
        else: # skip evaluation with test data
            print('*********************** Saving results ****************************')
            # save model
            self.generator.save(self.directory + '/gn2_model.h5')
            # save training/validation/test losses
            np.save(self.directory + '/mae_train.npy',self.mae_train)
            np.save(self.directory + '/mae_val.npy',self.mae_val)

#======================================test functions=========================================

    def mae(self, y_true, y_pred):
        data_1 = np.reshape(y_true, (np.shape(y_true)[0]*np.shape(y_true)[1], 1))
        data_2 = np.reshape(y_pred, (np.shape(y_pred)[0]*np.shape(y_pred)[1], 1))
        loss = (np.mean(np.abs(data_1 - data_2)))
        return loss

    def training_test(self, num = 1000):
        # initiate variabels for results
        mae_train = []
        mae_test = []
        # loop for generating and calculating MAE for 'num' examples
        for _ in np.arange(num):
            # load train data
            train_subject = random.sample(range(len(self.train_subjects)), 1)[0]
            eeg_train,norm_train = self.generate_eeg(0,train_subject)
            while norm_train == 0:
                train_subject = random.sample(range(len(self.train_subjects)), 1)[0]
                eeg_train,norm_train = self.generate_eeg(0, train_subject)
            # load test data
            test_subject = random.sample(range(len(self.val_subjects)), 1)[0]
            eeg_test,norm_test = self.generate_eeg(1, test_subject)
            while norm_test == 0:
                test_subject = random.sample(range(len(self.val_subjects)), 1)[0]
                eeg_test,norm_test = self.generate_eeg(1, test_subject)
            # generate artificial EEG:s
            eeg_synthetic_train = self.generator.predict(eeg_train[:, self.input_electrodes, :, :])
            eeg_synthetic_test = self.generator.predict(eeg_test[:, self.input_electrodes, :, :])
            # calculate MAE for training and test examples
            mae_train.append(self.mae(np.reshape(eeg_train[:, self.output_electrodes, :, :],(7, 2560))*norm_train, np.reshape(eeg_synthetic_train, (7, 2560))*norm_train))
            mae_test.append(self.mae(np.reshape(eeg_test[:, self.output_electrodes, :, :],(7, 2560))*norm_test ,np.reshape(eeg_synthetic_test, (7, 2560))*norm_test))
        # calculate total means
        mae_train = np.round(np.mean(mae_train), 1)
        mae_test = np.round(np.mean(mae_test), 1)
        return mae_train,mae_test

    def final_test(self, num = 5000):
        # initiate variabels for results
        mae_test = []
        eeg_data = []
        # loop for generating and calculating MAE for 'num' examples
        for _ in np.arange(num):
            # load test data
            test_subject = random.sample(range(len(self.test_subjects)), 1)[0]
            eeg_test,norm_test = self.generate_eeg(2, test_subject)
            while norm_test == 0:
                test_subject = random.sample(range(len(self.test_subjects)), 1)[0]
                eeg_test,norm_test = self.generate_eeg(2, test_subject)
            # generate artificial EEG:s
            eeg_synthetic_test = self.generator.predict(eeg_test[:, self.input_electrodes, :, :])
            # calculate MAE for training and test examples
            mae_test.append(self.mae(np.reshape(eeg_test[:, self.output_electrodes, :, :],(7, 2560))*norm_test ,np.reshape(eeg_synthetic_test, (7, 2560))*norm_test))
            # store eeg:s
            eeg_data.append([np.reshape(eeg_synthetic_test, (7, 2560))*norm_test, np.reshape(eeg_test, (21, 2560))*norm_test])
        # calculate total means
        mae_test = np.round(np.mean(mae_test), 1)
        return mae_test, eeg_data

#================================data handling==========================================

    def generate_eeg(self,dataset, subject_num):
        # train, validate, or test?
        if dataset == 0:
            subject = self.train_subjects[subject_num]
            indices = self.train_indices[subject_num]
        elif dataset == 1:
            subject = self.val_subjects[subject_num]
            indices = self.val_indices[subject_num]
        elif dataset == 2:
            subject = self.test_subjects[subject_num]
            indices = self.test_indices[subject_num]
        # initialize out data
        eeg = np.zeros((21,2560))
        # randomize recording order (if multiple recordings)
        recording = random.sample(range(len(indices)), len(indices))
        # choose epoch of recording, skip first and last couple of epochs to avoid artefacts
        epoch = random.randint(indices[recording[0]][0] + 4, indices[recording[0]][1] - 4)
        # load two consecutive epochs and concatenate
        epoch_1 = np.load('data/256/' + subject + '/eeg_' + str(epoch) + '.npy', allow_pickle = True)
        epoch_2 = np.load('data/256/' + subject + '/eeg_' + str(epoch+1) + '.npy', allow_pickle = True)
        data = np.concatenate((epoch_1, epoch_2), axis = 1)
        # check i amplitude > threshold, if so draw new data
        try_count = 0           # counts how many times examples have been drawn from a recording
        recording_count = 0     # tracks the randomized order of the recordings
        while np.max(np.abs(data)) > self.threshold and recording_count < len(recording) - 1:
            epoch = random.randint(indices[recording[recording_count]][0] + 4, indices[recording[recording_count]][1] - 4)
            epoch_1 = np.load('data/256/' + subject + '/eeg_' + str(epoch) + '.npy', allow_pickle = True)
            epoch_2 = np.load('data/256/' + subject + '/eeg_' + str(epoch+1) + '.npy', allow_pickle = True)
            data = np.concatenate((epoch_1, epoch_2), axis = 1)
            try_count += 1
            if try_count > 100:
                # if the max number of tries for a recording is reach, the search is continued in the next recording
                recording_count += 1
                try_count = 0
        if np.max(np.abs(data)) > self.threshold:
            # no examples with acceptable amplitude was created
            eeg = 'Amplitude error'
            norm = 0
        else:
            # choose random starting position in the first epoch and extract 10 s of data
            position = random.randint(0, 2559)
            eeg = data[:, position:(position + 2560)]
            eeg = eeg[np.newaxis, :, :, np.newaxis]
            if self.use_data_normalization:
                eeg,norm = self.normalize_std(eeg)
            else:
                norm = 1
        return eeg, norm

    def normalize_std(self,data):
        # normalizes the amplitude to a standard deviation of one
        norm = np.std(data)
        return data/norm, norm

# create object and train

GN1 = generator_2()
GN1.train()

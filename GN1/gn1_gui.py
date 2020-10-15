#==================Imports=============================================
# gpu imports
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from tensorflow.keras.layers import LeakyReLU, Layer, Input, Conv2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

import numpy as np
import random
import pickle
import datetime
import time
import sys

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import *

class generator_1():
    def __init__(self):

        # input & output shapes
        self.shape_in = (4, 2560, 1)
        self.shape_out = (17, 2560, 1)
        # electrodes used for input & output
        self.input_electrodes = [8, 10, 12, 14]
        self.output_electrodes = [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15, 16, 17, 18, 19, 20]
        self.labels = ['Fp1', 'F7', 'T3', 'T5', 'Fp2', 'F8', 'T4', 'T6', 'F3', 'C3', 'P3', 'O1', 'F4', 'C4', 'P4', 'O2', 'A1', 'A2', 'Fz', 'Cz', 'Pz']
        self.row_map = [0,1,2,3,0,1,2,3,2,4,2,4,2,2,1,2,3]
        self.col_map = [2,1,1,1,4,5,5,5,2,2,4,4,0,6,3,3,3]
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
        self.file_indices = np.load('data/256/indices.npy', allow_pickle = True)
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
        print('===',len(self.train_subjects),len(self.val_subjects),len(self.test_subjects),'===')

#==================== generator network functions ===========================================

    # building blocks

    def conv(self, x):
        # convolutional block
        for i in range(self.layers):
            x = Conv2D(filters = 32*2**i, kernel_size = (1, 3), strides = (1, self.strides), padding = 'same')(x)
            x = LeakyReLU(alpha = 0.2)(x)
        return x

    def deconv(self, x):
        # deconvolutional block
        for i in range(self.layers):
            x = Conv2DTranspose(filters = 32*2**(self.layers - i - 1), kernel_size = (1, 3), strides = (1, self.strides), padding = 'same')(x)
            if i != self.layers - 1:
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
        # create generator network
        self.generator = self.generator_model()
        self.generator.compile(loss='mae',optimizer=Adam(1e-4, 0.5, 0.999))
        print(self.generator.summary())
        # create directory for saving results
        self.directory = os.path.join('results',time.strftime('%Y%m%d-%H%M%S', time.localtime()) + '-' + sys.argv[0][:-3])
        if not os.path.exists(self.directory):
            print(self.directory)
            os.makedirs(self.directory)
        self.mae_train = []
        self.mae_val = []
        self.epochs = self.iterations_input.var.get()
        start_time = time.time()
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
                # update counter
                self.countVar.set(str(loop_index+1) + '/' + str(len(self.train_subjects)) + ' - ' + str(epoch + 1) + '/' + str(self.epochs))
                self.root.update_idletasks()
                # update ETA
                if loop_index % 10 == 0:
                    self.print_eta(start_time, epoch, loop_index)
                # plot EEG example
                if loop_index % 100 == 0:
                    # load validation example
                    test_subject = random.sample(range(len(self.val_subjects)), 1)[0]
                    eeg_test,norm = self.generate_eeg(1, test_subject)
                    while norm == 0:
                        test_subject = random.sample(range(len(self.val_subjects)), 1)[0]
                        eeg_test, norm = self.generate_eeg(1, test_subject)
                    # generate artificial EEG
                    eeg_pred = self.generator.predict(eeg_test[:, self.input_electrodes, :, :])
                    if self.use_data_normalization:
                        eeg_test = np.reshape(eeg_test, (21, 2560))*norm
                        eeg_pred = np.reshape(eeg_pred, (17, 2560))*norm
                    else:
                        eeg_test = np.reshape(eeg_test, (21, 2560))
                        eeg_pred = np.reshape(eeg_pred, (17, 2560))
                    self.plot_eeg(self.eeg_plot, eeg_pred, eeg_test)
                if loop_index == len(self.train_subjects) - 1:
                    self.plot_progression()
        # -------- (test and) save ---------
        if self.use_test_set: # perform evaluation with test data
            mae_test, test_eeg = self.final_test()
            print('-------------------- Final testing MAE:', mae_test, 'µV -------------------------')
            print('*********************** Saving results ****************************')
            # save model
            self.generator.save(self.directory + '/gn1_model.h5')
            # save training/validation/test losses
            np.save(self.directory + '/mae_train.npy', self.mae_train)
            np.save(self.directory + '/mae_val.npy', self.mae_val)
            np.save(self.directory + '/mae_test.npy', mae_test)
            # save eegs
            np.save(self.directory + '/test_eeg.npy', test_eeg)
        else: # skip evaluation with test data
            print('*********************** Saving results ****************************')
            # save model
            self.generator.save(self.directory + '/gn1_model.h5')
            # save training/validation/test losses
            np.save(self.directory + '/mae_train.npy', self.mae_train)
            np.save(self.directory + '/mae_val.npy', self.mae_val)

#======================================test functions=========================================

    def mae(self, y_true, y_pred):
        data_1 = np.reshape(y_true, (np.shape(y_true)[0]*np.shape(y_true)[1], 1))
        data_2 = np.reshape(y_pred, (np.shape(y_pred)[0]*np.shape(y_pred)[1], 1))
        loss = (np.mean(np.abs(data_1 - data_2)))
        return loss

    def mae_field(self,y_true,y_pred):
        field = np.zeros((5, 7))
        for i in range(self.shape_out[0]):
            field[self.row_map[i], self.col_map[i]] = np.mean(np.abs(y_true[i] - y_pred[i]))
        return field

    def training_test(self, num = 1000):
        # initiate variabels for results
        mae_train = []
        mae_val = []
        mae_fields = []
        # loop for generating and calculating MAE for 'num' examples
        for _ in np.arange(num):
            # load train data
            train_subject = random.sample(range(len(self.train_subjects)), 1)[0]
            eeg_train,norm_train = self.generate_eeg(0, train_subject)
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
            mae_train.append(self.mae(np.reshape(eeg_train[:, self.output_electrodes, :, :], (17, 2560))*norm_train, np.reshape(eeg_synthetic_train, (17, 2560))*norm_train))
            mae_val.append(self.mae(np.reshape(eeg_test[:, self.output_electrodes, :, :], (17, 2560))*norm_test ,np.reshape(eeg_synthetic_test, (17, 2560))*norm_test))
            # mae fields
            mae_field = self.mae_field(np.reshape(eeg_test[:, self.output_electrodes, :, :], (17, 2560))*norm_test, np.reshape(eeg_synthetic_test, (17, 2560))*norm_test)
            mae_fields.append(mae_field)
        # calculate total means
        mae_train = np.round(np.mean(mae_train), 1)
        mae_val = np.round(np.mean(mae_val), 1)
        mae_fields = np.mean(mae_fields, axis=0)
        return mae_train, mae_val, mae_fields

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
            mae_test.append(self.mae(np.reshape(eeg_test[:, self.output_electrodes, :, :], (17, 2560))*norm_test ,np.reshape(eeg_synthetic_test, (17, 2560))*norm_test))
            # store eeg:s
            eeg_data.append([np.reshape(eeg_synthetic_test, (17, 2560))*norm_test, np.reshape(eeg_test, (21, 2560))*norm_test])
        # calculate total means
        mae_test = np.round(np.mean(mae_test), 1)
        return mae_test, eeg_data

#================================data handling==========================================

    def generate_eeg(self, dataset, subject_num):
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
        eeg = np.zeros((21, 2560))
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
                # if the max number of tries for a recording is reached, the search is continued in the next recording
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
                eeg, norm = self.normalize_std(eeg)
            else:
                norm = 1
        return eeg, norm

    def normalize_std(self, data):
        # normalizes the amplitude to a standard deviation of one
        norm = np.std(data)
        return data/norm, norm

#=====================plot========================================

    def plot_eeg(self, frame ,eeg_pred, eeg_test):
        norm = 50/np.max(np.abs(eeg_test))
        frame.ax.cla()
        spacing = 50
        t = np.arange(np.shape(eeg_test)[1])
        norm = 50/np.max(np.abs(eeg_test))
        for idx in range(self.shape_out[0]):
            frame.ax.plot(t/256, norm*eeg_pred[idx] + spacing*self.output_electrodes[idx], color = 'black', linewidth = 0.5)
        for idx in range(21):
            if idx == 8 or idx == 10 or idx == 12 or idx == 14:
                frame.ax.plot(t/256, norm*eeg_test[idx] + spacing*idx, color = 'tab:red', linewidth = 0.5)
            else:
                frame.ax.plot(t/256, norm*eeg_test[idx] + spacing*idx, color = 'tab:blue', linewidth = 0.5)
        frame.ax.set_yticks(np.arange(0, spacing*21, spacing))
        frame.ax.set_yticklabels(self.labels)
        frame.ax.set_ylim(-spacing, 21*spacing)
        frame.ax.invert_yaxis()
        frame.canvas.draw()

    def plot_progression(self):
        # get validation values
        mae_train,mae_val,test_field = self.training_test()
        self.mae_train.append(mae_train)
        self.mae_val.append(mae_val)
        # plto training progression
        self.trainingPlot.ax.cla()
        self.trainingPlot.ax.plot(np.arange(len(self.mae_train)), self.mae_train, color = 'tab:blue', label = 'MAE training')
        self.trainingPlot.ax.plot(np.arange(len(self.mae_val)), self.mae_val, color = 'tab:orange', label = 'MAE validation')
        self.trainingPlot.ax.legend()
        self.trainingPlot.ax.set_xlabel('Epochs')
        self.trainingPlot.ax.set_title('Training progress')
        self.trainingPlot.ax.set_ylim(0, 15)
        self.trainingPlot.ax2.set_ylim(0, 15)
        self.trainingPlot.ax.set_ylabel('MAE (µV)')
        self.trainingPlot.ax.grid(b = True, which = 'major', linestyle = '-')
        self.trainingPlot.ax.grid(b = True, which = 'minor', linestyle = '--')
        self.trainingPlot.ax.minorticks_on()
        self.trainingPlot.ax.tick_params(right=False, labelright=False)
        self.trainingPlot.canvas.draw()
        # field
        self.plot_field(test_field)

    def plot_field(self,mat):
        thresh = 7.5
        # plot matrix
        self.field_plot.ax.cla()
        self.field_plot.ax.set_title('Channel MAE (µV)')
        self.field_plot.ax.imshow(mat, interpolation='nearest', vmin=0, vmax=15)
        self.field_plot.ax.tick_params(left = False, bottom = False, right = False, labelleft = False, labelbottom = False, labelright = False)
        for i in range(self.shape_out[0]):
            string = str(np.round(mat[self.row_map[i], self.col_map[i]], 1))
            self.field_plot.ax.text(self.col_map[i], self.row_map[i], string,
                                    horizontalalignment = "center",
                                    color = "white" if mat[self.row_map[i], self.col_map[i]] < thresh else "black")
        self.field_plot.canvas.draw()

#=====================misc========================================

    def print_eta(self,start_time,epoch,loop_index):
        passed_time = time.time() - start_time
        eta = round((passed_time/(epoch*len(self.train_subjects) + loop_index + 1))*(self.epochs*len(self.train_subjects) - (epoch*len(self.train_subjects) + loop_index + 1)))
        passed_time_string = str(datetime.timedelta(seconds = round(passed_time)))
        eta_string = str(datetime.timedelta(seconds = eta))
        self.etaVar.set(' Time passed ' + passed_time_string + '  ETA in ' + eta_string+' ')
        self.root.update_idletasks()

#=====================gui=========================================

class figureFrame(object):
    # creates a figure for plotting data in the GUI
    def __init__(self,frame,figure_size,side,hide_axes,color):

        self.fig = Figure(figsize = figure_size, dpi = 100, tight_layout = True, facecolor=color)
        self.ax = self.fig.add_subplot(111)
        if hide_axes:
            self.ax.tick_params(right = False, left = False, top = False, bottom = False, labelleft = False, labelbottom = False)
            self.ax.spines['top'].set_visible(False)
            self.ax.spines['bottom'].set_visible(False)
            self.ax.spines['right'].set_visible(False)
            self.ax.spines['left'].set_visible(False)
        self.canvas = FigureCanvasTkAgg(self.fig,frame)
        self.canvas.draw()
        self.canvas._tkcanvas.pack(side = side, fill = BOTH, padx = 5, pady = 5)

class labelWithOption(object):
    # creates an option menu with a label above
    def __init__(self, frame, text, nums, init_var, row, column, width, color):

        Label(frame, text = text, background = color).grid(row = row, column = column, padx = 5, pady = 5)
        self.option_nums = nums
        if isinstance(nums[0], int):
            self.var = IntVar()
        elif isinstance(nums[0], str):
            self.var = StringVar()
        else:
            self.var = DoubleVar()
        self.var.set(init_var)
        self.oM = OptionMenu(frame, self.var, *self.option_nums)
        self.oM.config(width = width, bg = color)
        self.oM.grid(row = row + 1, column = column, padx = 5, pady = 5)

G = generator_1()
# ***** Init GUI *****
color = '#dadada'
G.root = Tk()
G.root.configure(background = color)
G.root.title('Upsampling from 4 to 17 electrodes')
mainFrame = Frame(G.root, background = color)
mainFrame.pack(fill = BOTH)
topFrame = Frame(mainFrame, background = color)
topFrame.pack(fill = BOTH, padx = 5, pady = 5)
bottomFrame = Frame(mainFrame, background = color)
bottomFrame.pack(side = BOTTOM, padx = 5, pady = 5)

topFrame_L = Frame(topFrame, background = color)
topFrame_L.pack(side = LEFT, fill = BOTH, padx = 5, pady = 5)
topFrame_R = Frame(topFrame, background = color)
topFrame_R.pack(side = LEFT, fill = BOTH, padx = 5, pady = 5)

# ***** EEG examples *****
G.eeg_plot = figureFrame(topFrame_L, (10, 8), 'left', False, color)

# ***** Training progress *****
G.trainingPlot = figureFrame(topFrame_R, (7, 4), 'top', False, color)
G.trainingPlot.ax2 = G.trainingPlot.ax.twinx()

G.field_plot = figureFrame(topFrame_R, (7, 4), 'bottom', False, color)

# ***** Settings *****
W = 12 # width of widgets

Button(bottomFrame, text = ' Quit', width = W, command = G.root.quit, background = color, highlightbackground = color).grid(row = 2, column = 0, padx = 5, pady = 5)
G.iterations_input = labelWithOption(bottomFrame, 'Epochs', [10, 20, 50, 100, 500, 1000, 10000], 1000, 1, 1, W, color)
Button(bottomFrame, text = 'Train', width = W, command = G.train, background = color, highlightbackground = color).grid(row = 2, column = 2, padx = 5, pady = 5)
# iteration count
Label(bottomFrame,text = '                           ').grid(row = 0, column = 4)
#
G.countVar = StringVar()
G.countVar.set('-/-------')
G.count_label = Label(bottomFrame, textvariable = G.countVar, font = ('Courier', 20), relief = SUNKEN, foreground = '#00ff00', background = 'black', highlightbackground = color).grid(row = 1, column = 6, columnspan = 2, padx = 5, pady = 5)
# eta count
G.etaVar = StringVar()
G.etaVar.set('---------------------------------------------------------------')
Label(bottomFrame, textvariable = G.etaVar, font = ('Courier', 20), relief = SUNKEN, foreground = '#00ff00', background = 'black', highlightbackground = color).grid(row = 5, column = 5, columnspan = 3, padx = 5, pady = 5)

# ***** Mainloop *****
G.root.mainloop()
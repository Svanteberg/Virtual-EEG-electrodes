import numpy as np
import pickle

from tensorflow.keras.layers import LeakyReLU, Input, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model

"""
This script creates a pre-trained the GN1 network by loading the weights provided in the GitHub repository.

The weights in 'gn1_weights.h5' have to be downloaded into a folder named 'models' (if it does not exist you will have to create it).

The results will be saved as 'gn1_new_eeg_from_pretrained.npy' in a folder 'results' (if it does not exist you will have to create it).

"""

# =============== model functions ================

shape_in = (4, 2560, 1)
shape_out = (17, 2560, 1)
input_electrodes = [8, 10, 12, 14]
layers = 4
strides = 2

def conv(x):
    # convolutional block
    for i in range(layers):
        x = Conv2D(filters = 32*2**i, kernel_size = (1, 3), strides = (1, strides), padding = 'same')(x)
        x = LeakyReLU(alpha = 0.2)(x)
    return x

def deconv(x):
    # deconvolutional block
    for i in range(layers):
        x = Conv2DTranspose(filters = 32*2**(layers - i - 1), kernel_size = (1, 3), strides = (1, strides), padding = 'same')(x)
        if i != layers - 1:
            x = LeakyReLU(alpha = 0.2)(x)
    return x

def generator_model():
    input_eeg = Input(shape = shape_in)
    # temporal encoder
    x = conv(input_eeg)
    # spatial analysis
    x = Conv2D(1024, kernel_size = (shape_in[0], 1), strides = 1, padding = 'valid')(x)
    x = LeakyReLU(alpha = 0.2)(x)
    x = Conv2DTranspose(filters = 256, kernel_size = (shape_out[0], 1), strides = 1, padding = 'valid')(x)
    x = LeakyReLU(alpha = 0.2)(x)
    # temporal decoder
    x = deconv(x)
    # merging all filters
    x = Conv2D(1,kernel_size = (1, 1), strides = 1)(x)
    return Model(inputs = input_eeg, outputs = x, name = 'generator')

# ===============================================

def load_data():
    # load subject data
    with open('data/256/subject_folders', 'rb') as fp:
        subjects = pickle.load(fp)
    file_indices = np.load('data/256/indices.npy', allow_pickle=  True)
    # load numpy files
    data = []
    for i in range(len(subjects)):
        for j in range(len(file_indices[i])):
            for k in range(file_indices[i][j][0],file_indices[i][j][1]+1):
                data.append(np.load('data/256/' + subjects[i] + '/eeg_' + str(k) + '.npy', allow_pickle=True))
    return data

# create the model
model = generator_model()
# load weights into model
model.load_weights('models/gn1_weights.h5')
# load data
original_eeg = np.array(load_data())
# predict data
predicted_eeg = model.predict(np.reshape(original_eeg[:,input_electrodes,:],(np.shape(original_eeg)[0],shape_in[0],shape_in[1],shape_in[2])))
# save data
save_data = []
for i in range(len(original_eeg)):
    save_data.append([predicted_eeg[i],original_eeg[i]])
np.save('results/gn1_new_eeg_from_pretrained.npy',save_data)
import numpy as np
import pickle


from tensorflow.keras.models import Model

"""
This script load a trained model of the GN1 type and generate.

The path to the model will have to change at the indicated position in the script ('<===== CHANGE THE PATH TO USE YOUR MODEL')

The results will be saved as 'gn1_new_eeg_from_model.npy' in a folder 'results' (if it does not exist you will have to create it).

"""

shape_in = (4, 2560, 1)
input_electrodes = [8, 10, 12, 14]
layers = 4
strides = 2

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

# load model
model = Model('models/gn1_model.h5') # <===== CHANGE THE PATH TO USE YOUR MODEL
# load data
original_eeg = np.array(load_data())
# predict data
predicted_eeg = model.predict(np.reshape(original_eeg[:,input_electrodes,:],(np.shape(original_eeg)[0],shape_in[0],shape_in[1],shape_in[2])))
# save data
save_data = []
for i in range(len(original_eeg)):
    save_data.append([predicted_eeg[i],original_eeg[i]])
np.save('results/gn1_new_eeg_from_model.npy',save_data)
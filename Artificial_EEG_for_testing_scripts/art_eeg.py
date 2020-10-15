import numpy as np
import matplotlib.pyplot as pl
from scipy.stats import norm, skewnorm
import pickle
import os

# electrodes
leads = ['Fp1','F7','T3','T5','Fp2','F8','T4','T6','F3','C3','P3','O1','F4','C4','P4','O2','A1','A2','Fz','Cz','Pz']
row_map = [0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 2, 2, 1, 2, 3]
col_map = [2, 1, 1, 1, 4, 5, 5, 5, 2, 2, 2, 2, 4, 4, 4, 4, 0, 6, 3, 3, 3]

# duration of each recording

# ============== transformation functions ==============

def R(x, y, phi):
    '''
    coordinate rotation
    '''
    r = np.zeros(2)
    r[0] = x*np.cos(phi) - y*np.sin(phi)
    r[1] = x*np.sin(phi) - y*np.cos(phi)
    return r

def transform_field(matrix_field):
    '''
    transform electrodes from 2D to 1D
    '''
    array_field = matrix_field[row_map, col_map]
    return array_field

def field_map(size, idx, idy, angle = True):
    '''
    generates a 2D amplitude distribution
    '''
    A = np.random.normal(0, 0.01, ((2, 5, 7)))
    std = np.zeros(2)
    if idx < 0:
        idx = np.random.randint(7)
        idy = np.random.randint(5)
    std[0] = (size[1] - size[0])*np.random.rand() + size[0] + 0.00001
    std[1] = (size[1] - size[0])*np.random.rand() + size[0] + 0.00001
    if angle:
        phi = (np.pi/2)*np.random.rand()
    else:
        phi = 0
    for j in range(21):
        r = R(col_map[j] - idx, row_map[j] - idy, phi)
        for k in range(2):
            A[k,row_map[j], col_map[j]] = norm.pdf(r[k], 0, std[k])
    A[0] = A[0]/np.max(A[0])
    A[1] = A[1]/np.max(A[1])
    A = transform_field(A[0]*A[1])
    return A

# =========== artifact generation =============

def muscle(EEG):
    '''
    generates 1 - 5 muscle artifacts with probability 0.25
    '''
    if np.random.rand() < 0.25:
        for i in range(np.random.randint(1,5)):
            ch = np.random.randint(21)
            A = field_map((0, 2), row_map[ch],col_map[ch])
            d = np.random.randint(2560)
            start = np.random.randint(2560 - d)
            for j in range(21):
                EEG[j,start:start + d] += A[j]*np.random.normal(0, 10, d)
    return EEG

def blink(EEG):
    '''
    generates 1 - 5 blink artifacts with probability 0.75
    '''
    if np.random.rand() < 0.75:
        for i in range(np.random.randint(1,5)):
            d = np.random.randint(100,500)
            while len(np.arange(0,10,10/d)) != d:
                d = np.random.randint(100,500)
            amp = np.random.randint(50,200)
            skewness = np.random.randint(2,10)
            start = np.random.randint(2560 - d)
            blink = amp*skewnorm.pdf(np.arange(0,10,10/d), skewness, 2, 1)
            std = 2*np.random.rand()
            for ch in [0, 4]:
                A = field_map((std, std), col_map[ch], row_map[ch], False)
                for j in range(21):
                    EEG[j,start:start + d] += A[j]*blink
    return EEG

def add_artifacts(EEG):
    epochs = int(np.shape(EEG)[1]/2560)
    for i in range(epochs):
        EEG[:,2560*i:2560*(i + 1)] = muscle(EEG[:,2560*i:2560*(i + 1)])
        EEG[:,2560*i:2560*(i + 1)] = blink(EEG[:,2560*i:2560*(i + 1)])
    return EEG

# =========== EEG generation =============

def source(duration):
    '''
    generates an EEG signal. (adapted from: Bai et al., ”Nonlinear Markov process amplitude EEG model for nonlinear coupling interaction of spontaneous EEG,” 
    IEEE Transactions of biomedicine engineering, vol. 47, nr 9, pp. 1141-1146, 2000)
    '''
	# init variables and parameters
    num = np.random.randint(1, 4)
    x = np.zeros(duration)
    gamma = np.zeros(num)
    f = np.zeros(num)
    phi = np.zeros(num)
    std = np.zeros(num)
    for i in range(num):
        a = np.abs(np.random.normal(20, 10))
        gamma[i] = 0.029*np.random.rand() + 0.97
        f[i] = np.abs(np.random.normal(10, 5))
        phi[i] = 2*np.pi*np.random.rand()
        std[i] = 2.5*np.random.rand() + 2.5
	# generate signal
    for t in range(duration):
        for i in range(num):
            a = gamma[i]*a + np.random.normal(0, std[i])
            x[t] += a*np.sin(2*np.pi*f[i]*t/256 + phi[i])
    return x

def generate_eeg(duration):
    '''
    generates 21 electrode EEG; number of samples = 'duration'; 'num' sources
    '''
    num = np.random.randint(2, 15)
    EEG = np.random.normal(0, 0.1, (21, duration))
    for i in range(num):
        A = field_map((1, 5), -1, -1)
        S = source(duration)
        for j in range(21):
            EEG[j] += A[j]*S
    EEG = add_artifacts(EEG/num)
    #print(np.mean(np.abs(EEG)),np.min(EEG),np.max(EEG))
    return EEG

# =========== Data generation =============

def generate_data(num, duration):
    subjects = []
    file_indices = []
    directory = 'data/256'
    # create directory for saving data if not existing
    if not os.path.exists(directory):
        os.makedirs(directory)
    subject_id = 0
    for i in range(num):
        # create subject directory
        folder_name = ''
        for _ in range(8 - len(str(subject_id))): folder_name += '0'
        for i in range(len(str(subject_id))): folder_name += str(subject_id)[i]
        if not os.path.exists(directory + '/' + folder_name):
            os.makedirs(directory + '/' + folder_name)
        # add subject to subject list
        subjects.append(folder_name)
        # add empty list to index list to store indices
        indices = []
        last = 0
        # generate 1 - 3 EEGs
        for _ in range(np.random.randint(1, 4)):
            EEG = generate_eeg()
            epochs = int(np.shape(EEG)[1]/2560)
            for j in range(epochs):
                np.save(directory + '/' + folder_name+  '/eeg_' + str(last + j) + '.npy', EEG[:, 2560*j:2560*(j + 1)])
            indices.append([last, last + epochs - 1])
            last += epochs
        file_indices.append(indices)
        # new subject with random higher number
        subject_id += np.random.randint(1, 20)
    # save subject list & index list
    pickle.dump(subjects, open('data/256/subject_folders', 'wb') )
    np.save('data/256/indices.npy', file_indices)
    print(subjects)
    print(file_indices)

def plot_example():
    pl.figure(figsize=(18, 8))
    pl.rcParams.update({'font.size':16})
    F = generate_eeg(2560)
    for i in range(21):
        pl.plot(np.arange(2560)/256, F[i] + 50*i, color = 'black', linewidth = 0.5)
    pl.yticks(np.arange(0, 21*50, 50), leads)
    pl.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    pl.gca().invert_yaxis()
    pl.tight_layout()
    pl.show()

# set duration of EEGs and generate data
duration = 768000 # (5 mins per EEG)
num = 10 # number of subjects, 1 - 3 EEGs of length = 'duration' will be created
generate_data(num, duration)

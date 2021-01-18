# Virtual EEG-electrodes ***** under construction *****

## Introduction

In this study, deep neural networks based on convolutional layers were used to process EEG data. Two networks were trained to upsample the data and one network was trained to recreate single channels.

Electrodes are placed according to the international 10-20 system (Jasper, Electroencephalogr. Clin. Neurophysiol. 1958, 10:367-380). Here, a full EEG montage is defined  as consisting of 21 electrodes positioned (skalp positions approximated by a two-dimensional grid):

<p align="center">
<img src="https://github.com/Svanteberg/Virtual-EEG-electrodes/blob/master/images/10-20.png" width="40%">
</p>

### The networks

The *networks* were of *generative* character and are here referred to as [GN1](https://github.com/Svanteberg/Virtual-EEG-electrodes/tree/master/GN1), [GN2](https://github.com/Svanteberg/Virtual-EEG-electrodes/tree/master/GN2), and [GN3](https://github.com/Svanteberg/Virtual-EEG-electrodes/tree/master/GN3).

#### [GN1](https://github.com/Svanteberg/Virtual-EEG-electrodes/tree/master/GN1)

This network upsampled from 4 to 21 electrodes (17 recreated as output from the network). Here, the electrode density was very low, the distances were large and the problem was hence ill posed.

<p align="center">
<img src="https://github.com/Svanteberg/Virtual-EEG-electrodes/blob/master/images/10-20_4-17.png" width="50%">
</p>

#### [GN2](https://github.com/Svanteberg/Virtual-EEG-electrodes/tree/master/GN2)

This network upsampled from 14 to 21 electrodes (7 recreated as output from the network). For this case, the electrode density was higher, the electrodes had an even distribution and the recreated values lay within a field of known values (in reality, the density decreases in radial direction due to the spherical geometry). The conditions for finding a solution for the problem was thus more favorable.

<p align="center">
<img src="https://github.com/Svanteberg/Virtual-EEG-electrodes/blob/master/images/10-20_14-7.png" width="50%">
</p>

#### [GN3](https://github.com/Svanteberg/Virtual-EEG-electrodes/tree/master/GN3)

This network recreated the value of any one blocked channel. The signal of the blocked channel was replaced by low amplitude white noise. In addition to recreating the signal, the network therefore also had to learn to detect which channel was missing.

<p align="center">
<img src="https://github.com/Svanteberg/Virtual-EEG-electrodes/blob/master/images/movie_gn3.gif" width="50%">
</p>

An example of the training progression for GN1 of the first 0 to 200 examples is given below. The original signal is in red and the recreated is in blue.

<img src="https://github.com/Svanteberg/Virtual-EEG-electrodes/blob/master/images/movie.gif" width="110%">

---

## Installing software and running scripts

Blue font in the text indicate link to directory or script.

### Directories of the repo

#### GN1, GN2, and GN3 directories

##### Content

The directory [GN1](https://github.com/Svanteberg/Virtual-EEG-electrodes/tree/master/GN1) contains the files:

   [gn1.py](https://github.com/Svanteberg/Virtual-EEG-electrodes/tree/master/GN1/gn1.py)

   [gn1_generate_from_model.py](https://github.com/Svanteberg/Virtual-EEG-electrodes/tree/master/GN1/gn1_generate_from_model.py)

   [gn1_generate_from_pretrained.py](https://github.com/Svanteberg/Virtual-EEG-electrodes/tree/master/GN1/gn1_generate_from_pretrained.py)

   [gn1_gui.py](https://github.com/Svanteberg/Virtual-EEG-electrodes/tree/master/GN1/gn1_gui.py)

   [gn1_weights.h5](https://github.com/Svanteberg/Virtual-EEG-electrodes/tree/master/GN1/gn1_weights.h5)

   [plot_eeg_gn1.py](https://github.com/Svanteberg/Virtual-EEG-electrodes/tree/master/GN1/plot_eeg_gn1.py).

The directory [GN2](https://github.com/Svanteberg/Virtual-EEG-electrodes/tree/master/GN2) contains the files:

   [gn2.py](https://github.com/Svanteberg/Virtual-EEG-electrodes/tree/master/GN2/gn2.py)

   [gn2_gui.py](https://github.com/Svanteberg/Virtual-EEG-electrodes/tree/master/GN2/gn2_gui.py)

   [gn2_weights.h5](https://github.com/Svanteberg/Virtual-EEG-electrodes/tree/master/GN2/gn2_weights.h5)

   [plot_eeg_gn2.py](https://github.com/Svanteberg/Virtual-EEG-electrodes/tree/master/GN2/plot_eeg_gn2.py).

The directory [GN3](https://github.com/Svanteberg/Virtual-EEG-electrodes/tree/master/GN3) contains the files:

   [gn3.py](https://github.com/Svanteberg/Virtual-EEG-electrodes/tree/master/GN3/gn3.py)

   [gn3_gui.py](https://github.com/Svanteberg/Virtual-EEG-electrodes/tree/master/GN3/gn3_gui.py)

   [gn3_weights.h5](https://github.com/Svanteberg/Virtual-EEG-electrodes/tree/master/GN3/gn3_weights.h5)

   [plot_eeg_gn3.py](https://github.com/Svanteberg/Virtual-EEG-electrodes/tree/master/GN3/plot_eeg_gn3.py).

##### Description of files

The files `gn1.py`, `gn2.py`, and `gn3.py` are simpler versions for training the respective networks.

The files `gn1_gui.py`, `gn2_gui.py`, and `gn3_gui.py` are GUI versions that shows the training progression and intermittently shows EEG examples during training.

The files `gn1_weight.h5`, `gn2_weights.h5`, and `gn3_weight.h5` contain weights of networks trained on real EEG data.

The files `plot_eeg_gn1.py`, `plot_eeg_gn2.py`, and `plot_eeg_gn3.py` are simple GUIs for visualizing the resulting generated data and compare it to the original EEG. 

For the GN1 network, a script for generating data from a model that you have trained is provided in the file `gn1_generate_from_model.py` and the file `gn1_generate_from_pretrained.py` will create the model using the provided weights `gn1_weights.h5` and generate data.

#### Auxiliary directories

##### [Artificial_EEG_for_testing_scripts](https://github.com/Svanteberg/Virtual-EEG-electrodes/tree/master/Artificial_EEG_for_testing_scripts)

The directory contains a script [art_eeg.py](https://github.com/Svanteberg/Virtual-EEG-electrodes/tree/master/Artificial_EEG_for_testing_scripts/art_eeg.py) for generating artificial EEG data and a script [plot_original_eeg.py](https://github.com/Svanteberg/Virtual-EEG-electrodes/tree/master/Artificial_EEG_for_testing_scripts/plot_original_eeg.py) for visualizing the data. The data can be used as a substitute to real EEG data to train and test the networks, since it may take a larger effort to setup for using real data.

##### [Environments](https://github.com/Svanteberg/Virtual-EEG-electrodes/tree/master/Environments)

This directory contain four different yml-files of environments that can be tried to possibly make it easier to use the scripts.

---

### Preparation

#### Setting up Anaconda environment

The following instructions have been written for Linux. It is assumed that a computer with working installed (nVidia) graphics card and drivers are used.

To use / train the networks, first install Anaconda, https://www.anaconda.com/products/individual

```
    wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
    chmod +x Anaconda3-2020.07-Linux-x86_64.sh
    ./Anaconda3-2020.07-Linux-x86_64.sh
```


Then create an Anaconda environment from one of the available [environment files](https://github.com/Svanteberg/Virtual-EEG-electrodes/tree/master/Environments) (here examplified for [environment_1.yml](https://github.com/Svanteberg/Virtual-EEG-electrodes/blob/master/Environments/environment_1.yml); the file you choose of course have to downloaded to your computer)

```
    conda env create -f environment_1.yml
```

Activate the environment you just created

```
    conda activate eeg1
```

#### Data

Choose one of the options:

   1. Setup your EEG data according to the proposed organization (see *Data organization*, below).
   2. Download a small sample of available [artificial data](https://github.com/Svanteberg/Virtual-EEG-electrodes/tree/master/Artificial_EEG_for_testing_scripts/data).
   3. Create some artificial data using the available [script](https://github.com/Svanteberg/Virtual-EEG-electrodes/blob/master/Artificial_EEG_for_testing_scripts/art_eeg.py):

 ```
    python3.7 art_eeg.py
 ```

#### Setting up a structure for directories

The figure below illustrates how to setup the needed folders for working with the GN1 network.

<img src="https://github.com/Svanteberg/Virtual-EEG-electrodes/blob/master/images/directory_structure.png" width="100%">

This mean that root should contain folders named *data* and *models*, and all Python scripts (all files with the `.py` extension). The folder *models* should contain provided weights of pre-trained networks or models you have trained (all files with the `.h5` extension).

Downloading the provided small sample of artificial data or running the script for generating artificial data should create the correct structure for working with the networks, as long as the *data* folder is located as described above.

---

### Using scripts

Examples given here are for using/working with the GN1 network.

#### Train a network

To train a network of type GN1, run

```
    python3.7 gn1.py
```

or use the GUI version

```
    python3.7 gn1_gui.py
```

#### Use a trained network for generating new artificial data

To use a model, that you have trained, on new EEG data, find the path of the model you would like to use, and modify the script `gn1_generate_from_model.py` to use this path for loading the model. Or, you can put the file containing the model in the folder *models* and make sure it is named `gn1_model.h5`. Then run the script

```
    python3.7 gn1_generate_from_model.py
```

To instead use one of our pre-trained network of type GN1 on new EEG data, the file `gn1_weights.h5` will have to be downloaded into a folder named *models*. Then run

```
    python3.7 gn1_generate_from_pretrained.py
```

#### Visualizing data

To visualize the generated data, and to compare it to the original EEG data, run

 ```
    python3.7 plot_eeg_gn1.py
 ```

This will open a GUI. Use the 'Open directory' button to navigate in the *results* folder to find the EEG data you want to visualize.

<p align="center">
<img src="https://github.com/Svanteberg/Virtual-EEG-electrodes/blob/master/images/eeg_gui_results_screenshot.png" width="50%">
</p>

The available weights are from networks trained on real EEG data. It is possible to use a network with these weights applied to the artificial data but with mixed results. The network will mainly have problem with examples having random signals and/or signals with limited spatial distributions.

---

## Data

The EEG data from the published database created at the Temple University Hospital (TUH), Philadelphia (Obeid & Picone, Frontiers of neuroscience 2016, 10:1-5) was used for this study. The TUH EEG Corpus (v1.1.0) with average reference was used (downloaded during 17-21 January 2019).

The Python library ‘pyEDFlib’ (Nahrstaedt & Lee-Messer, https://github.com/holgern/pyedflib) was used to extract EEG data. A total of 11,163 recordings (roughly 5,144 hours, from 1,385 subjects) with duration > 300 seconds and sampled at 256 Hz was extracted from the data set. The data was bandpass filtered between 0.3 Hz and 40 Hz using second-degree Butterworth filters. A 60 Hz notch filter was used to remove residual AC-noise. Filtering was applied with zero phase shift.

### Data organization

*The developed scripts require that the data is organized in a specific way. This may well be the main challenge for anyone attempting to use the scripts, and it will likely save time to instead modify them to accommodate your own data structure. An example, containing artificial EEG data, is provided [here](https://github.com/Svanteberg/Virtual-EEG-electrodes/tree/master/Artificial_EEG_for_testing_scripts). A script for generating more data is also provided.*

The data was organized with each subject having a folder containing one or more of their respective EEG recordings. All EEGs in each folder were divided into numpy files of 10 s epochs and numbered in consecutive order. 

<p align="center">
<img src="https://github.com/Svanteberg/Virtual-EEG-electrodes/blob/master/images/data_architecture.png" width="75%">
</p>

Two lists mapping the numpy files to the subjects and EEG recordings were created. A subject list:

```
    [subject id 0, subject id 1, ..., subject id n]
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
    subject_list = ['00000000','00000032', ...,'00013453']
    
    index_list = [[[0,121],[122,205]],
                [[0,93],[94,303],[304,511],[512,789]],
                    .
                    .
                    .
                [[0,64],[65,247],[248,388],[389,601]]]
```

so that each row in the index list corresponds to a subject and the numbers in each bracket correspond to the start and end of an EEG recording. *In hindsight, a better option may be to store each recording in individual folders. This would reduce the risk of accidently concatenate files from different recordings due to programming errors or faulty information in the index file.*

#### Data format
Each EEG example that the networks process were 10 s in duration, or 2,560 samples. The numpy files hence had the size (21, 2560). The electrode order was (and must be for the scripts to work): FP1, F7 ,T3, T5, Fp2, F8, T4, T6, F3, C3, P3, O1, F4, C4, P4, O2, A1, A2, FZ, CZ, PZ.

#### Data split
The data was split in a 80, 10 and 10 percent distribution for training, validation and testing. The distribution was with regard to the number of subjects to keep the data sets disjoint.

---

## Network architecture

The network analyzed temporal and spatial dimensions separately. First, a series of convolutional layers analyzed the data for temporal features. Second, all electrodes were analyzed using a convolutional layer with kernel size equal to the number of electrodes, followed by upsampling to the correct number of electrodes by a convolutional transpose layer. Third, convolutional transpose layers assembles the signals. Fourth the network ends with a convolutional layer that merges all filters. LeakyReLU activations follow most convolutional layers. A schematic of the data flow through the network is shown below, illustrating how temporal and spatial dimensions are compressed/decompressed.

<p align="center">
<img src="https://github.com/Svanteberg/Virtual-EEG-electrodes/blob/master/images/network_data_flow.png" width="75%">
</p>

For example, the structure of GN1 was:

Temporal encoder block:
```
    def conv(self,x):
        # convolutional block
        for i in range(4):
            x = Conv2D(filters = 32*2**i, kernel_size = (1, 3), strides = (1, 2), padding = 'same')(x)
            x = LeakyReLU(alpha = 0.2)(x)
        return x
```

Spatial analysis:
```
        x = Conv2D(1024, kernel_size = (4, 1), strides = 1, padding = 'valid')(x)
        x = LeakyReLU(alpha = 0.2)(x)
        x = Conv2DTranspose(filters = 256, kernel_size = (17, 1), strides = 1, padding = 'valid')(x)
        x = LeakyReLU(alpha = 0.2)(x)
```

Temporal decoder
```
    def deconv(self,x):
        # deconvolutional block
        for i in range(4):
            x = Conv2DTranspose(filters = 32*2**(3 - i), kernel_size = (1, 3), strides = (1, 2), padding = 'same')(x)
            if i != 3:
                x = LeakyReLU(alpha = 0.2)(x)
        return x
```

Assembled network:
```
    def generator_model(self):
        input_eeg = Input(shape = (4,2560,1))
        # temporal encoder
        x = self.conv(input_eeg)
        # spatial analysis
        x = Conv2D(1024, kernel_size = (4, 1), strides = 1, padding = 'valid')(x)
        x = LeakyReLU(alpha = 0.2)(x)
        x = Conv2DTranspose(filters = 256, kernel_size = (17, 1), strides = 1, padding = 'valid')(x)
        x = LeakyReLU(alpha = 0.2)(x)
        # temporal decoder
        x = self.deconv(x)
        # merging all filters
        x = Conv2D(1,kernel_size = (1, 1), strides = 1)(x)
        return Model(inputs = input_eeg, outputs = x, name = 'generator')
```

---

## Training schedule

An epoch fo training was defined as training with one example from each subject in the training set. For each epoch of training, the training order of the subjects were randomized

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

The motivation for this was to allow for a variation in the training data to counteract overfitting, compared to using non-overlapping static 10 s examples. Given the realtively large total amount of data, it was not feasable to load all data and store it in the primary memory. Using 10 s instead of whole recordings hence allowed for faster loading times and more varied training content.

If the amplitude was between -500 and 500 µV, the example was accepted and used for training. If not, a new starting position in the recording was randomly chosen and the new example was checked for amplitude. This was repeated up to 100 times. If all 100 examples of that recording were rejected, the same procedure was performed for the next recording, and so on. If all examples of all recordings of a subject were rejected, no training took place that epoch for that specific subject.


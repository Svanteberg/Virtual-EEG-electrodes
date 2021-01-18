# Step-by-step demo

This demo will create a set of artificial data to work with, train a network of type GN1, generate new data which then will be visualized.

## A. Setting up Anaconda environment

The following instructions have been written for Linux. It is assumed that a computer with working installed (nVidia) graphics card and drivers are used.

1) To use / train the networks, first install Anaconda, https://www.anaconda.com/products/individual

```
    wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
    chmod +x Anaconda3-2020.07-Linux-x86_64.sh
    ./Anaconda3-2020.07-Linux-x86_64.sh
```
2) Then create an Anaconda environment from one of the available [environment files](https://github.com/Svanteberg/Virtual-EEG-electrodes/tree/master/Environments) (here examplified for [environment_1.yml](https://github.com/Svanteberg/Virtual-EEG-electrodes/blob/master/Environments/environment_1.yml); the file you choose of course have to downloaded to your computer)

```
    conda env create -f environment_1.yml
```

3) Activate the environment you just created

```
    conda activate eeg1
```

## B. Setting up folders

Positioned att *root*, create folders named:

1) *data*
2) *models*
3) *results*

Download the scripts:

4) `gn1.py`
5) `art_eeg.py`
6) plot_eeg_gn1.py`

and put them in *root*.

## C. Data

Create some artificial data using the available [script](https://github.com/Svanteberg/Virtual-EEG-electrodes/blob/master/Artificial_EEG_for_testing_scripts/art_eeg.py):

 ```
    python3.7 art_eeg.py
 ```

(This may take 20 - 40 min depending on hardware.)

## D. Train the GN1 network

```
    python3.7 gn1.py
```

This may consume a couple of hours.

## E. Use the trained network for generating new artificial data

1) Find the file `model.h5`in the right subfolder of the *results* folder.
2) Move it to the *models* folder
3) Run

```
    python3.7 gn1_generate_from_model.py
```

## F. Visualizing the data

1) Run

 ```
    python3.7 plot_eeg_gn1.py
 ```

2)  Use the 'Open directory' button to navigate in the *results* folder to find the EEG data you want to visualize.

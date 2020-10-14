To facilitate the process of setting up an organization for the data, an example is provided containing artificial EEG data. The script `art_eeg.py` can be used to generate more data. The data is simple, consisting of multiple sources with spatial distributions, including occasional muscle and blink artifacts, which are added together (linearly). The basic cortical signal is generated by amlitude modulating a sinus wave with a Markov process (adapted from: Bai et al., ”Nonlinear Markov process amplitude EEG model for nonlinear coupling interaction of spontaneous EEG,” IEEE Transactions of biomedicine engineering, vol. 47, nr 9, pp. 1141-1146, 2000).

<p align="center">
<img src="https://github.com/Svanteberg/Virtual-EEG-electrodes/blob/master/images/example_artificial_eeg.png" width="100%">
</p>
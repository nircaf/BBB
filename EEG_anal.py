#!/usr/bin/python
import os, zipfile, time, glob
import mne
import matplotlib.pyplot as plt
import scipy.fft as sfft
import numpy as np
import yasa

import pandas as pd
import antropy as ant
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=1.2)
from numpy import apply_along_axis as apply


def yassa(data):
    # Apply a bandpass filter from 0.1 to 40 Hz
    # data.filter(0.1, 40)
    raw_data = data.get_data()
    # choose one electrode
    one_electrode = raw_data[0, :]
    fig = yasa.plot_spectrogram(one_electrode, data.info["sfreq"])
    # Convert the EEG data to 30-sec data
    sf = data.info["sfreq"]
    times, data_win = yasa.sliding_window(one_electrode, data.info["sfreq"], window=30)
    # Convert times to minutes
    times /= 60
    feature_selection(data_win, times, data.info["sfreq"])

    # The simplest example.
    # yasa.topoplot(one_electrode, vmin=0, vmax=1);


def feature_selection(data_win, times, sf):
    df_feat = {
        # Entropy
        "perm_entropy": apply(ant.perm_entropy, axis=1, arr=data_win, normalize=True),
        "svd_entropy": apply(ant.svd_entropy, 1, data_win, normalize=True),
        "sample_entropy": apply(ant.sample_entropy, 1, data_win),
        # Fractal dimension
        "dfa": apply(ant.detrended_fluctuation, 1, data_win),
        "petrosian": apply(ant.petrosian_fd, 1, data_win),
        "katz": apply(ant.katz_fd, 1, data_win),
        "higuchi": apply(ant.higuchi_fd, 1, data_win),
    }
    df_feat = pd.DataFrame(df_feat)
    df_feat.head()
    df_feat["lziv"] = apply(lziv, 1, data_win)
    from scipy.signal import welch

    freqs, psd = welch(data_win, sf, nperseg=int(4 * sf))
    bp = yasa.bandpower_from_psd_ndarray(psd, freqs)
    bp = pd.DataFrame(
        bp.T, columns=["delta", "theta", "alpha", "sigma", "beta", "gamma"]
    )
    df_feat = pd.concat([df_feat, bp], axis=1)
    df_feat.head()
    from sklearn.feature_selection import f_classif

    # Extract sorted F-values
    fvals = pd.Series(
        f_classif(X=df_feat, y=hypno)[0], index=df_feat.columns
    ).sort_values()

    # Plot best features
    plt.figure(figsize=(6, 6))
    sns.barplot(y=fvals.index, x=fvals, palette="RdYlGn")
    plt.xlabel("F-values")
    plt.xticks(rotation=20)
    # Plot hypnogram and higuchi
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    hypno = pd.Series(hypno).map({-1: -1, 0: 0, 1: 2, 2: 3, 3: 4, 4: 1}).values
    hypno_rem = np.ma.masked_not_equal(hypno, 1)

    # Plot the hypnogram
    ax1.step(times, -1 * hypno, color="k", lw=1.5)
    ax1.step(times, -1 * hypno_rem, color="r", lw=2.5)
    ax1.set_yticks([0, -1, -2, -3, -4])
    ax1.set_yticklabels(["W", "R", "N1", "N2", "N3"])
    ax1.set_ylim(-4.5, 0.5)
    ax1.set_ylabel("Sleep stage")
    # Plot the non-linear feature
    ax2.plot(times, df_feat["higuchi"])
    ax2.set_ylabel("Higuchi Fractal Dimension")
    ax2.set_xlabel("Time [minutes]")
    ax2.set_xlim(0, times[-1])


def lziv(x):
    """Binarize the EEG signal and calculate the Lempel-Ziv complexity."""
    return ant.lziv_complexity(x > x.mean(), normalize=True)


import re
import glob


def get_values_from_file(filename):
    # Open the file and read the contents
    with open(filename) as f:
        text = f.read()

    # Use regular expressions to match the channel names
    channel_match = re.findall(r"Channel\s(\d+):\s(\w+)", text)

    # Create a dictionary from the matched channel names
    channels = dict(channel_match)

    # Use regular expressions to match the seizure details
    seizure_match = re.findall(
        r"Seizure n (\d+)\nFile name: ([\w\d-]+).edf\nRegistration start time: ([\d.]+)\nRegistration end time: ([\d.]+)\nSeizure start time: ([\d.]+)\nSeizure end time: ([\d.]+)",
        text,
    )
    # Create a dictionary from the matched seizure details
    seizures = {}
    for m in seizure_match:
        seizures[int(m[0])] = {
            "File name": m[1],
            "Registration start time": m[2],
            "Registration end time": m[3],
            "Seizure start time": m[4],
            "Seizure end time": m[5],
        }

    return channels, seizures


if __name__ == "__main__":
    # Get the list of all files ending with ".txt" in the current directory
    files = glob.glob(r"EEG_epilepsy/siena-scalp-eeg-database-1.0.0/**/*.txt")

    # Print the resulting dictionaries for each file
    for file in files:
        print(get_values_from_file(file))

    for eegfile in glob.glob("EEG_epilepsy/**/*.edf", recursive=True):
        # read edf file
        data = mne.io.read_raw_edf(eegfile)
        yassa(data)
        raw_data = data.get_data()
        # you can get the metadata included in the file and a list of all channels:
        info = data.info
        channels = data.ch_names

        eeg_fft = sfft.fft(raw_data)
        eeg_rfft = sfft.rfft(raw_data)
        xf = sfft.rfftfreq(data.n_times, 1 / data.info["sfreq"])

        plt.plot(xf, np.abs(eeg_rfft).T)
        plt.show()
        print("Number of channels: ", str(len(raw_data)))
        print("Number of samples: ", str(len(raw_data[0])))
        plt.plot(raw_data[0, :4999])
        plt.title("Raw EEG, electrode 0, samples 0-4999")
        plt.show()
        # Extract events from raw data
        events, event_ids = mne.events_from_annotations(data, event_id="auto")
        event_ids
        tmin, tmax = -1, 4  # define epochs around events (in s)
        # event_ids = dict(hands=2, feet=3)  # map event IDs to tasks

        epochs = mne.Epochs(
            data, events, event_ids, tmin - 0.5, tmax + 0.5, baseline=None, preload=True
        )

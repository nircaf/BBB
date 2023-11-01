import pyedflib
import numpy as np
import re
from collections import Counter
import mne
import glob


def convert_txt_to_edf(txt_file, edf_file, sample_rate=256):
    with open(txt_file, "r") as file:
        lines = file.readlines()
    # get line 7
    line_7 = lines[6].strip().split()
    # find items in line 7 that appear multiple times
    di_duplicates = {}
    counter = Counter(line_7)
    duplicates = [item for index, item in enumerate(line_7) if counter[item] > 1]
    # run over set(duplicates)
    for item in duplicates:
        di_duplicates[item] = [
            index for index, value in enumerate(line_7) if value == item
        ]
    # change duplicates to dict 'item': [index]
    duplicates_indices = [
        index for index, item in enumerate(line_7) if counter[item] > 1
    ]
    # get list without duplicates
    labels = list(set(line_7))
    header = {
        "technician": "",
        "recording_additional": "",
        "patientname": "Unknown",
        "patient_additional": "",
        "patientcode": "",
        "gender": "",
        "birthdate": "",
        "patient_additional": "",
        "admincode": "",
        "equipment": "Unknown",
        "recordingdate": "",
        "startdate": "",
        "starttime": "",
        "enddate": "",
        "endtime": "",
        "patient": "",
        "recording": "",
        "datarecord_duration": "",
        "datarecords_in_file": 1,
        "channels": [],
        "annotations": [],
        "sample_rate": sample_rate,
        "sample_sizes": [2] * len(labels),
        "label": labels,
        "physical_max": [1000.0] * len(labels),
        "physical_min": [-1000.0] * len(labels),
        "digital_max": [32767] * len(labels),
        "digital_min": [-32768] * len(labels),
        "prefiltering": [""] * len(labels),
        "transducer": [""] * len(labels),
    }

    data = np.zeros((len(labels), len(lines) - 2))
    pattern = re.compile(r"\d{2}:\d{2}:\d{2}")

    for i, line in enumerate(lines[8:]):
        values = line.strip().split()
        if not values:
            continue
        header["annotations"].append(values[0])
        j = -1
        jj = 0
        for value in values:
            # if value is time hh:mm:ss
            if re.match(pattern, value):
                continue
            j += 1
            if j in duplicates_indices:
                # get index of jj in duplicates_indices
                j_in_duplicates = duplicates_indices[duplicates_indices.index(j) - 1]
                # if index of jj in duplicates_indices is odd
                if duplicates_indices.index(j) % 2 == 1:
                    # get mean of data(jj,i) and data(jj_in_duplicates,i)
                    value = np.mean([data[j, i], data[j_in_duplicates, i]])
                else:
                    jj -= 1
            try:
                data[jj, i] = float(value)
                jj += 1
            except ValueError:
                try:
                    continue
                except:
                    pass

    """        channel_info should be a
        list of dicts, one for each channel in the data. Each dict needs
        these values:

            'label' : channel label (string, <= 16 characters, must be unique)
            'dimension' : physical dimension (e.g., mV) (string, <= 8 characters)
            'sample_rate' : sample frequency in hertz (int). Deprecated: use 'sample_frequency' instead.
            'sample_frequency' : number of samples per record (int)
            'physical_max' : maximum physical value (float)
            'physical_min' : minimum physical value (float)
            'digital_max' : maximum digital value (int, -2**15 <= x < 2**15)
            'digital_min' : minimum digital value (int, -2**15 <= x < 2**15)
            """
    channel_info = []
    # run over labels
    for i, label in enumerate(header["label"]):
        di = {}
        di["label"] = label
        di["dimension"] = "uV"
        di["sample_frequency"] = header["sample_rate"]
        di["physical_max"] = header["physical_max"][i]
        di["physical_min"] = header["physical_min"][i]
        di["digital_max"] = header["digital_max"][i]
        di["digital_min"] = header["digital_min"][i]
        channel_info.append(di)

    with pyedflib.EdfWriter(
        edf_file, len(header["label"]), file_type=pyedflib.FILETYPE_EDFPLUS
    ) as writer:
        writer.setPatientCode(header["patientcode"])
        writer.setPatientName(header["patientname"])
        writer.setTechnician(header["technician"])
        writer.setEquipment(header["equipment"])
        # run over labels
        for i, label in enumerate(header["label"]):
            writer.setPhysicalMaximum(i, header["physical_max"][i])
            writer.setSamplefrequency(i, header["sample_rate"])
            writer.setPhysicalMinimum(i, header["physical_min"][i])
            writer.setDigitalMaximum(i, header["digital_max"][i])
            writer.setDigitalMinimum(i, header["digital_min"][i])
            writer.setLabel(i, header["label"][i])
            writer.setTransducer(i, header["transducer"][i])
            writer.setPrefilter(i, header["prefiltering"][i])
        writer.writeSamples(data)

    print(f"Conversion complete. Saved as {edf_file}")
    # Load the .edf file
    raw = mne.io.read_raw_edf(edf_file)
    mat_file = edf_file[:-4] + ".mat"


# glob all .txt files in a folder
for file in glob.glob(r"eeg_txt/*.txt"):
    # Usage example
    convert_txt_to_edf(file, f"{file[:-4]}.edf")

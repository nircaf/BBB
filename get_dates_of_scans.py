import glob
import pandas as pd
from datetime import datetime


def all_epilepsy_eeg_mri_match():
    # runover EEG folder
    eeg_folder = "struct_double EEG"
    # get all the files in the folder
    eeg_files = glob.glob(eeg_folder + "/*.mat")
    # get the dates of the scans
    eeg_dates = [f.split("_")[-1][:-4] for f in eeg_files]
    # make df with [f.split('\\')[-1].split('_')[:1] for f in eeg_files]
    pat_code = ["_".join(f.split("\\")[-1].split("_")[:2]) for f in eeg_files]
    # get MRI scan date from
    df = pd.DataFrame({"code": pat_code, "EEG_date": eeg_dates})
    # run over Results_r_3 folder
    mri_folder = "Results r_3"
    # run over unique pat codes split [-1]
    init_codes = list(set(pat_code.split("_")[-1] for pat_code in pat_code))
    # run over init_codes
    for init_code in init_codes:
        # find the matching folder in mri_folder
        pat_scan_folder = glob.glob(mri_folder + "/*" + init_code + "*")
        # get .txt file from pat_scan_folder
        pat_scan_file = glob.glob(pat_scan_folder[0] + "/*.txt")
        bool_break = False
        for file_path in pat_scan_file:
            with open(file_path, "r") as file:
                for line in file:
                    if "Study date" in line:
                        # Process the line here
                        bool_break = True
                        break
            if bool_break:
                break
        # get the date from the line
        date = line.strip().split(": ")[-1]
        # convert yyyymmdd to dd.mm.yyyy
        date_obj = datetime.strptime(date, "%Y%m%d")
        formatted_date = date_obj.strftime("%d.%m.%Y")
        # add the date to the df
        df.loc[df["code"].str.contains(init_code), "MRI_date"] = formatted_date
    # get only dd.mm.yyyy and dd.mm.yy from str_t with re
    import re

    # run over df['EEG_date']
    for eegdate in df["EEG_date"]:
        # get only dd.mm.yyyy from eegdate with re
        eegdate2 = re.findall(r"\d{2}.\d{2}.\d{2}", eegdate)
        if len(eegdate2) > 0:
            df.loc[df["EEG_date"] == eegdate, "EEG_date"] = eegdate2[0]
        else:
            print(eegdate)

    # df['EEG_date'] remove non dd.mm.yyyy string
    df["EEG_date"] = df["EEG_date"].apply(
        lambda x: re.findall(r"\d{2}.\d{2}.\d{2}", x)[0]
        if len(re.findall(r"\d{2}.\d{2}.\d{2}", x)) > 0
        else None
    )
    # add column of abs difference between dates
    df["abs_diff"] = abs(
        pd.to_datetime(df["EEG_date"]) - pd.to_datetime(df["MRI_date"])
    )
    # save df to csv
    df.to_csv("Epilepsy all.csv", index=False)


def losartan_eeg_mri_match():
    # runover EEG folder
    eeg_folder = "struct_double EEG"
    # get all the files in the folder
    eeg_files = glob.glob(eeg_folder + "/*.mat")
    # run over Results_r_3 folder
    mri_folder = "Results_r_3_Los_After"
    mri_folder_pre = "Results r_3"
    mri_pre_folders = glob.glob(mri_folder_pre + "/*")
    df = pd.DataFrame(columns=["code", "EEG_date", "MRI_date"])
    # run over files in mri_folder
    for file_path in glob.glob(mri_folder + "/*"):
        initials_code = "_".join(file_path.split("\\")[-1].split("_")[:2])
        # get code from file_path
        code = file_path.split("\\")[-1].split("_")[1]
        # find match between code in eeg_files
        eeg_file = [f for f in eeg_files if code in f]
        # get the date of the scan
        eeg_dates = [f.split("_")[-1][:-4] for f in eeg_file]
        formatted_date = find_date_in_pat_scan_file(file_path)
        # find date in mri_folder_pre
        mri_pre_folder = [f for f in mri_pre_folders if code in f]
        formatted_date_pre = find_date_in_pat_scan_file(mri_pre_folder[0])
        # concat formatted_date and  formatted_date_pre
        formatted_date = formatted_date_pre + "\n" + formatted_date
        # create df with code, eeg_date, mri_date
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "code": initials_code,
                        "EEG_date": eeg_dates,
                        "MRI_date": formatted_date,
                    }
                ),
            ]
        )
    # save df to csv
    df.to_csv("Epilepsy losartan.csv", index=False)


def find_date_in_pat_scan_file(file_path):
    # find the matching folder in mri_folder
    pat_scan_folder = glob.glob(file_path)
    # get .txt file from pat_scan_folder
    pat_scan_file = glob.glob(pat_scan_folder[0] + "/*.txt")
    bool_break = False
    for file_path in pat_scan_file:
        with open(file_path, "r") as file:
            for line in file:
                if "Study date" in line:
                    # Process the line here
                    bool_break = True
                    break
        if bool_break:
            break
    # get the date from the line
    date = line.strip().split(": ")[-1]
    # convert yyyymmdd to dd.mm.yyyy
    date_obj = datetime.strptime(date, "%Y%m%d")
    formatted_date = date_obj.strftime("%d.%m.%Y")
    return formatted_date


def files_instruct_doubleEEG_not_inIndividualresults():
    path_full = "struct_double EEG"
    path_part = "EEG/Individual results"
    files_full = glob.glob(path_full + "/*.mat")
    files_part = glob.glob(path_part + "/*.pptx")
    files_full = [f.split("\\")[-1] for f in files_full]
    # remove .mat from files_full
    files_full = [f[:-4] for f in files_full]
    files_part = [f.split("\\")[-1] for f in files_part]
    # remove .pptx from files_part
    files_part = [f[:-5] for f in files_part]
    # find the files in files_full that are not in files_part
    files_not_in_part = [f for f in files_full if f not in files_part]
    # save files_not_in_part to csv
    df = pd.DataFrame({"files_not_in_Individual results": files_not_in_part})
    df.to_csv("files_not_in_Individual results.csv", index=False)

import pandas as pd
import os
import glob
import re
from scipy import stats

# read excel Epilepsy_clinical_data.xlsx sheet All
import numpy as np

area_names = [
    "Brain Stem",
    "Cerebellar Vermal Lobules I-V",
    "Cerebellar Vermal Lobules VI-VII",
    "Cerebellar Vermal Lobules VIII-X",
    "Left Accumbens Area",
    "Left ACgG anterior cingulate gyrus",
    "Left AIns anterior insula",
    "Left Amygdala",
    "Left AnG angular gyrus",
    "Left AOrG anterior orbital gyrus",
    "Left Basal Forebrain",
    "Left Calc calcarine cortex",
    "Left Caudate",
    "Left Cerebellum Exterior",
    "Left Cerebellum White Matter",
    "Left Cerebral White Matter",
    "Left CO central operculum",
    "Left Cun cuneus",
    "Left Ent entorhinal area",
    "Left FO frontal operculum",
    "Left FRP frontal pole",
    "Left FuG fusiform gyrus",
    "Left GRe gyrus rectus",
    "Left Hippocampus",
    "Left IOG inferior occipital gyrus",
    "Left ITG inferior temporal gyrus",
    "Left LiG lingual gyrus",
    "Left LOrG lateral orbital gyrus",
    "Left MCgG middle cingulate gyrus",
    "Left MFC medial frontal cortex",
    "Left MFG middle frontal gyrus",
    "Left MOG middle occipital gyrus",
    "Left MOrG medial orbital gyrus",
    "Left MPoG postcentral gyrus medial segment",
    "Left MPrG precentral gyrus medial segment",
    "Left MSFG superior frontal gyrus medial segment",
    "Left MTG middle temporal gyrus",
    "Left OCP occipital pole",
    "Left OFuG occipital fusiform gyrus",
    "Left OpIFG opercular part of the inferior frontal gyrus",
    "Left OrIFG orbital part of the inferior frontal gyrus",
    "Left Pallidum",
    "Left PCgG posterior cingulate gyrus",
    "Left PCu precuneus",
    "Left PHG parahippocampal gyrus",
    "Left PIns posterior insula",
    "Left PO parietal operculum",
    "Left PoG postcentral gyrus",
    "Left POrG posterior orbital gyrus",
    "Left PP planum polare",
    "Left PrG precentral gyrus",
    "Left PT planum temporale",
    "Left Putamen",
    "Left SCA subcallosal area",
    "Left SFG superior frontal gyrus",
    "Left SMC supplementary motor cortex",
    "Left SMG supramarginal gyrus",
    "Left SOG superior occipital gyrus",
    "Left SPL superior parietal lobule",
    "Left STG superior temporal gyrus",
    "Left Thalamus Proper",
    "Left TMP temporal pole",
    "Left TrIFG triangular part of the inferior frontal gyrus",
    "Left TTG transverse temporal gyrus",
    "Left Ventral DC",
    "Right Accumbens Area",
    "Right ACgG anterior cingulate gyrus",
    "Right AIns anterior insula",
    "Right Amygdala",
    "Right AnG angular gyrus",
    "Right AOrG anterior orbital gyrus",
    "Right Basal Forebrain",
    "Right Calc calcarine cortex",
    "Right Caudate",
    "Right Cerebellum Exterior",
    "Right Cerebellum White Matter",
    "Right Cerebral White Matter",
    "Right CO central operculum",
    "Right Cun cuneus",
    "Right Ent entorhinal area",
    "Right FO frontal operculum",
    "Right FRP frontal pole",
    "Right FuG fusiform gyrus",
    "Right GRe gyrus rectus",
    "Right Hippocampus",
    "Right IOG inferior occipital gyrus",
    "Right ITG inferior temporal gyrus",
    "Right LiG lingual gyrus",
    "Right LOrG lateral orbital gyrus",
    "Right MCgG middle cingulate gyrus",
    "Right MFC medial frontal cortex",
    "Right MFG middle frontal gyrus",
    "Right MOG middle occipital gyrus",
    "Right MOrG medial orbital gyrus",
    "Right MPoG postcentral gyrus medial segment",
    "Right MPrG precentral gyrus medial segment",
    "Right MSFG superior frontal gyrus medial segment",
    "Right MTG middle temporal gyrus",
    "Right OCP occipital pole",
    "Right OFuG occipital fusiform gyrus",
    "Right OpIFG opercular part of the inferior frontal gyrus",
    "Right OrIFG orbital part of the inferior frontal gyrus",
    "Right Pallidum",
    "Right PCgG posterior cingulate gyrus",
    "Right PCu precuneus",
    "Right PHG parahippocampal gyrus",
    "Right PIns posterior insula",
    "Right PO parietal operculum",
    "Right PoG postcentral gyrus",
    "Right POrG posterior orbital gyrus",
    "Right PP planum polare",
    "Right PrG precentral gyrus",
    "Right PT planum temporale",
    "Right Putamen",
    "Right SCA subcallosal area",
    "Right SFG superior frontal gyrus",
    "Right SMC supplementary motor cortex",
    "Right SMG supramarginal gyrus",
    "Right SOG superior occipital gyrus",
    "Right SPL superior parietal lobule",
    "Right STG superior temporal gyrus",
    "Right Thalamus Proper",
    "Right TMP temporal pole",
    "Right TrIFG triangular part of the inferior frontal gyrus",
    "Right TTG transverse temporal gyrus",
    "Right Ventral DC",
]
dict_sides = {
    "R": "Right",
    "L": "Left",
    "T": "temporal",
    "F": "frontal",
    "P": "parietal",
    "O": "occipital",
}


def read_data():
    df = pd.read_excel(r"Epilepsy_clinical_data.xlsx", sheet_name="All")
    df = df.dropna(how="all")
    # remove rows after 33
    df = df.iloc[:34]
    return df


# define a function to check if a string contains another string
def check_strings(str1, str2):
    # convert to string
    str1 = str(str1)
    str2 = str(str2)
    return str2 in str1 or str1 in str2


def anal_eeg_lesion(df):
    df = read_data()
    col_to_comp = ["EEG", "Lesion"]
    # compare EEG and Lesion
    df2 = df[col_to_comp].copy()
    df2["contains"] = df.apply(
        lambda row: check_strings(row[col_to_comp[0]], row[col_to_comp[1]]), axis=1
    )
    # find number of NaN in EEG
    num_nan = df2[col_to_comp[0]].isnull().sum()
    # sum true and false in contains
    num_true = df2["contains"].sum()
    num_false = len(df2) - num_true
    num_false = num_false - num_nan
    # run 1 var ttest on contains
    ttest = stats.ttest_1samp(df2["contains"], 0)
    dict_resulst = {
        "matches": num_true,
        "not_matches": num_false,
        "num_nan": num_nan,
        "total": len(df2),
        "percent": 100 * num_true / (num_false + num_true),
        "ttest_pval": ttest[1],
    }


def missing_pat_data(df):
    # get column code
    codes = df["code"]
    # get 4 numbers after _
    codes = codes.str.split("_").str[1]
    # clean NaN
    codes = codes.dropna()

    # get names of files from Clinical Data
    files = glob.glob(r"C:\Nir\BBB\Clinical Data\*")
    # get name without path
    files = [os.path.basename(x) for x in files]
    # get list of 4 numbers
    files = [re.findall(r"\d+", x)[0] for x in files]

    # compare codes and files with set
    codes = set(codes)
    files = set(files)
    # get difference
    diff = codes.difference(files)
    # find in df['code'] the rows with diff
    df_clinical_miss = pd.DataFrame()
    for i in diff:
        df_clinical_miss = pd.concat([df_clinical_miss, df[df["code"].str.contains(i)]])
    df_clinical_miss = df[df["code"].str.contains("|".join(diff))]
    print(df_clinical_miss["code"])


import math
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import matplotlib.pyplot as plt


def get_df(mat, lin_tofts):
    exit_while = False
    df = pd.DataFrame()
    i = 0
    while not exit_while:
        try:
            # concat to df with column name mat['allstruct'][0][i][1][0]
            if lin_tofts == "Slow":
                df_t = pd.DataFrame(mat["allstruct"][0][i][9][0])
            else:
                df_t = pd.DataFrame(mat["allstruct"][0][i][-1][0])  # lin =9 tofts = -1
            # rename column
            df_t = df_t.rename(columns={0: mat["allstruct"][0][i][1][0]})
            df = pd.concat([df, df_t], axis=1)
            # rename column
            i += 1
        except:
            exit_while = True
    # change df index to index=[x[0][0] for x in mat['areas_names']]
    df.index = area_names
    return df


def get_mat_data(lin_tofts="lin"):
    df_lesion = read_data()
    # set code as index
    df_lesion = df_lesion.set_index("code")
    file = r"from_matlab_to_python/epilepsy_controls_g_f.mat"
    import scipy.io

    mat = scipy.io.loadmat(file)
    df = get_df(mat, lin_tofts)
    df_controls = pd.DataFrame(mat["result_mat_tofts_age_control"]).T
    # rename index to index=[x[0][0] for x in mat['areas_names']] if type(x[0][0])==str else x[0][0][0]
    df_controls.index = area_names
    df_results = pd.DataFrame(index=df.index, columns=df.columns)
    # fill with 0
    df_results = df_results.fillna(0)
    # ---------- STD 2 run
    # run over each row in df and df_controls and
    for index, row, row_controls in zip(df.index, df.values, df_controls.values):
        controls_std = row_controls.std()
        controls_mean = row_controls.mean()
        # check if each val in row is in 2 std from mean
        for i in range(len(row)):
            # get column name
            col_name = df.columns[i]
            # get lesion data
            lesion_data = df_lesion.loc[col_name]["Lesion"]
            if lesion_data == 0 or np.NaN:
                continue
            # check if lesion data is right side (R) or left side (L) and match with index name
            if (
                "Right" in index
                and "R" in lesion_data
                or "Left" in index
                and "L" in lesion_data
            ):
                # put 1 in df_results if in 2 std
                if row[i] > controls_mean + 2 * controls_std:
                    df_results.loc[index][i] = 1
    df_percent = df_results.sum(axis=0) / len(df.index)

    all_arr_lesion, all_arr_rest = lesion_vs_bbb(
        df, lesion_data, df_lesion, df_controls, "Lesion", lin_tofts
    )
    all_arr_lesion, all_arr_rest = lesion_vs_std(
        df, lesion_data, df_lesion, df_controls, "Lesion", lin_tofts
    )

    # all_arr_lesion, all_arr_rest = lesion_vs_bbb(df,lesion_data,df_lesion,df_controls,'Focal Type Side',lin_tofts)


def lesion_vs_std(
    df, lesion_data, df_lesion, df_controls, str_col="Lesion", lin_tofts="lin"
):
    all_arr_lesion = []
    all_arr_rest = []
    pvalues = []
    # run over df columns
    for col in df.columns:
        lesion_data = df_lesion.loc[col][str_col]
        try:
            # remove , and duplicate letters
            lesion_data = lesion_data.replace(",", "")
            lesion_data = "".join(sorted(set(lesion_data), key=lesion_data.index))
            # change lesion data to dict_sides
            lesion_data = [dict_sides[x] for x in list(lesion_data)]
        except:
            continue
        arr_lesion = []
        arr_rest = []
        # run over each row in df and df_controls and
        for index, row, row_controls in zip(
            df[col].index, df[col].values, df_controls.values
        ):
            controls_std = row_controls.std()
            controls_mean = row_controls.mean()
            # val = how many std above mean is row
            val = (row - controls_mean) / controls_std
            # check if each val in row is in 2 std from mean
            # if col_name is in lesion_data put it in arr_lesion
            # if index is in lesion_data put it in arr_lesion
            if sum(item in index for item in lesion_data) >= 2:
                arr_lesion.append(val)
            else:
                arr_rest.append(val)
        # remove nan from both
        arr_lesion = [x for x in arr_lesion if not math.isnan(x)]
        arr_rest = [x for x in arr_rest if not math.isnan(x)]
        # run ttest on arr_lesion and arr_rest
        ttest = stats.ttest_ind(arr_lesion, arr_rest)
        # if pval sig
        if ttest[1] < 0.05:
            print(col, ttest[1])
        pvalues.append(ttest[1])
        all_arr_lesion.extend(arr_lesion)
        all_arr_rest.extend(arr_rest)
    # multiple comparison correction
    reject, pvals_corrected, alphacSidak, alphacBonf = multipletests(
        pvalues, alpha=0.05, method="fdr_bh"
    )
    # ttest all arr_lesion and arr_rest
    ttest = stats.ttest_ind(all_arr_lesion, all_arr_rest)
    print(f"pval: {ttest[1]} for test {str_col}")
    # Determine the desired length for the arrays
    desired_length = max(len(all_arr_lesion), len(all_arr_rest))

    # Truncate or pad the arrays to the desired length
    lesion_arr = np.pad(
        all_arr_lesion[:desired_length],
        (0, desired_length - len(all_arr_lesion)),
        "constant",
        constant_values=np.nan,
    )
    rest_arr = np.pad(
        all_arr_rest[:desired_length],
        (0, desired_length - len(all_arr_rest)),
        "constant",
        constant_values=np.nan,
    )

    # Create DataFrame with the arrays
    df = pd.DataFrame(
        {f"{str_col} {lin_tofts}": lesion_arr, f"Rest {lin_tofts}": rest_arr}
    )

    # Save DataFrame to CSV
    df.to_csv(f"std_{str_col}_{lin_tofts}.csv", index=False)
    return all_arr_lesion, all_arr_rest


def lesion_vs_bbb(
    df, lesion_data, df_lesion, df_controls, str_col="Lesion", lin_tofts="lin"
):
    all_arr_lesion = []
    all_arr_rest = []
    pvalues = []
    # run over df columns
    for col in df.columns:
        lesion_data = df_lesion.loc[col][str_col]
        try:
            # remove , and duplicate letters
            lesion_data = lesion_data.replace(",", "")
            lesion_data = "".join(sorted(set(lesion_data), key=lesion_data.index))
            # change lesion data to dict_sides
            lesion_data = [dict_sides[x] for x in list(lesion_data)]
        except:
            continue
        arr_lesion = []
        arr_rest = []
        # run over each row in df and df_controls and
        for index, row, row_controls in zip(
            df[col].index, df[col].values, df_controls.values
        ):
            controls_std = row_controls.std()
            controls_mean = row_controls.mean()
            # check if each val in row is in 2 std from mean
            # if col_name is in lesion_data put it in arr_lesion
            # if index is in lesion_data put it in arr_lesion
            if sum(item in index for item in lesion_data) >= 2:
                arr_lesion.append(row)
            else:
                arr_rest.append(row)
        # remove nan from both
        arr_lesion = [x for x in arr_lesion if not math.isnan(x)]
        arr_rest = [x for x in arr_rest if not math.isnan(x)]
        # run ttest on arr_lesion and arr_rest
        ttest = stats.ttest_ind(arr_lesion, arr_rest)
        # if pval sig
        if ttest[1] < 0.05:
            print(col, ttest[1])
        pvalues.append(ttest[1])
        all_arr_lesion.extend(arr_lesion)
        all_arr_rest.extend(arr_rest)
    # multiple comparison correction
    reject, pvals_corrected, alphacSidak, alphacBonf = multipletests(
        pvalues, alpha=0.05, method="fdr_bh"
    )
    # ttest all arr_lesion and arr_rest
    ttest = stats.ttest_ind(all_arr_lesion, all_arr_rest)
    print(f"pval: {ttest[1]} for test {str_col}")
    # Determine the desired length for the arrays
    desired_length = max(len(all_arr_lesion), len(all_arr_rest))

    # Truncate or pad the arrays to the desired length
    lesion_arr = np.pad(
        all_arr_lesion[:desired_length],
        (0, desired_length - len(all_arr_lesion)),
        "constant",
        constant_values=np.nan,
    )
    rest_arr = np.pad(
        all_arr_rest[:desired_length],
        (0, desired_length - len(all_arr_rest)),
        "constant",
        constant_values=np.nan,
    )

    # Create DataFrame with the arrays
    df = pd.DataFrame(
        {f"{str_col} {lin_tofts}": lesion_arr, f"Rest {lin_tofts}": rest_arr}
    )

    # Save DataFrame to CSV
    df.to_csv(f"data_{str_col}_{lin_tofts}.csv", index=False)
    return all_arr_lesion, all_arr_rest


import pandas as pd
import glob


def merge_csv_files(folder_path, output_file):
    # Find all CSV files in the folder
    file_pattern = f"{folder_path}/*.csv"
    csv_files = glob.glob(file_pattern)
    # remove output_file from csv files
    csv_files = [x for x in csv_files if x != output_file]
    # Initialize an empty DataFrame to store the merged data
    merged_data = pd.DataFrame()

    # Read and merge the CSV files
    for file in csv_files:
        df = pd.read_csv(file)
        merged_data = pd.concat([merged_data, df], axis=1)

    # Save the merged DataFrame to a CSV file
    merged_data.to_csv(output_file, index=False)


import os

if __name__ == "__main__":
    get_mat_data("Slow")
    get_mat_data("Fast")
    # Example usage
    folder_path = os.getcwd()
    output_file = "merged.csv"
    merge_csv_files(folder_path, output_file)

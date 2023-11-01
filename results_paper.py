import glob
import pandas as pd
import numpy as np
import glob
import datetime
from scipy.stats import mannwhitneyu


def get_data_from_df(Epilepsy_clinical_data):
    di = {}
    #  (mean age = 27.39 ± 9.98, 53.57% male)
    # get age mean and std
    di["age_mean"] = Epilepsy_clinical_data["'age'"].mean()
    di["age_std"] = Epilepsy_clinical_data["'age'"].std()
    # Epilepsy_clinical_data["'gender'"]=="'M'" or Epilepsy_clinical_data["'gender'"]=='M'
    di["male_mean"] = (
        sum(Epilepsy_clinical_data["'gender'"] == "'M'")
        + sum(Epilepsy_clinical_data["'gender'"] == "M")
    ) / len(Epilepsy_clinical_data["'gender'"])
    di["BBBP_lin_mean"] = Epilepsy_clinical_data["BBBP_lin"].mean()
    di["BBBP_tofts_mean"] = Epilepsy_clinical_data["BBBP_tofts"].mean()
    di["BBBP_lin_std"] = Epilepsy_clinical_data["BBBP_lin"].std()
    di["BBBP_tofts_std"] = Epilepsy_clinical_data["BBBP_tofts"].std()
    di["Linear regions % above 2SD_mean"] = Epilepsy_clinical_data[
        "Linear regions % above 2SD"
    ].mean()
    di["Linear regions % above 2SD_std"] = Epilepsy_clinical_data[
        "Linear regions % above 2SD"
    ].std()
    di["Tofts regions % above 2SD_mean"] = Epilepsy_clinical_data[
        "Tofts regions % above 2SD"
    ].mean()
    di["Tofts regions % above 2SD_std"] = Epilepsy_clinical_data[
        "Tofts regions % above 2SD"
    ].std()
    # create (mean age = 27.39 ± 9.98, 53.57% male) from di
    di[
        "str_age_geder"
    ] = f'mean age = {di["age_mean"]:.2f} ± {di["age_std"]:.2f}, {di["male_mean"]*100:.2f}% male'
    """ Regression analysis of brain volume using the Tofts model (see Methods) in all epileptic patients showed that 9.99 ± 9.74% of voxels exhibited BBBD, while the percentage of areas with BBBD was 23.64 ± 21.92%. Using the Linear model (see Methods), regression analysis revealed that 14.26 ± 13.91% of voxels had BBBD, with a corresponding percentage of areas with BBBD at 33.19 ± 30.20%."""
    di[
        "tofts"
    ] = f'Regression analysis of brain volume using the Tofts model (see Methods) in all epileptic patients showed that {di["BBBP_tofts_mean"]:.2f} ± {di["BBBP_tofts_std"]:.2f}% of voxels exhibited BBBD, while the percentage of areas with BBBD was {di["Tofts regions % above 2SD_mean"]:.2f} ± {di["Tofts regions % above 2SD_std"]:.2f}%.'
    di[
        "linear"
    ] = f'Using the Linear model (see Methods), regression analysis revealed that {di["BBBP_lin_mean"]:.2f} ± {di["BBBP_lin_std"]:.2f}% of voxels had BBBD, with a corresponding percentage of areas with BBBD at {di["Linear regions % above 2SD_mean"]:.2f} ± {di["Linear regions % above 2SD_std"]:.2f}%.'
    di["results"] = f'{di["tofts"]}. {di["linear"]}'
    return di


def stat_compare(focal_df, general_df):
    """Statistical comparisons demonstrated significant differences in fast BBBD% between groups (p=0.007) as well as in the percentage of areas with BBBD (p=0.009). For the slow model, there were significant differences in BBBD% between groups (p<0.001), as well as in the percentage of areas with BBBD (p<0.0001)."""
    # do mann whitney u test for BBBP_lin, BBBP_tofts, Linear regions % above 2SD, Tofts regions % above 2SD
    di = {}
    di["BBBP_lin"] = mannwhitneyu(focal_df["BBBP_lin"], general_df["BBBP_lin"])
    di["BBBP_tofts"] = mannwhitneyu(focal_df["BBBP_tofts"], general_df["BBBP_tofts"])
    di["Linear regions % above 2SD"] = mannwhitneyu(
        focal_df["Linear regions % above 2SD"], general_df["Linear regions % above 2SD"]
    )
    di["Tofts regions % above 2SD"] = mannwhitneyu(
        focal_df["Tofts regions % above 2SD"], general_df["Tofts regions % above 2SD"]
    )
    # get mean of di
    di_mean = np.mean(
        [
            di["BBBP_lin"][0],
            di["BBBP_tofts"][0],
            di["Linear regions % above 2SD"][0],
            di["Tofts regions % above 2SD"][0],
        ]
    )
    if di_mean < 0.05:
        di[
            "results"
        ] = f'Statistical comparisons demonstrated significant differences in fast BBBD% between groups (p={di["BBBP_lin"][1]:.3f}) as well as in the percentage of areas with BBBD (p={di["Linear regions % above 2SD"][1]:.3f}). For the slow model, there were significant differences in BBBD% between groups (p={di["BBBP_tofts"][1]:.3f}), as well as in the percentage of areas with BBBD (p={di["Tofts regions % above 2SD"][1]:.3f}).'
    else:
        # Statistical comparison of fast BBBD% between groups p=0.46, and for percentage of areas with BBBD p=0.77. For the slow model, BBBD% between groups p=0.90, and for percentage of areas with BBBD p=0.95.
        di[
            "results"
        ] = f'Statistical comparison of fast BBBD% between groups p={di["BBBP_lin"][1]:.2f}, and for percentage of areas with BBBD p={di["Linear regions % above 2SD"][1]:.2f}. For the slow model, BBBD% between groups p={di["BBBP_tofts"][1]:.2f}, and for percentage of areas with BBBD p={di["Tofts regions % above 2SD"][1]:.2f}.'
    return di


# load Epilepsy_clinical_data.xlsx
Epilepsy_clinical_data = pd.read_excel("Epilepsy_clinical_data.xlsx", sheet_name="All")
controls_df = pd.read_excel("Epilepsy_clinical_data.xlsx", sheet_name="Controls")
# remove rows after index 28
Epilepsy_clinical_data = Epilepsy_clinical_data.iloc[:29]
Epilepsy_clinical_data.columns
di_control_eppilepsy = stat_compare(controls_df, Epilepsy_clinical_data)
di_epilepsy = get_data_from_df(Epilepsy_clinical_data)
focal_df = Epilepsy_clinical_data[Epilepsy_clinical_data["Focal/General"] == "F"]
di_f = get_data_from_df(focal_df)
general_df = Epilepsy_clinical_data[Epilepsy_clinical_data["Focal/General"] == "G"]
di_g = get_data_from_df(general_df)
di_f_g = stat_compare(focal_df, general_df)

print("done")

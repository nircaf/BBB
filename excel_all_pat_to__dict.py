import pandas as pd
import os


def get_folders():
    this_folder = os.getcwd()
    files_location = this_folder + r"\Excel not age gender match"
    excel_file = files_location + r"\merged.xlsx"
    controls_file = files_location + r"\Controls.xlsx"
    clinical_data = this_folder + r"\Epilepsy_clinical_data.xlsx"
    return files_location, excel_file, controls_file, clinical_data


def get_results():
    # this folder
    files_location, excel_file, controls_file, clinical_data = get_folders()
    clinical_data_df = pd.read_excel(clinical_data, sheet_name="All")
    Epilepsy_clinical_data = clinical_data_df
    # Epilepsy_clinical_data drop after row 47
    Epilepsy_clinical_data = Epilepsy_clinical_data.iloc[:44, :]
    di = {}
    result_mat_lin_age = pd.read_excel(excel_file, sheet_name="BBB_percent_Linear")
    result_mat_lin_age = get_126_areas(clinical_data_df, result_mat_lin_age)
    controls_lin = pd.read_excel(controls_file, sheet_name="BBB_percent_Linear")
    #  (mean age = 27.39 ± 9.98, 53.57% male)
    # get age mean and std
    di["age_mean"] = Epilepsy_clinical_data["'age'"].mean()
    di["age_std"] = Epilepsy_clinical_data["'age'"].std()
    epilepsy_gender = Epilepsy_clinical_data["'gender'"]
    epilepsy_gender = epilepsy_gender.str.strip("'").str.strip().str.upper()
    # Epilepsy_clinical_data["'gender'"]=="'M'" or Epilepsy_clinical_data["'gender'"]=='M'
    di["male_mean"] = (sum(epilepsy_gender == "M")) / len(epilepsy_gender)
    di["BBBP_lin_mean"] = result_mat_lin_age["Unnamed: 1"].mean()
    di["BBBP_lin_std"] = result_mat_lin_age["Unnamed: 1"].std()
    controls_mean = controls_lin["Linear"].mean()
    contorls_std = controls_lin["Linear"].std()
    control_gender = controls_lin["Gender"]
    controls_age = controls_lin["age"]
    control_gender = control_gender.str.strip("'").str.strip().str.upper()
    di["male_mean_controls"] = (sum(control_gender == "M")) / len(control_gender)
    di[
        "str_age_gender"
    ] = f'mean age = {di["age_mean"]:.2f} ± {di["age_std"]:.2f}, {di["male_mean"]*100:.2f}% male'
    focal_epilepsy_age = Epilepsy_clinical_data[
        Epilepsy_clinical_data["Focal/General"] == "F"
    ]["'age'"]
    focal_epilepsy_gender = Epilepsy_clinical_data[
        Epilepsy_clinical_data["Focal/General"] == "F"
    ]["'gender'"]
    focal_epilepsy_gender = focal_epilepsy_gender.str.strip("'").str.strip().str.upper()
    male_mean_f = (sum(focal_epilepsy_gender == "M")) / len(focal_epilepsy_gender)
    di[
        "str_age_focal"
    ] = f"mean age = {focal_epilepsy_age.mean():.2f} ± {focal_epilepsy_age.std():.2f}, {male_mean_f*100:.2f}% male"
    g_epilepsy_age = Epilepsy_clinical_data[
        Epilepsy_clinical_data["Focal/General"] == "G"
    ]["'age'"]
    g_epilepsy_gender = Epilepsy_clinical_data[
        Epilepsy_clinical_data["Focal/General"] == "G"
    ]["'gender'"]
    g_epilepsy_gender = g_epilepsy_gender.str.strip("'").str.strip().str.upper()
    male_mean_g = (sum(g_epilepsy_gender == "M")) / len(g_epilepsy_gender)
    di[
        "str_age_g"
    ] = f"mean age = {g_epilepsy_age.mean():.2f} ± {g_epilepsy_age.std():.2f}, {male_mean_g*100:.2f}% male"
    di[
        "str_age_gender_controls"
    ] = f'mean age = {controls_age.mean():.2f} ± {controls_age.std():.2f}, {di["male_mean_controls"]*100:.2f}% male'
    # do mann whitney u test on age
    from scipy.stats import mannwhitneyu

    stat, di["p_age_controls_epilepsy"] = mannwhitneyu(
        controls_age, Epilepsy_clinical_data["'age'"]
    )
    # mann whitney on gender
    stat, di["p_gender_controls_epilepsy"] = mannwhitneyu(
        control_gender == "M", Epilepsy_clinical_data["'gender'"].dropna() == "M"
    )
    # mann whitney on focal epilepsy age
    stat, di["p_age_focal_general"] = mannwhitneyu(controls_age, focal_epilepsy_age)
    # mann whitney on focal epilepsy gender
    g_epilepsy_gender = g_epilepsy_gender.dropna().reset_index(drop=True)
    focal_epilepsy_gender = focal_epilepsy_gender.dropna().reset_index(drop=True)
    stat, di["p_gender_focal_general"] = mannwhitneyu(
        g_epilepsy_gender == "M", focal_epilepsy_gender == "M"
    )
    frontal_epilepsy = Epilepsy_clinical_data[
        Epilepsy_clinical_data["Focal Type"].str.contains("F") == True
    ]
    frontal_epilepsy_age = frontal_epilepsy["'age'"]
    frontal_epilepsy_gender = frontal_epilepsy["'gender'"]
    frontal_epilepsy_gender = (
        frontal_epilepsy_gender.str.strip("'").str.strip().str.upper()
    )
    male_mean_frontal = (sum(frontal_epilepsy_gender == "M")) / len(
        frontal_epilepsy_gender
    )
    temporal_epilepsy = Epilepsy_clinical_data[
        Epilepsy_clinical_data["Focal Type"].str.contains("T") == True
    ]
    temporal_epilepsy_age = temporal_epilepsy["'age'"]
    temporal_epilepsy_gender = temporal_epilepsy["'gender'"]
    temporal_epilepsy_gender = (
        temporal_epilepsy_gender.str.strip("'").str.strip().str.upper()
    )
    male_mean_temporal = (sum(temporal_epilepsy_gender == "M")) / len(
        temporal_epilepsy_gender
    )
    stat, di["p_age_frontal_temporal"] = mannwhitneyu(
        frontal_epilepsy_age, temporal_epilepsy_age
    )
    stat, di["p_gender_frontal_temporal"] = mannwhitneyu(
        frontal_epilepsy_gender == "M", temporal_epilepsy_gender == "M"
    )
    di[
        "paper"
    ] = f"""To conduct the study we included {len(Epilepsy_clinical_data)} epileptic patients (mean age = {di["age_mean"]:.2f} ±
    {di["age_std"]:.2f}, {di['male_mean']*100:.2f}% male). And {len(controls_lin)} healthy controls (mean age = {controls_age.mean():.2f} ± {controls_age.std():.2f}, {100*di['male_mean_controls']:.2f}% male).
    Among the epileptic patients, {len(focal_epilepsy_age)} had focal epilepsy
    (mean age = {focal_epilepsy_age.mean():.2f} ± {focal_epilepsy_age.std():.2f}, {male_mean_f*100:.2f}% male),
    this group divides into {len(frontal_epilepsy)} frontal epileptic patients (mean age = {frontal_epilepsy_age.mean():.2f} ± {frontal_epilepsy_age.std():.2f}, {male_mean_frontal*100:.2f}% male),
    and to {len(temporal_epilepsy)} temporal epileptic patients (mean age = {temporal_epilepsy_age.mean():.2f} ± {temporal_epilepsy_age.std():.2f}, {male_mean_temporal*100:.2f}% male).
    and {len(g_epilepsy_age)} had generalized epilepsy
    (mean age = {g_epilepsy_age.mean():.2f} ± {g_epilepsy_age.std():.2f}, {male_mean_g*100:.2f}% male).
    After conducting Mann-Whitney U tests, there were no significant differences in age (p={di['p_age_controls_epilepsy']:.2f})
    or gender distribution (p={di['p_gender_controls_epilepsy']:.2f}) between the controls and epileptic patients. Additionally,
    there were no significant differences between focal and generalized epileptic patients in age (p={di['p_age_focal_general']:.2f})
    or gender distribution (p={di['p_gender_focal_general']:.2f}).
    Finally, there were no significant differences between frontal and temporal epileptic patients in age (p={di['p_age_frontal_temporal']:.2f}) or gender distribution (p={di['p_gender_frontal_temporal']:.2f}).
    """.replace(
        "\n", " "
    )
    # create df of result_mat_lin_age["Unnamed: 1"] and controls_lin["Linear"]
    df = pd.DataFrame(
        {
            "Controls": controls_lin["Linear"],
            "Epilepsy": result_mat_lin_age["Unnamed: 1"],
            "Focal Epilepsy": result_mat_lin_age[
                result_mat_lin_age["ID"]
                .iloc[:, 1]
                .isin(
                    clinical_data_df[
                        clinical_data_df["Focal/General"].str.contains("F") == True
                    ]["code"]
                )
            ]["Unnamed: 1"].reset_index(drop=True),
            "Generalized Epilepsy": result_mat_lin_age[
                result_mat_lin_age["ID"]
                .iloc[:, 1]
                .isin(
                    clinical_data_df[
                        clinical_data_df["Focal/General"].str.contains("G") == True
                    ]["code"]
                )
            ]["Unnamed: 1"].reset_index(drop=True),
            "Frontal Epilepsy": result_mat_lin_age[
                result_mat_lin_age["ID"]
                .iloc[:, 1]
                .isin(
                    clinical_data_df[
                        clinical_data_df["Focal Type"].str.contains("F") == True
                    ]["code"]
                )
            ]["Unnamed: 1"].reset_index(drop=True),
            "Temporal Epilepsy": result_mat_lin_age[
                result_mat_lin_age["ID"]
                .iloc[:, 1]
                .isin(
                    clinical_data_df[
                        clinical_data_df["Focal Type"].str.contains("T") == True
                    ]["code"]
                )
            ]["Unnamed: 1"].reset_index(drop=True),
        }
    )
    result_mat_lin_age126 = pd.read_excel(excel_file, sheet_name="126_Regions_Linear")
    controls_126 = pd.read_excel(controls_file, sheet_name="126_Regions_Linear").drop(
        columns=["'ID'"]
    )
    controls_126.columns = controls_126.columns.str.strip("'")
    controls_126_mean = controls_126.mean()
    controls_126_std = controls_126.std()
    df_126 = get_126_areas(clinical_data_df, result_mat_lin_age126)
    df_126_focal = (
        df_126[
            df_126["ID"].isin(
                clinical_data_df[
                    clinical_data_df["Focal/General"].str.contains("F") == True
                ]["code"]
            )
        ]
        .reset_index(drop=True)
        .drop(columns=["ID"])
    )
    df_126_general = (
        df_126[
            df_126["ID"].isin(
                clinical_data_df[
                    clinical_data_df["Focal/General"].str.contains("G") == True
                ]["code"]
            )
        ]
        .reset_index(drop=True)
        .drop(columns=["ID"])
    )
    df_126_frontal = (
        df_126[
            df_126["ID"].isin(
                clinical_data_df[
                    clinical_data_df["Focal Type"].str.contains("F") == True
                ]["code"]
            )
        ]
        .reset_index(drop=True)
        .drop(columns=["ID"])
    )
    df_126_temporal = (
        df_126[
            df_126["ID"].isin(
                clinical_data_df[
                    clinical_data_df["Focal Type"].str.contains("T") == True
                ]["code"]
            )
        ]
        .reset_index(drop=True)
        .drop(columns=["ID"])
    )
    df_126 = df_126.drop(columns=["ID"])
    # for each row in result_mat_lin_age get % of areas above 2 sd of controls
    df_2sd = pd.DataFrame(
        {
            "Controls": controls_126.apply(
                lambda row: 100
                * sum((row - controls_126_mean) / controls_126_std >= 2)
                / len(controls_126_mean),
                axis=1,
            ).reset_index(drop=True),
            "Epilepsy": df_126.apply(
                lambda row: 100
                * sum((row - controls_126_mean) / controls_126_std >= 2)
                / len(controls_126_mean),
                axis=1,
            ).reset_index(drop=True),
            "Focal Epilepsy": df_126_focal.apply(
                lambda row: 100
                * sum((row - controls_126_mean) / controls_126_std >= 2)
                / len(controls_126_mean),
                axis=1,
            ).reset_index(drop=True),
            "Generalized Epilepsy": df_126_general.apply(
                lambda row: 100
                * sum((row - controls_126_mean) / controls_126_std >= 2)
                / len(controls_126_mean),
                axis=1,
            ).reset_index(drop=True),
            "Frontal Epilepsy": df_126_frontal.apply(
                lambda row: 100
                * sum((row - controls_126_mean) / controls_126_std >= 2)
                / len(controls_126_mean),
                axis=1,
            ).reset_index(drop=True),
            "Temporal Epilepsy": df_126_temporal.apply(
                lambda row: 100
                * sum((row - controls_126_mean) / controls_126_std >= 2)
                / len(controls_126_mean),
                axis=1,
            ).reset_index(drop=True),
        }
    )
    df2 = pd.concat([df, df_2sd], axis=1)
    # df to csv
    # df2.to_csv("figures/df.csv")
    return df, df_2sd
    df_epilepsy = pd.DataFrame(
        {
            "Epilepsy": result_mat_lin_age["Unnamed: 1"],
        }
    )
    result_mat_lin_age.to_csv("figures/df_epilepsy.csv", index=False)
    df_126.to_csv("figures/df_126.csv", index=False)


def results_paper_dyn():
    # import mannwhitneyu
    from scipy.stats import mannwhitneyu
    import math

    df, df_2sd = get_results()
    mat = get_mat()
    mat_lin_control = mat["result_mat_lin_age_control"]
    mat_tofts_control = mat["result_mat_tofts_age_control"]
    mat_lin = mat["result_mat_lin_age"]
    mat_tofts = mat["result_mat_tofts_age"]
    focal_pat_mat_lin = mat["focal_pat_mat_lin"]
    focal_pat_mat_tofts = mat["focal_pat_mat_tofts"]
    general_pat_mat_lin = mat["general_pat_mat_lin"]
    general_pat_mat_tofts = mat["general_pat_mat_tofts"]
    controls_avg_regions = (
        (mat_lin_control - mat_lin_control.mean(axis=0)) / mat_lin_control.std(axis=0)
    ).mean()
    mat_lin_avg_regions = (
        (mat_lin - mat_lin_control.mean(axis=0)) / mat_lin_control.std(axis=0)
    ).mean()
    focal_pat_mat_lin_avg_regions = (
        (focal_pat_mat_lin - mat_lin_control.mean(axis=0)) / mat_lin_control.std(axis=0)
    ).mean()
    general_pat_mat_lin_avg_regions = (
        (general_pat_mat_lin - mat_lin_control.mean(axis=0))
        / mat_lin_control.std(axis=0)
    ).mean()
    # mann whitney df['Epilepsy'], df['Controls']
    df["Epilepsy"] = pd.to_numeric(df["Epilepsy"], errors="coerce")
    df["Controls"] = pd.to_numeric(df["Controls"], errors="coerce")
    stats, p = mannwhitneyu(df["Epilepsy"].dropna(), df["Controls"].dropna())
    exponent = math.floor(math.log10(abs(p)))
    # mann whitney df['Epilepsy'], df['Controls']
    d1 = pd.to_numeric(mat_lin_avg_regions, errors="coerce")
    d2 = pd.to_numeric(controls_avg_regions, errors="coerce")
    stats, p = mannwhitneyu(d1.dropna(), d2.dropna())
    exponent2 = math.floor(math.log10(abs(p)))
    pass
    di = {}
    """Regression analysis of brain volume using the in all patients with epilepsy revealed that 3.98 ± 8.96% of voxels exhibited BBBD,
      while the mean standard deviation from the mean value of controls per region was 16.56 ± 23.16%. Statistical comparisons demonstrated significant differences BBBD%
    between groups (p<0.0001) as well as in the percentage of areas with BBBD (p<0.0001)."""
    di[
        "Patients with epilepsy"
    ] = f"""
    Regression analysis of brain volume using the in all patients with epilepsy revealed that {df['Epilepsy'].mean():.2f} ± {df['Epilepsy'].std():.2f}% of voxels exhibited BBBD,
    while the mean standard deviation from the controls for all regions was {mat_lin_avg_regions.mean():.2f} ± {mat_lin_avg_regions.std():.2f}%.
    Statistical comparisons demonstrated significant differences BBBD% between groups (p<10^{exponent}) as well as in the mean standard deviation from the controls for all regions (p<10^{exponent2}).
    """.replace(
        "\n", " "
    )
    # mann whitney df['Focal Epilepsy'], df['Controls']
    df["Focal Epilepsy"] = pd.to_numeric(df["Focal Epilepsy"], errors="coerce")
    df["Controls"] = pd.to_numeric(df["Controls"], errors="coerce")
    stats, p = mannwhitneyu(df["Focal Epilepsy"].dropna(), df["Controls"].dropna())
    exponent = math.floor(math.log10(abs(p)))
    # mann whitney df['Epilepsy'], df['Controls']
    d1 = pd.to_numeric(focal_pat_mat_lin_avg_regions, errors="coerce")
    d2 = pd.to_numeric(controls_avg_regions, errors="coerce")
    stats, p = mannwhitneyu(d1.dropna(), d2.dropna())
    exponent2 = math.floor(math.log10(abs(p)))
    # mann whitney df['Generalized Epilepsy'], df['Controls']
    df["Generalized Epilepsy"] = pd.to_numeric(
        df["Generalized Epilepsy"], errors="coerce"
    )
    df["Controls"] = pd.to_numeric(df["Controls"], errors="coerce")
    stats, p = mannwhitneyu(
        df["Generalized Epilepsy"].dropna(), df["Controls"].dropna()
    )
    exponent3 = math.floor(math.log10(abs(p)))
    # mann whitney df['Epilepsy'], df['Controls']
    d1 = pd.to_numeric(general_pat_mat_lin_avg_regions, errors="coerce")
    d2 = pd.to_numeric(controls_avg_regions, errors="coerce")
    stats, p = mannwhitneyu(d1.dropna(), d2.dropna())
    exponent4 = math.floor(math.log10(abs(p)))
    di[
        "focal_generalized"
    ] = f"""
    Regression analysis of brain volume using the in focal epilepsy revealed that {df['Focal Epilepsy'].mean():.2f} ± {df['Focal Epilepsy'].std():.2f}% of voxels exhibited BBBD,
    while the mean standard deviation from the controls for all regions was {focal_pat_mat_lin_avg_regions.mean():.2f} ± {focal_pat_mat_lin_avg_regions.std():.2f}%.
    For generalized epilepsy, the regression analysis revealed that {df['Generalized Epilepsy'].mean():.2f} ± {df['Generalized Epilepsy'].std():.2f}% of voxels exhibited BBBD,
    while the mean standard deviation from the controls for all regions was {general_pat_mat_lin_avg_regions.mean():.2f} ± {general_pat_mat_lin_avg_regions.std():.2f}%.
    Statistical comparisons demonstrated significant differences in focal epilepsy compares to controls (p<10^{exponent}) as well as in the mean standard deviation from the controls for all regions (p<10^{exponent2}).
    Statistical comparisons demonstrated significant differences in Generalized epilepsy compares to controls (p<10^{exponent3}) as well as in the mean standard deviation from the controls for all regions (p<10^{exponent4}).
    """.replace(
        "\n", " "
    )


def get_126_areas(clinical_data_df, result_mat_lin_age):
    df_126 = pd.DataFrame()
    codes = clinical_data_df["code"]
    # remove nan from codes
    codes = codes.dropna()
    # run over result_mat_lin_age
    for index, row in result_mat_lin_age.iterrows():
        if "'ID'" in row:
            # get ID if it is in codes
            Id = row["'ID'"].strip("'")
        else:
            Id = row["ID"]
        try:
            # find index in codes that id is contained in
            index_in_codes = codes[[i for i, x in enumerate(codes) if x in Id][0]]
            # change result_mat_lin_age row["'ID'"] to index_in_codes
            row["'ID'"] = index_in_codes
            result_mat_lin_age.loc[index, "'ID'"] = index_in_codes
            # concat row to df_126
            df_126 = pd.concat([df_126, pd.DataFrame(row).T], ignore_index=True)
        except:
            continue
    # strip "'" from df_126 columns
    df_126.columns = df_126.columns.str.strip("'")
    return df_126


def get_mat():
    files_location, excel_file, controls_file, clinical_data = get_folders()

    clinical_data_df = pd.read_excel(clinical_data, sheet_name="All")
    # until row 44
    clinical_data_df = clinical_data_df.iloc[:44]
    mat = {}
    # read sheet 128_Reagions_Linear
    result_mat_lin_age = pd.read_excel(excel_file, sheet_name="126_Regions_Linear")
    mat["result_mat_lin_age_control"] = pd.read_excel(
        controls_file, sheet_name="126_Regions_Linear"
    ).drop(columns=["'ID'"])
    df_126 = get_126_areas(clinical_data_df, result_mat_lin_age)
    # remove rows from clinical_data_df where code is not in result_mat_lin_age
    clinical_data_df = clinical_data_df[
        clinical_data_df["code"].isin(df_126["ID"])
    ].reset_index(drop=True)
    mat["result_mat_lin_age"] = df_126.drop(columns=["ID"])
    # mat['focal_pat_mat_lin'] = result_mat_lin_age where clinical_data_df['Focal/General'] == 'F'
    mat["focal_pat_mat_lin"] = df_126[clinical_data_df["Focal/General"] == "F"].drop(
        columns=["ID"]
    )
    mat["general_pat_mat_lin"] = df_126[clinical_data_df["Focal/General"] == "G"].drop(
        columns=["ID"]
    )
    mat["result_mat_tofts_age_control"] = mat["result_mat_lin_age_control"]
    mat["result_mat_tofts_age"] = mat["result_mat_lin_age"]
    mat["focal_pat_mat_tofts"] = mat["focal_pat_mat_lin"]
    mat["general_pat_mat_tofts"] = mat["general_pat_mat_lin"]
    mat["areas_names"] = [
        x.strip("'") for x in list(mat["result_mat_lin_age"].columns.values)
    ]
    # f_areas = areas in mat["areas_names"]  which contain "frontal"
    f_areas_indexes = [i for i, x in enumerate(mat["areas_names"]) if "frontal" in x]
    t_areas_indexes = [i for i, x in enumerate(mat["areas_names"]) if "temporal" in x]
    mat["fronal_epilepsy_f_areas_lin"] = df_126[
        clinical_data_df["Focal Type"].str.contains("F", na=False)
    ].drop(columns=["ID"])
    mat["fronal_epilepsy_f_areas_tofts"] = mat["fronal_epilepsy_f_areas_lin"]
    mat["rest_epilepsy_f_areas_lin"] = df_126[
        ~clinical_data_df["Focal Type"].str.contains("F", na=False)
    ].drop(columns=["ID"])
    mat["rest_epilepsy_f_areas_tofts"] = mat["rest_epilepsy_f_areas_lin"]
    mat["controls_f_areas_lin"] = mat["result_mat_lin_age_control"]
    mat["controls_f_areas_tofts"] = mat["controls_f_areas_lin"]
    mat["temporal_epilepsy_t_areas_lin"] = df_126[
        clinical_data_df["Focal Type"].str.contains("T", na=False)
    ].drop(columns=["ID"])
    mat["temporal_epilepsy_t_areas_tofts"] = mat["temporal_epilepsy_t_areas_lin"]
    mat["rest_epilepsy_t_areas_lin"] = df_126[
        ~clinical_data_df["Focal Type"].str.contains("T", na=False)
    ].drop(columns=["ID"])
    mat["rest_epilepsy_t_areas_tofts"] = mat["rest_epilepsy_t_areas_lin"]
    mat["controls_t_areas_lin"] = mat["result_mat_lin_age_control"]
    mat["controls_t_areas_lin"].columns = mat["controls_t_areas_lin"].columns.str.strip(
        "'"
    )
    mat["controls_t_areas_tofts"] = mat["controls_t_areas_lin"]
    # create df
    return mat
    # print codes which are in clinical_data_df but not in clinical_data_df2
    print(
        clinical_data_df[
            clinical_data_df["code"].isin(clinical_data_df2["code"]) == False
        ]["code"]
    )


# result_mat_lin_age read sheet
# mat_lin_control = mat['result_mat_lin_age_control']
# mat_tofts_control = mat['result_mat_tofts_age_control']
# mat_lin = mat['result_mat_lin_age']
# mat_tofts = mat['result_mat_tofts_age']
# focal_pat_mat_lin = mat['focal_pat_mat_lin']
# focal_pat_mat_tofts = mat['focal_pat_mat_tofts']
# general_pat_mat_lin = mat['general_pat_mat_lin']
# general_pat_mat_tofts = mat['general_pat_mat_tofts']
# "fronal_epilepsy_f_areas_lin": {},
# "fronal_epilepsy_f_areas_tofts": {},
# "rest_epilepsy_f_areas_lin": {},
# "rest_epilepsy_f_areas_tofts": {},
# "controls_f_areas_lin": {},
# "controls_f_areas_tofts": {},
# "temporal_epilepsy_t_areas_lin": {},
# "temporal_epilepsy_t_areas_tofts": {},
# "rest_epilepsy_t_areas_lin": {},
# "rest_epilepsy_t_areas_tofts": {},
# "controls_t_areas_lin": {},
# "controls_t_areas_tofts": {},
import os


def renmaes():
    path = r"D:\Dropbox (BGU DAL BBB Group)\Nir_Epilepsy_Controls\Epilepsy\DICOMS\1489752_DICOM"
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    print(folders)
    # path2 = r"D:\Dropbox (BGU DAL BBB Group)\Nir_Epilepsy_Controls\Epilepsy\DICOMS\1496759_DICOM"
    path2 = r"D:\Dropbox (BGU DAL BBB Group)\Nir_Epilepsy_Controls\Epilepsy\DICOMS\1493706_DICOM"
    folders2 = [f for f in os.listdir(path2) if os.path.isdir(os.path.join(path2, f))]
    # run over folders2
    for folder in folders2:
        if any(folder in f for f in folders):
            folder_from_folders = [f for f in folders if folder in f][0]
            # rename folder to folder_from_folders
            os.rename(
                os.path.join(path2, folder), os.path.join(path2, folder_from_folders)
            )
            print("found")


def merge_xlsx():
    import pandas as pd
    import numpy as np
    import os
    import glob
    import re

    # Read excel files from D:\Dropbox (BGU DAL BBB Group)\Nir_Epilepsy_Controls\Epilepsy\Analyse\Excel_results
    path = r"C:\Nir\BBB\BBB\Excel not age gender match"
    all_files = glob.glob(os.path.join(path, "*.xlsx"))
    # read get the names of sheet from all_files[0]
    excel_file = pd.ExcelFile(all_files[0])
    # get the names of sheets
    sheet_names = excel_file.sheet_names
    excel_file_name = os.path.join(path, "merged.xlsx")
    # create empty excel file
    pd.DataFrame().to_excel(excel_file_name)
    for i in range(len(sheet_names)):
        sheet = re.sub(r"\s+", "_", sheet_names[i])
        # run over sheet_names and merge them
        df_from_each_file = (pd.read_excel(f, sheet_name=sheet) for f in all_files)
        df_merged = pd.concat(df_from_each_file, ignore_index=True)
        # df_merged.to_excel("merged.xlsx") in path
        with pd.ExcelWriter(excel_file_name, engine="openpyxl", mode="a") as writer:
            df_merged.to_excel(writer, sheet_name=sheet, index=False)


if __name__ == "__main__":
    # merge_xlsx()
    results_paper_dyn()
    get_results()
    get_mat()

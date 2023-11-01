import scipy.io as sio
import numpy as np
import pandas as pd
import xlsxwriter
import scipy.stats as stats
from scipy.stats import mannwhitneyu
import plots_bars
import math


def divide(a, b):
    result = a / b
    if math.isnan(result):
        return 0
    elif math.isinf(result):
        return a
    else:
        return result


def side_ratio(left, right):
    arr = []
    # run over left side
    for index, left_side in enumerate(left):
        right_side = right[index]
        if left_side > right_side:
            arr.append(divide(left_side, right_side))
        else:
            arr.append(divide(right_side, left_side))
    return arr


# file Focal_Temporal.mat open
# load Focal_Temporal.mat
# mat = sio.loadmat("Focal_Temporal.mat")
import excel_all_pat_to__dict

mat = excel_all_pat_to__dict.get_mat()
# areas_names = [area[0][0] for area in mat["areas_names"]]
areas_names = mat["areas_names"]
fronal_epilepsy_f_areas_lin = pd.DataFrame()
fronal_epilepsy_f_areas_tofts = pd.DataFrame()
rest_epilepsy_f_areas_lin = pd.DataFrame()
rest_epilepsy_f_areas_tofts = pd.DataFrame()
controls_f_areas_lin = pd.DataFrame()
controls_f_areas_tofts = pd.DataFrame()

temporal_epilepsy_t_areas_lin = pd.DataFrame()
temporal_epilepsy_t_areas_tofts = pd.DataFrame()
rest_epilepsy_t_areas_lin = pd.DataFrame()
rest_epilepsy_t_areas_tofts = pd.DataFrame()
controls_t_areas_lin = pd.DataFrame()
controls_t_areas_tofts = pd.DataFrame()
di = {
    "fronal_epilepsy_f_areas_lin": {},
    "fronal_epilepsy_f_areas_tofts": {},
    "rest_epilepsy_f_areas_lin": {},
    "rest_epilepsy_f_areas_tofts": {},
    "controls_f_areas_lin": {},
    "controls_f_areas_tofts": {},
    "temporal_epilepsy_t_areas_lin": {},
    "temporal_epilepsy_t_areas_tofts": {},
    "rest_epilepsy_t_areas_lin": {},
    "rest_epilepsy_t_areas_tofts": {},
    "controls_t_areas_lin": {},
    "controls_t_areas_tofts": {},
}
max_length = max(
    len(mat["fronal_epilepsy_f_areas_lin"].iloc[:, 0]),
    len(mat["fronal_epilepsy_f_areas_tofts"].iloc[:, 0]),
    len(mat["rest_epilepsy_f_areas_lin"].iloc[:, 0]),  # [0, :
    len(mat["rest_epilepsy_f_areas_tofts"].iloc[:, 0]),  # [0, :
    len(mat["controls_f_areas_lin"].iloc[:, 0]),
    len(mat["controls_f_areas_tofts"].iloc[:, 0]),
)
di_per_costumer = {
    "fronal_epilepsy_f_areas_lin": np.zeros(max_length),
    "fronal_epilepsy_f_areas_tofts": np.zeros(max_length),
    "rest_epilepsy_f_areas_lin": np.zeros(max_length),
    "rest_epilepsy_f_areas_tofts": np.zeros(max_length),
    "controls_f_areas_lin": np.zeros(max_length),
    "controls_f_areas_tofts": np.zeros(max_length),
    "temporal_epilepsy_t_areas_lin": np.zeros(max_length),
    "temporal_epilepsy_t_areas_tofts": np.zeros(max_length),
    "rest_epilepsy_t_areas_lin": np.zeros(max_length),
    "rest_epilepsy_t_areas_tofts": np.zeros(max_length),
    "controls_t_areas_lin": np.zeros(max_length),
    "controls_t_areas_tofts": np.zeros(max_length),
    "temporal_epilepsy_f_areas_lin": np.zeros(max_length),
    "fronal_epilepsy_t_areas_lin": np.zeros(max_length),
}
dict_frontal_epilepsy = pd.DataFrame(columns=["lin", "tofts"])
dict_temporal_epilepsy = pd.DataFrame(columns=["lin", "tofts"])
# run over
for i, area in enumerate(areas_names):
    # if frontal in name
    if "frontal" in area:
        # add to fronal_epilepsy_f_areas
        fronal_epilepsy_f_areas_lin[area] = mat["fronal_epilepsy_f_areas_lin"].iloc[
            :, i
        ]
        fronal_epilepsy_f_areas_tofts[area] = mat["fronal_epilepsy_f_areas_tofts"].iloc[
            :, i
        ]
        rest_epilepsy_f_areas_lin[area] = mat["rest_epilepsy_f_areas_lin"].iloc[:, i]
        rest_epilepsy_f_areas_tofts[area] = mat["rest_epilepsy_f_areas_tofts"].iloc[
            :, i
        ]
        controls_f_areas_lin[area] = mat["controls_f_areas_lin"].iloc[:, i]
        controls_f_areas_tofts[area] = mat["controls_f_areas_tofts"].iloc[:, i]
        di["fronal_epilepsy_f_areas_lin"][area] = sum(
            2
            <= (fronal_epilepsy_f_areas_lin[area] - controls_f_areas_lin[area].mean())
            / controls_f_areas_lin[area].std()
        ) / len(fronal_epilepsy_f_areas_lin[area])
        di["fronal_epilepsy_f_areas_tofts"][area] = sum(
            2
            <= (
                fronal_epilepsy_f_areas_tofts[area]
                - controls_f_areas_tofts[area].mean()
            )
            / controls_f_areas_tofts[area].std()
        ) / len(fronal_epilepsy_f_areas_tofts[area])
        di["rest_epilepsy_f_areas_lin"][area] = sum(
            2
            <= (rest_epilepsy_f_areas_lin[area] - controls_f_areas_lin[area].mean())
            / controls_f_areas_lin[area].std()
        ) / len(rest_epilepsy_f_areas_lin[area])
        di["rest_epilepsy_f_areas_tofts"][area] = sum(
            2
            <= (rest_epilepsy_f_areas_tofts[area] - controls_f_areas_tofts[area].mean())
            / controls_f_areas_tofts[area].std()
        ) / len(rest_epilepsy_f_areas_tofts[area])
        di["controls_f_areas_lin"][area] = sum(
            2
            <= (controls_f_areas_lin[area] - controls_f_areas_lin[area].mean())
            / controls_f_areas_lin[area].std()
        ) / len(controls_f_areas_lin[area])
        di["controls_f_areas_tofts"][area] = sum(
            2
            <= (controls_f_areas_tofts[area] - controls_f_areas_tofts[area].mean())
            / controls_f_areas_tofts[area].std()
        ) / len(controls_f_areas_tofts[area])
        di["temporal_epilepsy_f_areas_lin"] = sum(
            2
            <= (
                mat["rest_epilepsy_t_areas_lin"].iloc[:, i]
                - controls_f_areas_lin[area].mean()
            )
            / controls_f_areas_lin[area].std()
        ) / len(mat["rest_epilepsy_t_areas_lin"].iloc[:, i])
        # run over rows in fronal_epilepsy_f_areas_lin
        for j, row in enumerate(mat["rest_epilepsy_f_areas_lin"].iloc[:, i]):
            di_per_costumer["fronal_epilepsy_f_areas_lin"][j] += (
                2
                <= mat["rest_epilepsy_f_areas_lin"].iloc[j, i]
                - controls_f_areas_lin[area].mean() / controls_f_areas_lin[area].std()
            )
            di_per_costumer["fronal_epilepsy_f_areas_tofts"][j] += (
                2
                <= mat["rest_epilepsy_t_areas_lin"].iloc[j, i]
                - controls_f_areas_tofts[area].mean()
                / controls_f_areas_tofts[area].std()
            )
        for j, row in enumerate(mat["rest_epilepsy_f_areas_lin"].iloc[:, i]):
            di_per_costumer["rest_epilepsy_f_areas_lin"][j] += (
                2
                <= mat["rest_epilepsy_f_areas_lin"].iloc[j, i]
                - controls_f_areas_lin[area].mean() / controls_f_areas_lin[area].std()
            )
            di_per_costumer["rest_epilepsy_f_areas_tofts"][j] += (
                2
                <= mat["mat_rest_f_tofts"].iloc[j, i]
                - controls_f_areas_tofts[area].mean()
                / controls_f_areas_tofts[area].std()
            )
        for j, row in enumerate(mat["result_mat_lin_age_control"][:, i]):
            di_per_costumer["controls_f_areas_lin"][j] += (
                2
                <= mat["result_mat_lin_age_control"].iloc[j, i]
                - controls_f_areas_lin[area].mean() / controls_f_areas_lin[area].std()
            )
            di_per_costumer["controls_f_areas_tofts"][j] += (
                2
                <= mat["result_mat_tofts_age_control"].iloc[j, i]
                - controls_f_areas_tofts[area].mean()
                / controls_f_areas_tofts[area].std()
            )
        for j, row in enumerate(mat["rest_epilepsy_t_areas_lin"].iloc[:, i]):
            di_per_costumer["temporal_epilepsy_f_areas_lin"][j] += (
                2
                <= mat["rest_epilepsy_t_areas_lin"].iloc[j, i]
                - controls_f_areas_lin[area].mean() / controls_f_areas_lin[area].std()
            )
        # mann whitney u test p values
        dict_frontal_epilepsy.loc[area, "lin"] = stats.mannwhitneyu(
            fronal_epilepsy_f_areas_lin[area].dropna(),
            rest_epilepsy_f_areas_lin[area].dropna(),
        )[1]
        dict_frontal_epilepsy.loc[area, "tofts"] = stats.mannwhitneyu(
            fronal_epilepsy_f_areas_tofts[area].dropna(),
            rest_epilepsy_f_areas_tofts[area].dropna(),
        )[1]
    if "temporal" in area:
        temporal_epilepsy_t_areas_lin[area] = mat["temporal_epilepsy_t_areas_lin"].iloc[
            :, i
        ]
        temporal_epilepsy_t_areas_tofts[area] = mat[
            "temporal_epilepsy_t_areas_tofts"
        ].iloc[:, i]
        rest_epilepsy_t_areas_lin[area] = mat["rest_epilepsy_t_areas_lin"].iloc[:, i]
        rest_epilepsy_t_areas_tofts[area] = mat["rest_epilepsy_t_areas_tofts"].iloc[
            :, i
        ]
        controls_t_areas_lin[area] = mat["controls_t_areas_lin"].iloc[:, i]
        controls_t_areas_tofts[area] = mat["controls_t_areas_tofts"].iloc[:, i]
        di["temporal_epilepsy_t_areas_lin"][area] = sum(
            2
            <= (temporal_epilepsy_t_areas_lin[area] - controls_t_areas_lin[area].mean())
            / controls_t_areas_lin[area].std()
        ) / len(temporal_epilepsy_t_areas_lin[area])
        di["temporal_epilepsy_t_areas_tofts"][area] = sum(
            2
            <= (
                temporal_epilepsy_t_areas_tofts[area]
                - controls_t_areas_tofts[area].mean()
            )
            / controls_t_areas_tofts[area].std()
        ) / len(temporal_epilepsy_t_areas_tofts[area])
        di["rest_epilepsy_t_areas_lin"][area] = sum(
            2
            <= (rest_epilepsy_t_areas_lin[area] - controls_t_areas_lin[area].mean())
            / controls_t_areas_lin[area].std()
        ) / len(rest_epilepsy_t_areas_lin[area])
        di["rest_epilepsy_t_areas_tofts"][area] = sum(
            2
            <= (rest_epilepsy_t_areas_tofts[area] - controls_t_areas_tofts[area].mean())
            / controls_t_areas_tofts[area].std()
        ) / len(rest_epilepsy_t_areas_tofts[area])
        di["controls_t_areas_lin"][area] = sum(
            2
            <= (controls_t_areas_lin[area] - controls_t_areas_lin[area].mean())
            / controls_t_areas_lin[area].std()
        ) / len(controls_t_areas_lin[area])
        di["controls_t_areas_tofts"][area] = sum(
            2
            <= (controls_t_areas_tofts[area] - controls_t_areas_tofts[area].mean())
            / controls_t_areas_tofts[area].std()
        ) / len(controls_t_areas_tofts[area])
        for j, row in enumerate(mat["rest_epilepsy_t_areas_lin"].iloc[:, i]):
            di_per_costumer["temporal_epilepsy_t_areas_lin"][j] += (
                2
                <= mat["rest_epilepsy_t_areas_lin"].iloc[j, i]
                - controls_t_areas_lin[area].mean() / controls_t_areas_lin[area].std()
            )
            di_per_costumer["temporal_epilepsy_t_areas_tofts"][j] += (
                2
                <= mat["rest_epilepsy_t_areas_lin"].iloc[j, i]
                - controls_t_areas_tofts[area].mean()
                / controls_t_areas_tofts[area].std()
            )
        for j, row in enumerate(mat["rest_epilepsy_t_areas_lin"].iloc[:, i]):
            di_per_costumer["rest_epilepsy_t_areas_lin"][j] += (
                2
                <= mat["rest_epilepsy_t_areas_lin"].iloc[j, i]
                - controls_t_areas_lin[area].mean() / controls_t_areas_lin[area].std()
            )
            di_per_costumer["rest_epilepsy_t_areas_tofts"][j] += (
                2
                <= mat["rest_epilepsy_t_areas_lin"].iloc[j, i]
                - controls_t_areas_tofts[area].mean()
                / controls_t_areas_tofts[area].std()
            )
        for j, row in enumerate(mat["result_mat_lin_age_control"].iloc[:, i]):
            di_per_costumer["controls_t_areas_lin"][j] += (
                2
                <= mat["result_mat_lin_age_control"].iloc[j, i]
                - controls_t_areas_lin[area].mean() / controls_t_areas_lin[area].std()
            )
            di_per_costumer["controls_t_areas_tofts"][j] += (
                2
                <= mat["result_mat_tofts_age_control"].iloc[j, i]
                - controls_t_areas_tofts[area].mean()
                / controls_t_areas_tofts[area].std()
            )
        for j, row in enumerate(mat["rest_epilepsy_f_areas_lin"].iloc[:, i]):
            di_per_costumer["fronal_epilepsy_t_areas_lin"][j] += (
                2
                <= mat["rest_epilepsy_f_areas_lin"].iloc[j, i]
                - controls_t_areas_lin[area].mean() / controls_t_areas_lin[area].std()
            )
        # mann whitney u test p values
        dict_temporal_epilepsy.loc[area, "lin"] = stats.mannwhitneyu(
            temporal_epilepsy_t_areas_lin[area].dropna(),
            rest_epilepsy_t_areas_lin[area].dropna(),
        )[1]
        dict_temporal_epilepsy.loc[area, "tofts"] = stats.mannwhitneyu(
            temporal_epilepsy_t_areas_tofts[area].dropna(),
            rest_epilepsy_t_areas_tofts[area].dropna(),
        )[1]
di["fronal_epilepsy_f_areas_lin_vals"] = [
    x for x in di["fronal_epilepsy_f_areas_lin"].values() if not math.isnan(x)
]
di["fronal_epilepsy_f_areas_tofts_vals"] = [
    x for x in di["fronal_epilepsy_f_areas_tofts"].values() if not math.isnan(x)
]
di["rest_epilepsy_f_areas_lin_vals"] = [
    x for x in di["rest_epilepsy_f_areas_lin"].values() if not math.isnan(x)
]
di["rest_epilepsy_f_areas_tofts_vals"] = [
    x for x in di["rest_epilepsy_f_areas_tofts"].values() if not math.isnan(x)
]
di["controls_f_areas_lin_vals"] = [
    x for x in di["controls_f_areas_lin"].values() if not math.isnan(x)
]
di["controls_f_areas_tofts_vals"] = [
    x for x in di["controls_f_areas_tofts"].values() if not math.isnan(x)
]
data = {
    "frontal_epilepsy_f_areas_lin_vals": di["fronal_epilepsy_f_areas_lin_vals"],
    "frontal_epilepsy_f_areas_tofts_vals": di["fronal_epilepsy_f_areas_tofts_vals"],
    "rest_epilepsy_f_areas_lin_vals": di["rest_epilepsy_f_areas_lin_vals"],
    "rest_epilepsy_f_areas_tofts_vals": di["rest_epilepsy_f_areas_tofts_vals"],
    "controls_f_areas_lin_vals": di["controls_f_areas_lin_vals"],
    "controls_f_areas_tofts_vals": di["controls_f_areas_tofts_vals"],
}
df_frontal = pd.DataFrame(data)
data = {
    "fronal_epilepsy_f_areas_lin": di_per_costumer["fronal_epilepsy_f_areas_lin"]
    / fronal_epilepsy_f_areas_lin.shape[1]
    * 100,
    "rest_epilepsy_f_areas_lin": di_per_costumer["rest_epilepsy_f_areas_lin"]
    / rest_epilepsy_f_areas_lin.shape[1]
    * 100,
    "controls_f_areas_lin": di_per_costumer["controls_f_areas_lin"]
    / controls_f_areas_lin.shape[1]
    * 100,
    "fronal_epilepsy_f_areas_tofts": di_per_costumer["fronal_epilepsy_f_areas_tofts"]
    / fronal_epilepsy_f_areas_tofts.shape[1]
    * 100,
    "rest_epilepsy_f_areas_tofts": di_per_costumer["rest_epilepsy_f_areas_tofts"]
    / rest_epilepsy_f_areas_tofts.shape[1]
    * 100,
    "controls_f_areas_tofts": di_per_costumer["controls_f_areas_tofts"]
    / controls_f_areas_tofts.shape[1]
    * 100,
    "temporal_epilepsy_f_areas_lin": di_per_costumer["temporal_epilepsy_f_areas_lin"]
    / controls_f_areas_tofts.shape[1]
    * 100,
}
# Create the DataFrame
frontal_per_costumer = pd.DataFrame(data)

di["temporal_epilepsy_t_areas_tofts_vals"] = [
    x for x in di["temporal_epilepsy_t_areas_tofts"].values() if not math.isnan(x)
]
di["temporal_epilepsy_t_areas_lin_vals"] = [
    x for x in di["temporal_epilepsy_t_areas_lin"].values() if not math.isnan(x)
]
di["rest_epilepsy_t_areas_lin_vals"] = [
    x for x in di["rest_epilepsy_t_areas_lin"].values() if not math.isnan(x)
]
di["rest_epilepsy_t_areas_tofts_vals"] = [
    x for x in di["rest_epilepsy_t_areas_tofts"].values() if not math.isnan(x)
]
di["controls_t_areas_lin_vals"] = [
    x for x in di["controls_t_areas_lin"].values() if not math.isnan(x)
]
di["controls_t_areas_tofts_vals"] = [
    x for x in di["controls_t_areas_tofts"].values() if not math.isnan(x)
]
data = {
    "temporal_epilepsy_t_areas_lin_vals": di["temporal_epilepsy_t_areas_lin_vals"],
    "temporal_epilepsy_t_areas_tofts_vals": di["temporal_epilepsy_t_areas_tofts_vals"],
    "rest_epilepsy_t_areas_lin_vals": di["rest_epilepsy_t_areas_lin_vals"],
    "rest_epilepsy_t_areas_tofts_vals": di["rest_epilepsy_t_areas_tofts_vals"],
    "controls_t_areas_lin_vals": di["controls_t_areas_lin_vals"],
    "controls_t_areas_tofts_vals": di["controls_t_areas_tofts_vals"],
}
df_temporal = pd.DataFrame(data)
data = {
    "temporal_epilepsy_t_areas_lin": di_per_costumer["temporal_epilepsy_t_areas_lin"]
    / temporal_epilepsy_t_areas_lin.shape[1]
    * 100,
    "rest_epilepsy_t_areas_lin": di_per_costumer["rest_epilepsy_t_areas_lin"]
    / rest_epilepsy_t_areas_lin.shape[1]
    * 100,
    "controls_t_areas_lin": di_per_costumer["controls_t_areas_lin"]
    / controls_t_areas_lin.shape[1]
    * 100,
    "temporal_epilepsy_t_areas_tofts": di_per_costumer[
        "temporal_epilepsy_t_areas_tofts"
    ]
    / temporal_epilepsy_t_areas_tofts.shape[1]
    * 100,
    "rest_epilepsy_t_areas_tofts": di_per_costumer["rest_epilepsy_t_areas_tofts"]
    / rest_epilepsy_t_areas_tofts.shape[1]
    * 100,
    "controls_t_areas_tofts": di_per_costumer["controls_t_areas_tofts"]
    / controls_t_areas_tofts.shape[1]
    * 100,
    "fronal_epilepsy_t_areas_lin": di_per_costumer["fronal_epilepsy_t_areas_lin"]
    / controls_t_areas_tofts.shape[1]
    * 100,
}
temporal_per_costumer = pd.DataFrame(data)


def post_processing(
    temporal_epilepsy_t_areas_lin, dict_temporal_epilepsy, writer, df_name
):
    temporal_epilepsy_t_areas_lin = temporal_epilepsy_t_areas_lin.T
    # get best 5 areas with losest p val
    indexes_sort_best = dict_temporal_epilepsy.sort_values(by=["lin"]).index
    # sort temporal_epilepsy_t_areas_lin by indexes_sort_best
    temporal_epilepsy_t_areas_lin = temporal_epilepsy_t_areas_lin.loc[indexes_sort_best]
    areas_names, dict_acronyzed = plots_bars.acronymize_list(indexes_sort_best)
    # change index to area names
    temporal_epilepsy_t_areas_lin.index = areas_names
    translation_table = str.maketrans("", "", "{'}")
    print(f"Regions legend: {str(dict_acronyzed).translate(translation_table)}")
    temporal_epilepsy_t_areas_lin.to_excel(writer, sheet_name=df_name)
    return temporal_epilepsy_t_areas_lin


# dict 1
dict_frontal = str(
    {k: v.replace("'", "") for k, v in enumerate(fronal_epilepsy_f_areas_lin)}
)
# remove ' from stings
print(dict_frontal.replace("'", ""))
dict_temporal = str(
    {k: v.replace("'", "") for k, v in enumerate(temporal_epilepsy_t_areas_lin)}
)
print(dict_temporal.replace("'", ""))
writer = pd.ExcelWriter("Frontal_temporal_areas.xlsx", engine="xlsxwriter")
frontal_per_costumer.to_excel(writer, sheet_name="Frontal")
temporal_per_costumer.to_excel(writer, sheet_name="Temporal")
post_processing(
    temporal_epilepsy_t_areas_lin,
    dict_temporal_epilepsy,
    writer,
    "TE Temporal Regions Slow",
)
post_processing(
    temporal_epilepsy_t_areas_tofts,
    dict_temporal_epilepsy,
    writer,
    "TE Temporal Regions Fast",
)
post_processing(
    rest_epilepsy_t_areas_lin,
    dict_temporal_epilepsy,
    writer,
    "Rest Temporal Regions Slow",
)
post_processing(
    rest_epilepsy_t_areas_tofts,
    dict_temporal_epilepsy,
    writer,
    "Rest  Temporal Regions Fast",
)
post_processing(
    fronal_epilepsy_f_areas_lin,
    dict_frontal_epilepsy,
    writer,
    "FE Frontal Regions Slow",
)
post_processing(
    fronal_epilepsy_f_areas_tofts,
    dict_frontal_epilepsy,
    writer,
    "FE Frontal Regions Fast",
)
post_processing(
    rest_epilepsy_f_areas_lin,
    dict_frontal_epilepsy,
    writer,
    "Rest Frontal Regions Slow",
)
post_processing(
    rest_epilepsy_f_areas_tofts,
    dict_frontal_epilepsy,
    writer,
    "Rest  Frontal Regions Fast",
)

writer.close()
print(
    f"""num frontal patients: {len(fronal_epilepsy_f_areas_lin)}
      \n num rest frontal patients: {len(rest_epilepsy_f_areas_lin)}
      \n num temporal patients: {len(temporal_epilepsy_t_areas_lin)}
      \n num rest temporal patients: {len(rest_epilepsy_t_areas_lin)}
      \n num controls patients: {len(controls_t_areas_lin)}"""
)
# temporal_epilepsy_t_areas_lin = temporal_epilepsy_t_areas_lin.T
# temporal_epilepsy_t_areas_tofts = temporal_epilepsy_t_areas_tofts.T
# rest_epilepsy_t_areas_lin = rest_epilepsy_t_areas_lin.T
# rest_epilepsy_t_areas_tofts = rest_epilepsy_t_areas_tofts.T
# controls_t_areas_lin = controls_t_areas_lin.T
# controls_t_areas_tofts = controls_t_areas_tofts.T
# fronal_epilepsy_f_areas_lin = fronal_epilepsy_f_areas_lin.T
# fronal_epilepsy_f_areas_tofts = fronal_epilepsy_f_areas_tofts.T
# rest_epilepsy_f_areas_lin = rest_epilepsy_f_areas_lin.T
# rest_epilepsy_f_areas_tofts = rest_epilepsy_f_areas_tofts.T
# controls_f_areas_lin = controls_f_areas_lin.T
# controls_f_areas_tofts = controls_f_areas_tofts.T
# # get best 5 areas with losest p val
# indexes_sort_best = dict_frontal_epilepsy.sort_values(by=['lin']).index
# # sort temporal_epilepsy_t_areas_lin by indexes_sort_best
# fronal_epilepsy_f_areas_lin = fronal_epilepsy_f_areas_lin.loc[indexes_sort_best]

# # export to one .excel all df give significant names to sheets
# fronal_epilepsy_f_areas_lin.to_excel(writer, sheet_name='Frontal_Epilepsy_F_Areas_Lin')
# fronal_epilepsy_f_areas_tofts.to_excel(writer, sheet_name='Frontal_Epilepsy_F_Areas_Tofts')
# rest_epilepsy_f_areas_lin.to_excel(writer, sheet_name='Rest_Epilepsy_F_Areas_Lin')
# rest_epilepsy_f_areas_tofts.to_excel(writer, sheet_name='Rest_Epilepsy_F_Areas_Tofts')
# temporal_epilepsy_t_areas_lin.to_excel(writer, sheet_name='Temporal_Epilepsy_T_Areas_Lin')
# temporal_epilepsy_t_areas_tofts.to_excel(writer, sheet_name='Temporal_Epilepsy_T_Areas_Tofts')
# rest_epilepsy_t_areas_lin.to_excel(writer, sheet_name='Rest_Epilepsy_T_Areas_Lin')
# rest_epilepsy_t_areas_tofts.to_excel(writer, sheet_name='Rest_Epilepsy_T_Areas_Tofts')
# controls_f_areas_lin.to_excel(writer, sheet_name='Controls_F_Areas_Lin')
# controls_f_areas_tofts.to_excel(writer, sheet_name='Controls_F_Areas_Tofts')
# controls_t_areas_lin.to_excel(writer, sheet_name='Controls_T_Areas_Lin')
# controls_t_areas_tofts.to_excel(writer, sheet_name='Controls_T_Areas_Tofts')
# writer.close()

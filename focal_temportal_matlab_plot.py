import scipy.io as sio
import numpy as np
import pandas as pd
import xlsxwriter
import scipy.stats as stats
from scipy.stats import mannwhitneyu
import plots_bars
import math
import matlab.engine


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
mat = sio.loadmat("Focal_Temporal.mat")
areas_names = [area[0][0] for area in mat["areas_names"]]
fronal_epilepsy_f_areas_lin = pd.DataFrame(columns=areas_names)
fronal_epilepsy_f_areas_tofts = pd.DataFrame(columns=areas_names)
rest_epilepsy_f_areas_lin = pd.DataFrame(columns=areas_names)
rest_epilepsy_f_areas_tofts = pd.DataFrame(columns=areas_names)
controls_f_areas_lin = pd.DataFrame(columns=areas_names)
controls_f_areas_tofts = pd.DataFrame(columns=areas_names)

temporal_epilepsy_t_areas_lin = pd.DataFrame(columns=areas_names)
temporal_epilepsy_t_areas_tofts = pd.DataFrame(columns=areas_names)
rest_epilepsy_t_areas_lin = pd.DataFrame(columns=areas_names)
rest_epilepsy_t_areas_tofts = pd.DataFrame(columns=areas_names)
controls_t_areas_lin = pd.DataFrame(columns=areas_names)
controls_t_areas_tofts = pd.DataFrame(columns=areas_names)
dict_frontal_epilepsy = pd.DataFrame(columns=["lin", "tofts"])
dict_temporal_epilepsy = pd.DataFrame(columns=["lin", "tofts"])
# run over
for i, area in enumerate(areas_names):
    # if frontal in name
    if "frontal" in area:
        # add to fronal_epilepsy_f_areas
        fronal_epilepsy_f_areas_lin[area] = mat["mat_front_rest_l"][:, i]
        fronal_epilepsy_f_areas_tofts[area] = mat["mat_front_rest_t"][:, i]
        rest_epilepsy_f_areas_lin[area] = mat["mat_rest_f_lin"][:, i]
        rest_epilepsy_f_areas_tofts[area] = mat["mat_rest_f_tofts"][:, i]
        controls_f_areas_lin[area] = mat["result_mat_lin_age_control"][:, i]
        controls_f_areas_tofts[area] = mat["result_mat_tofts_age_control"][:, i]
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
        temporal_epilepsy_t_areas_lin[area] = mat["mat_temporal_rest_l"][:, i]
        temporal_epilepsy_t_areas_tofts[area] = mat["mat_temporal_rest_t"][:, i]
        rest_epilepsy_t_areas_lin[area] = mat["mat_rest_t_lin"][:, i]
        rest_epilepsy_t_areas_tofts[area] = mat["mat_rest_t_tofts"][:, i]
        controls_t_areas_lin[area] = mat["result_mat_lin_age_control"][:, i]
        controls_t_areas_tofts[area] = mat["result_mat_tofts_age_control"][:, i]
        # mann whitney u test p values
        dict_temporal_epilepsy.loc[area, "lin"] = stats.mannwhitneyu(
            temporal_epilepsy_t_areas_lin[area].dropna(),
            rest_epilepsy_t_areas_lin[area].dropna(),
        )[1]
        dict_temporal_epilepsy.loc[area, "tofts"] = stats.mannwhitneyu(
            temporal_epilepsy_t_areas_tofts[area].dropna(),
            rest_epilepsy_t_areas_tofts[area].dropna(),
        )[1]


def post_processing(
    temporal_epilepsy_t_areas_lin, dict_temporal_epilepsy, writer, df_name
):
    temporal_epilepsy_t_areas_lin = temporal_epilepsy_t_areas_lin.fillna(0)
    # def matlbal_std_plot():
    eng = matlab.engine.start_matlab()
    eng.cd(r"C:\Nir\BBB\BBB_GUI\Nir", nargout=0)
    arr = eng.Plot_Regional_Lyna(
        [0, 1], temporal_epilepsy_t_areas_lin.mean(axis=1).tolist(), "a", "a"
    )
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

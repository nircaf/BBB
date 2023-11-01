import scipy.io as sio
import numpy as np
import pandas as pd
import xlsxwriter

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
mat = sio.loadmat("Focal_Temporal.mat")
areas_names = [area[0][0] for area in mat["areas_names"]]
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
# run over
for i, area in enumerate(areas_names):
    # change to string
    area = area.astype(str)
    # if Right in area continue
    if "Right" in area:
        continue
    elif "Left" in area:
        area_no_side = " ".join(area.split(" ")[1:])
        index_right_side = areas_names.index("Right " + area_no_side)
    # if frontal in name
    if "frontal" in area:
        # add to fronal_epilepsy_f_areas
        fronal_epilepsy_f_areas_lin[area] = side_ratio(
            mat["mat_front_rest_l"][:, i], mat["mat_front_rest_l"][:, index_right_side]
        )
        fronal_epilepsy_f_areas_tofts[area] = side_ratio(
            mat["mat_front_rest_t"][:, i], mat["mat_front_rest_t"][:, index_right_side]
        )
        rest_epilepsy_f_areas_lin[area] = side_ratio(
            mat["mat_rest_f_lin"][:, i], mat["mat_rest_f_lin"][:, index_right_side]
        )
        rest_epilepsy_f_areas_tofts[area] = side_ratio(
            mat["mat_rest_f_tofts"][:, i], mat["mat_rest_f_tofts"][:, index_right_side]
        )
        controls_f_areas_lin[area] = side_ratio(
            mat["result_mat_lin_age_control"][:, i],
            mat["result_mat_lin_age_control"][:, index_right_side],
        )
        controls_f_areas_tofts[area] = side_ratio(
            mat["result_mat_tofts_age_control"][:, i],
            mat["result_mat_tofts_age_control"][:, index_right_side],
        )
    if "temporal" in area:
        temporal_epilepsy_t_areas_lin[area] = side_ratio(
            mat["mat_temporal_rest_l"][:, i],
            mat["mat_temporal_rest_l"][:, index_right_side],
        )
        temporal_epilepsy_t_areas_tofts[area] = side_ratio(
            mat["mat_temporal_rest_t"][:, i],
            mat["mat_temporal_rest_t"][:, index_right_side],
        )
        rest_epilepsy_t_areas_lin[area] = side_ratio(
            mat["mat_rest_t_lin"][:, i], mat["mat_rest_t_lin"][:, index_right_side]
        )
        rest_epilepsy_t_areas_tofts[area] = side_ratio(
            mat["mat_rest_t_tofts"][:, i], mat["mat_rest_t_tofts"][:, index_right_side]
        )
        controls_t_areas_lin[area] = side_ratio(
            mat["result_mat_lin_age_control"][:, i],
            mat["result_mat_lin_age_control"][:, index_right_side],
        )
        controls_t_areas_tofts[area] = side_ratio(
            mat["result_mat_tofts_age_control"][:, i],
            mat["result_mat_tofts_age_control"][:, index_right_side],
        )
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
temporal_epilepsy_t_areas_lin = temporal_epilepsy_t_areas_lin.T
temporal_epilepsy_t_areas_tofts = temporal_epilepsy_t_areas_tofts.T
rest_epilepsy_t_areas_lin = rest_epilepsy_t_areas_lin.T
rest_epilepsy_t_areas_tofts = rest_epilepsy_t_areas_tofts.T
controls_t_areas_lin = controls_t_areas_lin.T
controls_t_areas_tofts = controls_t_areas_tofts.T
fronal_epilepsy_f_areas_lin = fronal_epilepsy_f_areas_lin.T
fronal_epilepsy_f_areas_tofts = fronal_epilepsy_f_areas_tofts.T
rest_epilepsy_f_areas_lin = rest_epilepsy_f_areas_lin.T
rest_epilepsy_f_areas_tofts = rest_epilepsy_f_areas_tofts.T
controls_f_areas_lin = controls_f_areas_lin.T
controls_f_areas_tofts = controls_f_areas_tofts.T
# export to one .excel all df give significant names to sheets
writer = pd.ExcelWriter("Frontal_temporal_areas.xlsx", engine="xlsxwriter")
fronal_epilepsy_f_areas_lin.to_excel(writer, sheet_name="Frontal_Epilepsy_F_Areas_Lin")
fronal_epilepsy_f_areas_tofts.to_excel(
    writer, sheet_name="Frontal_Epilepsy_F_Areas_Tofts"
)
rest_epilepsy_f_areas_lin.to_excel(writer, sheet_name="Rest_Epilepsy_F_Areas_Lin")
rest_epilepsy_f_areas_tofts.to_excel(writer, sheet_name="Rest_Epilepsy_F_Areas_Tofts")
temporal_epilepsy_t_areas_lin.to_excel(
    writer, sheet_name="Temporal_Epilepsy_T_Areas_Lin"
)
temporal_epilepsy_t_areas_tofts.to_excel(
    writer, sheet_name="Temporal_Epilepsy_T_Areas_Tofts"
)
rest_epilepsy_t_areas_lin.to_excel(writer, sheet_name="Rest_Epilepsy_T_Areas_Lin")
rest_epilepsy_t_areas_tofts.to_excel(writer, sheet_name="Rest_Epilepsy_T_Areas_Tofts")
controls_f_areas_lin.to_excel(writer, sheet_name="Controls_F_Areas_Lin")
controls_f_areas_tofts.to_excel(writer, sheet_name="Controls_F_Areas_Tofts")
controls_t_areas_lin.to_excel(writer, sheet_name="Controls_T_Areas_Lin")
controls_t_areas_tofts.to_excel(writer, sheet_name="Controls_T_Areas_Tofts")
writer.close()

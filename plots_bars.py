import pandas as pd
import os
import glob
import re
from scipy import stats

# read excel Epilepsy_clinical_data.xlsx sheet All
import numpy as np
import scipy.io
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
import math

# import scienceplots
import excel_all_pat_to__dict

# set sns
sns.set()
# plt.style.use('science')
# load epilepsy_data.mat
mat = excel_all_pat_to__dict.get_mat()
areas_names = mat["areas_names"]


# areas_names = [area[0][0] for area in areas_names]
# change area names to be the first letter of each word
def acronymize_list(areas_names):
    dict_acronyzed = {}
    acronym_list = []
    for item in areas_names:
        words = item.split()
        acronym = "".join(word[0].upper() for word in words)
        # if words[0] is Right or Left
        if "Right" in words[0] or "Left" in words[0]:
            # add space after first letter in acronym
            acronym = acronym[0] + "-" + acronym[1:]
        # check if acronym_list contains acronym
        while acronym in acronym_list:
            acronym = acronym + "2"
        dict_acronyzed[acronym] = item
        acronym_list.append(acronym)
    return acronym_list, dict_acronyzed


if __name__ == "__main__":
    areas_names, dict_acronyzed = acronymize_list(areas_names)
    dict_acronyzed_reversed = {v: k for k, v in dict_acronyzed.items()}
    mat_lin_control = mat["result_mat_lin_age_control"]
    mat_tofts_control = mat["result_mat_tofts_age_control"]
    mat_lin = mat["result_mat_lin_age"]
    mat_tofts = mat["result_mat_tofts_age"]
    focal_pat_mat_lin = mat["focal_pat_mat_lin"]
    focal_pat_mat_tofts = mat["focal_pat_mat_tofts"]
    general_pat_mat_lin = mat["general_pat_mat_lin"]
    general_pat_mat_tofts = mat["general_pat_mat_tofts"]
    # change the column names by dict_acronyzed_reversed
    mat_lin = mat_lin.rename(columns=dict_acronyzed_reversed)
    mat_tofts = mat_tofts.rename(columns=dict_acronyzed_reversed)
    mat_lin_control = mat_lin_control.rename(columns=dict_acronyzed_reversed)
    mat_tofts_control = mat_tofts_control.rename(columns=dict_acronyzed_reversed)
    focal_pat_mat_lin = focal_pat_mat_lin.rename(columns=dict_acronyzed_reversed)
    focal_pat_mat_tofts = focal_pat_mat_tofts.rename(columns=dict_acronyzed_reversed)
    general_pat_mat_lin = general_pat_mat_lin.rename(columns=dict_acronyzed_reversed)
    general_pat_mat_tofts = general_pat_mat_tofts.rename(
        columns=dict_acronyzed_reversed
    )
    df = pd.DataFrame(
        columns=[
            "lin",
            "tofts",
            "lin_sig",
            "tofts_sig",
            "sum_lin_tofts",
            "lin_focal",
            "lin_general",
            "tofts_focal",
            "tofts_general",
            "sig_lin_focal",
            "sig_lin_general",
            "sig_tofts_focal",
            "sig_tofts_general",
        ],
        index=areas_names,
    )
    # run over each area
    for i, area in enumerate(areas_names):
        pats_lin = mat_lin.iloc[:, i]
        pats_tofts = mat_tofts.iloc[:, i].dropna().tolist()
        controls_lin = mat_lin_control.iloc[:, i]
        controls_tofts = mat_tofts_control.iloc[:, i].dropna().tolist()
        controls_lin_mean = np.nanmean(controls_lin)
        controls_tofts_mean = np.nanmean(controls_tofts)
        controls_lin_std = np.nanstd(controls_lin)
        controls_tofts_std = np.nanstd(controls_tofts)
        pats_lin = pats_lin.dropna().reset_index(drop=True).tolist()
        controls_lin = controls_lin.dropna().reset_index(drop=True).tolist()
        if pats_lin == []:
            continue
        # get % of pats above 2 std from mean of controls
        # val_lin = (
        #     100
        #     * sum(((pats_lin - controls_lin_mean) / controls_lin_std) >= 2)
        #     / len(pats_lin)
        # )
        # number of stds above controls
        val_lin = (((pats_lin - controls_lin_mean) / controls_lin_std)).mean()
        df.loc[area, "lin"] = val_lin
        # mann whitney test
        df.loc[area, "lin_sig"] = stats.mannwhitneyu(pats_lin, controls_lin)[1]
        val_tofts = ((pats_tofts - controls_tofts_mean) / controls_tofts_std).mean()
        df.loc[area, "tofts"] = val_tofts
        df.loc[area, "tofts_sig"] = stats.mannwhitneyu(pats_tofts, controls_tofts)[1]
        df.loc[area, "sum_lin_tofts"] = val_lin + val_tofts
        df.loc[area, "sum_lin_tofts_sig"] = stats.mannwhitneyu(
            (pats_tofts + pats_lin), (controls_tofts + controls_lin)
        )[1]
        df.loc[area, "lin_focal"] = (
            (focal_pat_mat_lin.iloc[:, i] - controls_lin_mean) / controls_lin_std
        ).mean()
        df.loc[area, "lin_general"] = (
            (general_pat_mat_lin.iloc[:, i] - controls_lin_mean) / controls_lin_std
        ).mean()
        df.loc[area, "tofts_focal"] = (
            100
            * sum(
                (
                    (focal_pat_mat_tofts.iloc[:, i] - controls_tofts_mean)
                    / controls_tofts_std
                )
                >= 2
            )
            / len(focal_pat_mat_tofts)
        )
        df.loc[area, "tofts_general"] = (
            100
            * sum(
                (
                    (general_pat_mat_tofts.iloc[:, i] - controls_tofts_mean)
                    / controls_tofts_std
                )
                >= 2
            )
            / len(general_pat_mat_tofts)
        )
        df.loc[area, "sig_lin_focal"] = stats.mannwhitneyu(
            focal_pat_mat_lin.iloc[:, i].dropna().tolist(), controls_lin
        )[1]
        df.loc[area, "sig_lin_general"] = stats.mannwhitneyu(
            general_pat_mat_lin.iloc[:, i].dropna().tolist(), controls_lin
        )[1]
        df.loc[area, "sig_tofts_focal"] = stats.mannwhitneyu(
            focal_pat_mat_tofts.iloc[:, i].dropna().tolist(), controls_tofts
        )[1]
        df.loc[area, "sig_tofts_general"] = stats.mannwhitneyu(
            general_pat_mat_tofts.iloc[:, i].dropna().tolist(), controls_tofts
        )[1]

    # do Bonferroni  correction to lin_sig
    df["lin_sig"] = multipletests(df["lin_sig"], method="fdr_bh")[1]
    # do Bonferroni  correction to tofts_sig
    df["tofts_sig"] = multipletests(df["tofts_sig"], method="fdr_bh")[1]
    df["sig_lin_focal"] = multipletests(df["sig_lin_focal"], method="fdr_bh")[1]
    df["sig_lin_general"] = multipletests(df["sig_lin_general"], method="fdr_bh")[1]
    df["sig_tofts_focal"] = multipletests(df["sig_tofts_focal"], method="fdr_bh")[1]
    df["sig_tofts_general"] = multipletests(df["sig_tofts_general"], method="fdr_bh")[1]
    df["sum_lin_tofts_sig"] = multipletests(
        df["sum_lin_tofts_sig"], method="bonferroni"
    )[1]

    def plot_academic_bar(df, y, color, title, legend=False, indexes=None):
        font_size = 30
        font_name = {"fontname": "Times New Roman"}
        # set figure HD
        plt.figure(figsize=(12, 10))
        ax = plt.gca()
        df.plot(kind="bar", y=y, color=color, ax=plt.gca())
        if legend:
            plt.legend(legend, fontsize=font_size * 0.8, loc="upper right").set_visible(
                True
            )
        else:
            # legend off
            plt.legend().set_visible(False)
        # Get the current tick labels
        labels = ax.get_xticklabels()
        # set label indexes as yellow
        # for label in labels:
        #     txt_index = label.get_text()
        #     if txt_index in indexes:
        #         label.set_color('r')
        # backgound color white
        plt.gca().set_facecolor("w")
        ax.set_facecolor("w")
        # ylabel % of patients with BBBD
        plt.ylabel("% of patients with BBBD", fontsize=font_size, **font_name)
        # xlabel area
        # plt.xlabel('Region')
        # plt.title(title,fontsize=font_size,**font_name)
        # horizontal black grid lines
        plt.grid(color="k", linewidth=0.5, axis="y")
        # set gca text size 20
        plt.tick_params(labelsize=font_size)
        plt.savefig("figures/" + title + ".png", bbox_inches="tight")
        plt.show(block=False)

    def plot_academic_bar2(df, y, color, title, legend=False, indexes=None):
        font_size = 30
        font_name = {"fontname": "Times New Roman"}
        # set figure HD
        plt.figure(figsize=(12, 10))
        ax = plt.gca()
        # plt.bar
        # if df is dataframe
        df.plot(kind="bar", y=y, color=color, ax=plt.gca(), width=0.8)
        ax.set_xticks(range(0, df.shape[0], 20))
        ax.set_xticklabels(range(0, df.shape[0], 20))
        if legend:
            plt.legend(legend, fontsize=font_size * 0.8, loc="upper right").set_visible(
                True
            )
        else:
            # legend off
            plt.legend().set_visible(False)
        plt.gca().set_facecolor("w")
        ax.set_facecolor("w")
        # ylabel % of patients with BBBD
        plt.ylabel("Zscore", fontsize=font_size, **font_name)
        # yticks rotation 0
        plt.xticks(rotation=0)
        # tick fontsize
        plt.tick_params(labelsize=font_size*0.8)
        plt.xlabel("Region", fontsize=font_size, **font_name)
        plt.grid(color="k", linewidth=0.5, axis="y")
        plt.show(block=False)
        plt.savefig("figures/" + title + ".png", bbox_inches="tight")

    # sort lin descending
    df = df.sort_values(by="lin", ascending=False)
    inclusion = 0.00000000003
    # find lin_sig < 0.05
    df_lin = df.head(sum(df["lin_sig"] < inclusion))
    df_lin = df["lin"]
    # plot bar df lin in cyan

    # indexes in both df_t and df_lin
    indexes = df_lin.index
    print(f'Regions in both slow and fast: {", ".join(indexes.values)}')
    # plot lin
    plot_academic_bar2(
        df_lin,
        "lin",
        "red",
        "% of patients with slow BBBD per region",
        indexes=indexes,
    )
    print("Inclusion criteria p value < " + str(inclusion))
    # plot bar df tofts in magenta
    # plot_academic_bar(df_t,'tofts','magenta','% of patients with fast BBBD per region',indexes=indexes)
    # print('Inclusion criteria p value < '+str(inclusion2))
    # all indexes in df_t and df_lin
    # indexes = df_t.index.union(df_lin.index)
    indexes = df_lin.index
    dict_acronyzed_t = {k: v for k, v in dict_acronyzed.items() if k in indexes}
    # Create a translation table
    translation_table = str.maketrans("", "", "{'}")
    print(f"Regions legend: {str(dict_acronyzed_t).translate(translation_table)}")

    # dict dict_acronyzed of values in df_t and df_lin to text

    df["lin_focallin_general"] = df["lin_focal"] + df["lin_general"]
    inclusion = 0.000013
    df = df.sort_values(by="lin_focallin_general", ascending=False)
    df_lin_f = df[
        (df["sig_lin_focal"] < inclusion) & (df["sig_lin_general"] < inclusion)
    ]
    df_lin_f = df[["lin_focal", "lin_general"]]
    indexes = df_lin_f.index
    # plot bar df lin in cya
    plot_academic_bar2(
        df_lin_f,
        ["lin_general", "lin_focal"],
        ["#FF0000", "#800000"],
        "% of focal and generalized patients with slow BBBD per region",
        legend=["Focal", "Generalized"],
        indexes=indexes,
    )
    print("Inclusion criteria p value < " + str(inclusion))
    # plot bar df tofts in magenta
    # plot_academic_bar(df_t_f,['tofts_focal','tofts_general'],['magenta','darkmagenta']
    #                 ,'% of focal and generalized patients with fast BBBD per region',legend=['Focal','Generalized'],indexes=indexes)
    # print('Inclusion criteria p value < '+str(inclusion2))

    # all indexes in df_t and df_lin
    # indexes = df_t_f.index.union(df_lin_f.index)
    dict_acronyzed_t = {k: v for k, v in dict_acronyzed.items() if k in indexes}
    # Create a translation table
    translation_table = str.maketrans("", "", "{'}")
    print(f"Regions legend: {str(dict_acronyzed_t).translate(translation_table)}")

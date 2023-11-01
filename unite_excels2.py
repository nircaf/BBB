import glob
import pandas as pd
import numpy as np
import glob
import datetime

# load .csv files

# load csv Epilepsy losartan
mri_eeg_dates = pd.read_csv("Epilepsy losartan.csv")
# split code '_' to new column session
mri_eeg_dates[["initials", "Session"]] = mri_eeg_dates["code"].str.split(
    "_", expand=True
)
# session to int
mri_eeg_dates["Session"] = mri_eeg_dates["Session"].astype(int)
# split MRI_date '\n' on pre and post and astype datetime
mri_eeg_dates[["MRI_date_pre", "MRI_date_post"]] = mri_eeg_dates["MRI_date"].str.split(
    "\n", expand=True
)
mri_eeg_dates["MRI_date_pre"] = pd.to_datetime(
    mri_eeg_dates["MRI_date_pre"], format="%d.%m.%Y"
)
mri_eeg_dates["MRI_date_post"] = pd.to_datetime(
    mri_eeg_dates["MRI_date_post"], format="%d.%m.%Y"
)
# load Losartan_PSWE_analysis
losartan_pswe = pd.read_csv("Losartan_PSWE_analysis.csv")
# run over rows in losartan_pswe
for index, row in losartan_pswe.iterrows():
    mri_eeg_dates_row = mri_eeg_dates.loc[mri_eeg_dates["Session"] == row["Session"]]
    if not mri_eeg_dates_row.empty:
        mri_eeg_dates_row2 = mri_eeg_dates_row.iloc[0]
        date_eeg = row["Recording_date"]
        # convert to datetime dd.mm.YY
        date_eeg = datetime.datetime.strptime(date_eeg, "%d.%m.%y")
        # if date before MRI_date_pre 'pre' if date after MRI_date_post 'post' else 'middle'
        time_eeg = (
            "pre"
            if date_eeg < mri_eeg_dates_row2["MRI_date_pre"]
            else "post"
            if date_eeg > mri_eeg_dates_row2["MRI_date_post"]
            else "middle"
        )
        losartan_pswe.loc[index, "eeg_mri_time"] = time_eeg
# put row 'eeg_mri_time' first
cols = list(losartan_pswe)
cols.insert(0, cols.pop(cols.index("eeg_mri_time")))
losartan_pswe = losartan_pswe.loc[:, cols]
# save to csv
losartan_pswe.to_csv("Losartan_PSWE_analysis_eeg_mri_time.csv", index=False)

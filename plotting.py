import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import statannot


def barplot_annotate_brackets(
    num1,
    num2,
    data,
    center,
    height,
    yerr=None,
    dh=0.05,
    barh=0.05,
    fs=None,
    maxasterix=None,
):
    """
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ""
        p = 0.05

        while data < p:
            text += "*"
            p /= 10.0

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = "n. s."

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= ax_y1 - ax_y0
    barh *= ax_y1 - ax_y0

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y + barh, y + barh, y]
    mid = ((lx + rx) / 2, y + barh)

    plt.plot(barx, bary, c="black")

    kwargs = dict(ha="center", va="bottom")
    if fs is not None:
        kwargs["fontsize"] = fs

    plt.text(*mid, text, **kwargs)


def sig_bars(data, ax):
    # Initialise a list of combinations of groups that are significantly different
    significant_combinations = []
    # Check from the outside pairs of boxes inwards
    ls = list(range(1, len(data) + 1))
    combinations = [
        (ls[x], ls[x + y]) for y in reversed(ls) for x in range((len(ls) - y))
    ]
    for combination in combinations:
        data1 = data[combination[0] - 1]
        data2 = data[combination[1] - 1]
        # Significance
        U, p = scipy.stats.ttest_ind(data1, data2, alternative="two-sided")
        # if p < 0.05:
        significant_combinations.append([combination, p])
    # Get the y-axis limits
    bottom, top = ax.get_ylim()
    y_range = top - bottom
    # Significance bars
    for i, significant_combination in enumerate(significant_combinations):
        # Columns corresponding to the datasets of interest
        x1 = significant_combination[0][0]
        x2 = significant_combination[0][1]
        # What level is this bar among the bars above the plot?
        level = len(significant_combinations) - i
        # Plot the bar
        bar_height = (y_range * 0.07 * level) + top
        bar_tips = bar_height - (y_range * 0.02)
        plt.plot(
            [x1, x1, x2, x2], [bar_tips, bar_height, bar_height, bar_tips], lw=1, c="k"
        )
        # Significance level
        p = significant_combination[1]
        sig_symbol = convert_pvalue_to_asterisks(p)
        text_height = bar_height + (y_range * 0.01)
        plt.text(
            (x1 + x2) * 0.5, text_height, sig_symbol, ha="center", va="bottom", c="k"
        )


def annot_stat(star, x1, x2, y, h, col="k", ax=None):
    ax = plt.gca() if ax is None else ax
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    ax.text((x1 + x2) * 0.5, y + h, star, ha="center", va="bottom", color=col)


def read_excel(file_name):
    df = pd.read_excel(file_name)
    return df


def plot_xy(df, x, y):
    # use sns to plot x y
    sns.lineplot(df, x=x, y=y)


def plot_boxplot(df, x, y, ax):
    # use sns to plot boxplot with colors red and black
    sns.boxplot(data=df, x=x, y=y, ax=ax, palette=["k", "r"])
    # check for significance
    y_group_1 = df[df[x] == 1][y]
    y_group_0 = df[df[x] == 0][y]
    stat, pvalue = scipy.stats.ttest_ind(y_group_1, y_group_0)
    # find max of y_group_1 and y_group_2
    max_y = max(y_group_1.max(), y_group_0.max())
    # find max std of y_group_1 and y_group_2
    max_std = max(y_group_1.std(), y_group_0.std())
    annot_stat(convert_pvalue_to_asterisks(pvalue), 0, 1, max_y * 1.1, 5, ax=ax)
    # sig_bars([y_group_1, y_group_2],ax)
    # barplot_annotate_brackets(0, 1, convert_pvalue_to_asterisks(pvalue), [0, 1], [y_group_1.mean(), y_group_2.mean()],
    #                           yerr=[y_group_1.std(), y_group_2.std()])
    # show plot
    plt.show(block=False)


def convert_pvalue_to_asterisks(pvalue):
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return "ns"


def lin_tofts_boxplot(df, x, title):
    df = df.apply(lambda x: pd.to_numeric(x, errors="ignore"))
    # create 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2)
    # set title
    fig.suptitle(title)
    y = "BBBP_lin"
    plot_boxplot(df, x, y, ax1)
    y = "BBBP_tofts"
    plot_boxplot(df, x, y, ax2)


if "__main__" == __name__:
    # df = read_excel(r'E:\Dropbox (BGU DAL BBB Group)\Nir_Epilepsy_Controls\Epilepsy\Analyse\Epilepsy_clinical_data.xlsx').
    df = read_excel(r"C:\Users\ncafr\Downloads\Epilepsy_clinical_data.xlsx")
    # remove rows all nan
    df = df.dropna(how="all")
    x = "Lesion"
    df[x] = df[x].apply(lambda x: 0 if x != 0 else 1)
    # plot lin tofts
    lin_tofts_boxplot(df, x, "Lesion vs BBB%")
    df_g = df[df["Focal/General"] == "G"]
    # plot lin tofts
    lin_tofts_boxplot(df_g, x, "General epilepsy \n Lesion vs BBB%")
    df_f = df[df["Focal/General"] == "F"]
    lin_tofts_boxplot(df_f, x, "Focal epilepsy \n Lesion vs BBB%")
    print("done")

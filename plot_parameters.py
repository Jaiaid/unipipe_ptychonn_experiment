SYSTEM_NAME_LIST = ["pretrained_noipr", "pretrained", "multicontext", "unipipe", "unipipe_dp"]
SYSTEM_NAME_TO_LEGEND_DICT = {
    "pretrained_noipr": "Pretrained", "pretrained": "Pretrained-Comp.",
    "multicontext_nocoord": "MultiContext-No Coordination", "multicontext": "MultiContext",
    "unipipe": "Unipipe(static)", "unipipe_dp": "Unipipe"
}
SYSTEM_NAME_TO_HATCH_DICT = {
    "pretrained_noipr": "", "pretrained": "//", "multicontext_nocoord": "--",
    "multicontext": "xx", "unipipe": "++", "unipipe_dp": "**"
}
CSV_FILENAME_FMT = "{0}_5_{1}_{2}_{3}.csv"
SYSSTAT_FILENAME_FMT = "{0}_sysstat_5_{1}_{2}_{3}.csv"
CSV_FILENAME_FMT_TRANSMISSION = "{0}_transmission_state_5_{1}_{2}_{3}.csv"

FIGSIZE = (4, 2.4)
LEGEND_PROP = {"size": 9, "weight": "bold"}
LEGEND_COLSPACING = 0.5
AXLABEL_KW = {"fontsize": 10, "fontweight": "bold"} 
YTICK_LABEL_KW = {"fontsize": 10, "fontweight": "bold" }

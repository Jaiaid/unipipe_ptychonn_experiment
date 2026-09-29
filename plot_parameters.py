SYSTEM_NAME_LIST = ["pretrained_noipr", "multicontext", "unipipe_dp"]
# SYSTEM_NAME_LIST = ["pretrained_noipr", "unipipe_dp", "multicontext_mps_10_90", "multicontext_mps_50_50", "multicontext_mps_70_30", "multicontext_mps_90_10", ]
SYSTEM_NAME_TO_LEGEND_DICT = {
    "pretrained_noipr": "Pretrained", "pretrained": "Pretrained-Comp.",
    "multicontext_nocoord": "MultiContext-No Coordination", "multicontext": "MultiContext",
    "unipipe": "Unipipe(static)", "unipipe_dp": "Unipipe", "multicontext_mps_50_50": "MPS 50:50",
    "multicontext_mps_70_30": "MPS 70:30",
    "multicontext_mps_30_70": "MPS 30:70",
    "multicontext_mps_90_10": "MPS 90:10",
    "multicontext_mps_10_90": "MPS 10:90",
}
SYSTEM_NAME_TO_HATCH_DICT = {
    "pretrained_noipr": "", "pretrained": "//", "multicontext_nocoord": "--",
    "multicontext": "xx", "unipipe": "++", "unipipe_dp": "**", "multicontext_mps_50_50": "v",
    "multicontext_mps_70_30": "+", "multicontext_mps_90_10": "*", "multicontext_mps_10_90": "o",
    "multicontext_mps_30_70": "xx"
}
SYSTEM_NAME_TO_MARKER_DICT = {
    "pretrained_noipr": "o", "pretrained": "s", "multicontext_nocoord": "D",
    "multicontext": "x", "unipipe": "+", "unipipe_dp": "^", "multicontext_mps_50_50": "v",
    "multicontext_mps_70_30": "+", "multicontext_mps_90_10": "*", "multicontext_mps_10_90": "o",
    "multicontext_mps_30_70": "x"
}
CSV_FILENAME_FMT = "{0}_5_{1}_{2}_{3}.csv"
SYSSTAT_FILENAME_FMT = "{0}_sysstat_5_{1}_{2}_{3}.csv"
CSV_FILENAME_FMT_TRANSMISSION = "{0}_transmission_state_5_{1}_{2}_{3}.csv"

FIGSIZE = (4, 2.4)
LEGEND_PROP = {"size": 9, "weight": "bold"}
LEGEND_COLSPACING = 0.5
AXLABEL_KW = {"fontsize": 10, "fontweight": "bold"} 
YTICK_LABEL_KW = {"fontsize": 10, "fontweight": "bold" }

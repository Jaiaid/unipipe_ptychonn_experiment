SYSTEM_NAME_LIST = ["pretrained_noipr", "pretrained", "multicontext_nocoord", "multicontext", "unipipe", "unipipe_dp"]
SYSTEM_NAME_TO_LEGEND_DICT = {
    "pretrained_noipr": "Pretrained-No Learning", "pretrained": "Pretrained-Unipipe Coordination",
    "multicontext_nocoord": "MultiContext-No Coordination", "multicontext": "MultiContext-Compute Coordination",
    "unipipe": "Unipipe Heuristic", "unipipe_dp": "Unipipe Global"
}
SYSTEM_NAME_TO_HATCH_DICT = {
    "pretrained_noipr": "", "pretrained": "//", "multicontext_nocoord": "--",
    "multicontext": "xx", "unipipe": "++", "unipipe_dp": "**"
}
CSV_FILENAME_FMT = "{0}_5_{1}_{2}_{3}.csv"
CSV_FILENAME_FMT_TRANSMISSION = "{0}_transmission_state_5_{1}_{2}_{3}.csv"
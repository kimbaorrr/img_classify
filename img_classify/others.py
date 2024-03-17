import os
from datetime import datetime as dt

import seaborn as sns


def sns_global():
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="whitegrid", palette="pastel", rc=custom_params)

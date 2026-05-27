import pandas as pd
import sqlite3
import ast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from pathlib import Path
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

ctr_path = "../results.db"
vimet_path = "../results.db" # TODO Set to different database
font_size = 14
font_size_large = 16

def convert_listlike_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        # detect strings that *look* like lists
        sample = df[col].dropna().astype(str).head(20)
        if sample.map(lambda x: x.startswith("[") and x.endswith("]")).all():
            print(col)
            df[col] = df[col].map(lambda x: ast.literal_eval(x.replace("-inf", "None").replace("inf", "None")))
            df[col+"_corrected"] = df[col].apply(lambda l: [(0 if x is None or x < 0 else x) for x in l])
    return df



def df_to_latex_table(
    df: pd.DataFrame,
    filepath: str,
    caption: str = "",
    label: str = "",
    float_fmt: str = ".3f",
):
    """
    Convert a pandas DataFrame into a clean LaTeX booktabs table
    optimized for ML papers.
    """
    # Format floats
    df_formatted = df.copy()
    for col in df_formatted.select_dtypes(include=["float", "float64"]):
        df_formatted[col] = df_formatted[col].map(lambda x: f"{x:{float_fmt}}")

    latex_body = tabulate(
        df_formatted.values,
        headers=df_formatted.columns,
        tablefmt="latex_booktabs",
        showindex=False,
    )

    latex = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        f"{latex_body}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{tab:{label}}}\n"
        "\\end{table}\n"
    )
    pth = Path(filepath)
    pth.parent.mkdir(parents=True, exist_ok=True)
    pth.write_text(latex, encoding="utf-8")
    return latex




def compute_mean(s: pd.Series)-> pd.Series:
    return s.apply(
        lambda xs: np.nanmean([x for x in xs]) if None not in xs else np.nan
    )

_magma = cm.get_cmap("magma")


def _magma_rgba(pct: float):
    normalized = max(0.0, min(1.0, pct / 100.0))
    return _magma(normalized)


def _rgba_to_hex(rgba):
    return mcolors.to_hex(rgba, keep_alpha=False)[1:].upper()


def _relative_luminance(r, g, b):
    """
    WCAG relative luminance.
    Input RGB in [0,1]
    """
    def f(c):
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r_lin, g_lin, b_lin = f(r), f(g), f(b)
    return 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin


def _font_color_from_rgba(rgba):
    r, g, b, _ = rgba
    lum = _relative_luminance(r, g, b)
    return "white" if lum < 0.5 else "black"
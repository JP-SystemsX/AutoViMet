import pandas as pd
import sqlite3
import ast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from pathlib import Path

ctr_path = "../results.db"
vimet_path = "../results_vimet.db" 
font_size = 14
font_size_large = 16

def convert_listlike_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        # detect strings that *look* like lists
        sample = df[col].dropna().astype(str).head(20)
        if sample.map(lambda x: x.startswith("[") and x.endswith("]")).all():
            print(col)
            df[col] = df[col].map(ast.literal_eval)
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

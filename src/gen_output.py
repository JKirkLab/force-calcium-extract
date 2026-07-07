import pandas as pd
import streamlit as st
import numpy as np

def sort_order(row):
    file_name = row[0]
    parts = file_name.split("_")

    a = parts[-3]
    b = parts[-2]

    if a == "100" and b == "1":
        return (-1,)
    if a == "100" and b == "2":
        return (99999,)

    return (int(b),)

def extract_calc(filename):
    parts = filename.split("_")

    if parts[-1] == "relax":
        return 0
    
    elif parts[-1] == "contract" and parts[-3] == "100":
        return 100
    else:
        return int(parts[-2])

def generate_output(rows):
    rows.sort(key = sort_order)
    output = pd.DataFrame(rows, columns = ["filename", "2.1μm", "Slack"])
    output["cell #"] = output["filename"].apply(extract_calc)
    

    output["DF (V)"] = output["2.1μm"] - output["Slack"]

    output["AF (V)"] = np.nan

    output.loc[output.index[::2], "AF (V)"] = (
        output.loc[output.index[::2], "DF (V)"].to_numpy() -
        output.loc[output.index[1::2], "DF (V)"].to_numpy()
    )
    output["cell #"] = output["cell #"].astype(int)

    output["Group"] = np.nan

    mask = output["cell #"] != 0

    af_v = output.loc[mask, "AF (V)"].to_numpy()
    cell_numbers = output.loc[mask, "cell #"].to_numpy()



    output.loc[output.index[-1], "Group"] = (af_v[0] - af_v[-1]) / af_v[0]
    output = output.drop(columns = "filename")

    output = output[["cell #", "2.1μm", "Slack", "DF (V)", "AF (V)", "Group"]]
    return output, af_v, cell_numbers

def generate_force(af_voltage, cell_nums, height, width, fin_scale):
    csa =((((height + width) / 2) * (10**-3)/2) ** 2) * 3.14159
    af_voltage = af_voltage[1:]
    cell_nums = cell_nums[1:]
    force_df = pd.DataFrame({
        "Calcium": cell_nums,
        "Volts": af_voltage,
    })

    force_df["F (mN)"] = force_df["Volts"] * fin_scale
    force_df["F (mN/mm2)"] = force_df["F (mN)"] / csa

    return force_df, csa

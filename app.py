import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src import convert_voltage as cv
from src import extract_txt as ex
from src import select_points as sp

def set_padded_ylim(ax, time, y, xmin, xmax, pad_frac=0.05):
    mask = (time >= xmin) & (time <= xmax)
    if np.any(mask):
        ymin = np.min(y[mask])
        ymax = np.max(y[mask])
    else:
        ymin, ymax = np.min(y), np.max(y)

    yrange = ymax - ymin
    pad = yrange * pad_frac if yrange > 0 else 1.0
    ax.set_ylim(ymin - pad, ymax + pad)

st.title("Force Calcium Data Extractor")
st.write("Upload your data here")

st.sidebar.header("Upload:")

uploaded_files = st.sidebar.file_uploader(
    "Choose one or more files",
    accept_multiple_files=True
)

rows = []

if uploaded_files:
    st.write("Uploaded files:")
    for file in uploaded_files:
        st.subheader(file.name)

        df, scale, offset = ex.extract_metadata(file)
        result = sp.extract_4_points(df["Time_ms"], df["Aux2_mV"]/1000, frac=0.015)

        time = np.asarray(df["Time_ms"])
        force = np.asarray(df["Aux2_mV"]/1000)
        force_smooth = result["smoothed_force"]

        p1 = result["p1"]
        p2 = result["p2"]
        p3 = result["p3"]
        p4 = result["p4"]

        rows.append([file.name,p1['value'], p2['value']])
        rows.append([file.name,p4['value'], p3['value']])

        pad = 100  

        xmin1 = min(p1["time_ms"], p2["time_ms"]) - pad
        xmax1 = max(p1["time_ms"], p2["time_ms"]) + pad

        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(time, force, alpha=0.3, label="raw")
        ax1.plot(time, force_smooth, label="smoothed")
        ax1.scatter(p1["time_ms"], p1["value"], label="p1")
        ax1.scatter(p2["time_ms"], p2["value"], label="p2")

        ax1.set_xlim(xmin1, xmax1)
        set_padded_ylim(ax1, time, force_smooth, xmin1, xmax1)

        ax1.set_xlabel("Time (ms)")
        ax1.set_ylabel("Force")
        ax1.set_title("Zoomed view: p1 and p2")
        ax1.legend()
        st.pyplot(fig1)


        # Plot 2: p3 & p4
        xmin2 = min(p3["time_ms"], p4["time_ms"]) - pad
        xmax2 = max(p3["time_ms"], p4["time_ms"]) + pad

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(time, force, alpha=0.3, label="raw")
        ax2.plot(time, force_smooth, label="smoothed")
        ax2.scatter(p3["time_ms"], p3["value"], label="p3")
        ax2.scatter(p4["time_ms"], p4["value"], label="p4")

        ax2.set_xlim(xmin2, xmax2)
        set_padded_ylim(ax2, time, force_smooth, xmin2, xmax2)

        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("Force")
        ax2.set_title("Zoomed view: p3 and p4")
        ax2.legend()
        st.pyplot(fig2)

        # st.write(
        #     f"{file.name}: "
        #     f"2.1: ({p1['value']}, {p4['value']}), "
        #     f"slack: ({p2['value']}, {p3['value']})"
        # )
        table_df = pd.DataFrame(
        [
            [p1["value"], p2["value"]],  # top row
            [p4["value"], p3["value"]]   # bottom row (p4 left, p3 right)
        ],
        index=["Top", "Bottom"],
        columns=["Left", "Right"]
        )

        st.table(table_df)

    if rows:
        combined_table = pd.DataFrame(rows, columns=["File", "Label", "Value"])
        st.dataframe(combined_table)
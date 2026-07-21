import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src import convert_voltage as cv
from src import extract_txt as ex
from src import select_points as sp
from src.gen_output import generate_output, generate_force

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

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


uploaded_files = st.sidebar.file_uploader(
    "Choose one or more files",
    accept_multiple_files=True
)

folder_upload = st.sidebar.file_uploader(
    "Upload full folders here",
    accept_multiple_files = "directory",
    key=f"folder_uploader_{st.session_state.uploader_key}",
)

if st.sidebar.button("Clear uploaded folder"):
    st.session_state.uploader_key += 1
    st.rerun()

if uploaded_files:
    st.write(f"{len(uploaded_files)} files uploaded")

fin_scale = st.number_input(
    "Force In Scale (mN/V)",
    value=0.0,
    step=0.01,
    format="%.6f"
)


height = st.number_input(
    "Height (μm)",
    value=0.0,
    step=0.01,
    format="%.6f"
)

width = st.number_input(
    "Width (μm)",
    value=0.0,
    step=0.01,
    format="%.6f"
)



def process_group(group_files, height, width, fin_scale):
    rows = []
    for file in group_files:
        basename = file.name.replace("\\", "/").split("/")[-1]
        # st.write(f"DEBUG: processing file {i+1}/{len(group_files)}: {basename}")
        # st.subheader(basename)

        df, scale, offset = ex.extract_metadata(file)

        df['Voltage'] = df['Force'].apply(
            lambda x: cv.convert_force_voltage(x, scale, offset)
        )
        result = sp.extract_4_points(df["Time_ms"], df['Voltage'], frac=0.015)
        p3p4 = sp.extract_p3_p4(df["Time_ms"], df['Voltage'], df['Lout'])

        time = np.asarray(df["Time_ms"])
        force = np.asarray(df["Voltage"])
        force_smooth = result["smoothed_force"]

        p1 = p3p4["p1"]
        p2 = p3p4["p2"]
        p3 = p3p4["p3"]
        p4 = p3p4["p4"]

        rows.append([basename + "_contract", p1['value'], p2['value']])
        rows.append([basename + "_relax", p4['value'], p3['value']])

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

    if rows:
        voltage_df, af_v, cell_nums = generate_output(rows)
        st.write("Height: ", height)
        st.write("Width: ", width)
        force_df, csa = generate_force(af_v, cell_nums, height, width, fin_scale)
        st.write("CSA (mm^2): ", csa)
        st.dataframe(voltage_df, hide_index=True, column_order=["cell #", "2.1μm", "Slack", "DF (V)", "AF (V)", "Group"], height=38 + 35 * len(voltage_df))
        st.dataframe(force_df, hide_index=True, height=38 + 35 * len(force_df))


if uploaded_files:
    if not height or not width:
        st.warning("Please enter Height and Width to process files.")
    else:
        with st.spinner("Processing files..."):
            st.write("Uploaded files:")
            with st.expander("Show plots"):
                process_group(uploaded_files, height, width, fin_scale)

elif folder_upload:
    if not height or not width:
        st.warning("Please enter Height and Width to process files.")
    else:
        # st.write("DEBUG: all uploaded filenames:", [f.name for f in folder_upload])

        txt_files = [f for f in folder_upload if not f.name.replace("\\", "/").split("/")[-1].startswith(".")]

        # st.write(f"DEBUG: {len(folder_upload)} total files uploaded, {len(txt_files)} data files")

        # Browser directory uploads include the root folder in the path:
        # root/subfolder/file.txt (depth 3) → group by parts[1]
        # root/file.txt           (depth 2) → single flat group
        max_depth = max((len(f.name.replace("\\", "/").split("/")) for f in txt_files), default=1)
        # st.write(f"DEBUG: max path depth = {max_depth}")

        if max_depth >= 3:
            groups = {}
            for f in txt_files:
                parts = f.name.replace("\\", "/").split("/")
                key = parts[1]
                groups.setdefault(key, []).append(f)

            # st.write(f"DEBUG: {len(groups)} subfolders found: {list(groups.keys())}")

            for group_name, group_files in groups.items():
                st.header(f"Folder: {group_name}")
                # st.write(f"DEBUG: {len(group_files)} files in this folder")
                with st.spinner(f"Processing {group_name}..."):
                    with st.expander("Show plots"):
                        process_group(group_files, height, width, fin_scale)
        else:
            # st.write(f"DEBUG: flat folder, processing {len(txt_files)} files as one group")
            with st.spinner("Processing files..."):
                st.write("Uploaded files:")
                with st.expander("Show plots"):
                    process_group(txt_files, height, width, fin_scale)

        
        



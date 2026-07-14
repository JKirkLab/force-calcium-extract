# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

A Streamlit web app for processing ASI 600A force transducer data files from muscle fiber experiments. Users upload raw `.txt` data files, provide sample geometry (height/width in μm) and a force-in scale factor (mN/V), and the app extracts four characteristic points from each force trace, then computes active force (AF) normalized by cross-sectional area.

## Running the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Architecture

The pipeline runs left-to-right through these modules:

1. **`src/extract_txt.py`** — Parses ASI 600A binary-adjacent text files. Reads `scale` and `offset` from the `406A` hardware line, then extracts `(Time_ms, Force)` pairs from the `*** Force and Length Signals vs Time ***` section (column index 3 for force).

2. **`src/convert_voltage.py`** — Converts raw ADU force values to volts: `voltage = (fin - offset) / scale`.

3. **`src/select_points.py`** — `extract_4_points()` applies a Savitzky-Golay filter then finds four characteristic points (p1–p4) from the leading and trailing `frac` fraction of the smoothed trace:
   - p1: max in left edge region (contraction start) — **currently hard-coded to index 520**
   - p2: min in left edge region (slack at contraction)
   - p3: min in right edge region — **currently hard-coded to index 100020**
   - p4: max in right edge region (relaxation peak)
   
   The hard-coded indices in `select_points.py:91-93` are an in-progress experimental change; the commented-out lines show the original computed approach.

4. **`src/gen_output.py`** — `generate_output()` sorts files by calcium concentration (parsed from filename like `_practice2_80` → 80% calcium), computes developed force `DF = 2.1μm − Slack` and active force `AF = DF_contract − DF_relax`. `generate_force()` computes CSA from height/width and normalizes force to mN/mm².

5. **`app.py`** — Streamlit UI that wires the pipeline together, renders zoomed matplotlib plots around each detected point pair, and displays the result DataFrames.

## Filename convention

Data files are named like `<date>_<experiment>_<calcium_pct>` or `<date>_<experiment>_100_1` / `_100_2` for duplicate 100% calcium runs. The sort order puts `100_1` first, `100_2` last, with other percentages sorted numerically. This ordering determines the AF calculation pairing (contract/relax rows alternate).

## Key data flow note

`rows` in `app.py` accumulates `[filename+"_contract", p1_value, p2_value]` and `[filename+"_relax", p4_value, p3_value]` — note the argument ordering (p1/p2 for contract, p4/p3 for relax). The `generate_output` function expects rows to alternate contract/relax for the AF differencing at lines 40–43 of `gen_output.py`.

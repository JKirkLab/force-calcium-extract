import pandas as pd
def extract_metadata(file):
    colnames = [
        "Time_ms","Lin_mm","Lout_mm","Fin_mN","Fout_mN",
        "Aux1_C","Aux2_mV","SL_um","Stimulus_Triggers",
    ]

    rows = []
    in_section = False

    scale = None
    offset = None

    # read uploaded file as text
    for line in file:
        line = line.decode("utf-8")  # convert bytes → string

        if line.startswith("406A"):
            test_list = line.split()
            scale = float(test_list[-1])
            offset = float(test_list[-2])

        s = line.strip()

        if s == "*** Force and Length Signals vs Time ***":
            in_section = True
            continue

        if not in_section:
            continue

        if not s or s.startswith("Time (ms)"):
            continue

        parts = s.split()

        if len(parts) != 9:
            continue

        try:
            row = [float(x) for x in parts[:8]] + [int(parts[8])]
            rows.append(row)
        except ValueError:
            continue

    df = pd.DataFrame(rows, columns=colnames)

    return df, scale, offset
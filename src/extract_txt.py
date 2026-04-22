import pandas as pd

def extract_metadata(file):
    rows = []
    in_section = False
    scale = None
    offset = None
    FORCE_IDX = 3

    for line in file:
        line = line.decode("utf-8", errors="ignore")
        s = line.strip()

        if line.startswith("406A"):
            parts = line.split()
            try:
                scale = float(parts[-1])
                offset = float(parts[-2])
            except (ValueError, IndexError):
                pass

        if s == "*** Force and Length Signals vs Time ***":
            in_section = True
            continue

        if not in_section:
            continue

        if not s:
            continue

        parts = s.split()

        try:
            if len(parts) > FORCE_IDX:
                time_ms = float(parts[0])
                force = float(parts[FORCE_IDX])
                rows.append([time_ms, force])
        except ValueError:
            continue

    df = pd.DataFrame(rows, columns=["Time_ms", "Force"])
    return df, scale, offset
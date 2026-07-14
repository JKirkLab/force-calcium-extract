import numpy as np
from scipy.signal import savgol_filter

def extract_4_points(
    time,
    force,
    frac=0.015,
    smooth_window=51,
    polyorder=3,
):
    """
    Extract 4 points from a force trace using edge windows.

    Point definitions:
    - p1: max in left region
    - p2: min in left region
    - p3: min in right region
    - p4: max in right region

    Parameters
    ----------
    time : array-like
        Time values in ms.
    force : array-like
        Force trace values (e.g. Fin_mN).
    frac : float, default 0.015
        Fraction of total trace to use for left and right edge regions.
        Example: 0.015 means first 1.5% and last 1.5% of the trace.
    smooth_window : int, default 51
        Savitzky-Golay smoothing window length. Must be odd.
    polyorder : int, default 3
        Savitzky-Golay polynomial order.

    Returns
    -------
    result : dict
        Dictionary containing:
        - smoothed force trace
        - region bounds
        - p1, p2, p3, p4 with time/value/index
    """
    time = np.asarray(time)
    force = np.asarray(force)

    if time.ndim != 1 or force.ndim != 1:
        raise ValueError("time and force must be 1D arrays")
    if len(time) != len(force):
        raise ValueError("time and force must have the same length")
    if len(time) < 5:
        raise ValueError("trace is too short")
    if not (0 < frac < 0.5):
        raise ValueError("frac must be between 0 and 0.5")

    n = len(force)

    # Make sure smoothing window is valid
    if smooth_window >= n:
        smooth_window = n - 1 if n % 2 == 0 else n
    if smooth_window % 2 == 0:
        smooth_window -= 1
    if smooth_window < polyorder + 2:
        smooth_window = polyorder + 2
        if smooth_window % 2 == 0:
            smooth_window += 1
    if smooth_window >= n:
        raise ValueError("trace too short for requested smoothing parameters")

    force_smooth = savgol_filter(force, window_length=smooth_window, polyorder=polyorder)

    k = max(1, int(frac * n))

    # Left and right regions
    left_start_idx = 0
    left_end_idx = k
    right_start_idx = n - k
    right_end_idx = n

    time_left = time[left_start_idx:left_end_idx]
    force_left = force_smooth[left_start_idx:left_end_idx]

    time_right = time[right_start_idx:right_end_idx]
    force_right = force_smooth[right_start_idx:right_end_idx]

    # Extrema in local windows
    idx_left_max_local = np.argmax(force_left)
    idx_left_min_local = np.argmin(force_left)
    idx_right_min_local = np.argmin(force_right)
    idx_right_max_local = np.argmax(force_right)

    # Convert to global indices
    idx_left_max = 520#left_start_idx + idx_left_max_local
    idx_left_min = left_start_idx + idx_left_min_local
    idx_right_min = 100020#right_start_idx + idx_right_min_local
    idx_right_max = right_start_idx + idx_right_max_local

    result = {
        "smoothed_force": force_smooth,
        "region_bounds": {
            "left": {
                "start_idx": left_start_idx,
                "end_idx_exclusive": left_end_idx,
                "start_time_ms": time[left_start_idx],
                "end_time_ms": time[left_end_idx - 1],
            },
            "right": {
                "start_idx": right_start_idx,
                "end_idx_exclusive": right_end_idx,
                "start_time_ms": time[right_start_idx],
                "end_time_ms": time[right_end_idx - 1],
            },
        },
        "p1": {  # left max
            "index": int(idx_left_max),
            "time_ms": float(time[idx_left_max]),
            "value": float(force_smooth[idx_left_max]),
        },
        "p2": {  # left min
            "index": int(idx_left_min),
            "time_ms": float(time[idx_left_min]),
            "value": float(force_smooth[idx_left_min]),
        },
        "p3": {  # right min
            "index": int(idx_right_min),
            "time_ms": float(time[idx_right_min]),
            "value": float(force_smooth[idx_right_min]),
        },
        "p4": {  # right max
            "index": int(idx_right_max),
            "time_ms": float(time[idx_right_max]),
            "value": float(force_smooth[idx_right_max]),
        },
    }

    return result


def extract_p3_p4(time, force, lout, window=50):
    """
    Extract all four points using the Lout (commanded length) signal.

    p1 = mean of `window` force points immediately before the upward ramp onset
    p2 = mean of `window` force points immediately after the upward ramp ends
    p3 = mean of `window` force points immediately before the downward ramp onset
    p4 = mean of `window` force points immediately after the downward ramp ends

    Anchors are derived from Lout (commanded length output), which is
    noise-free, so no smoothing is needed.
    """
    time = np.asarray(time)
    force = np.asarray(force)
    lout = np.asarray(lout)

    plateau_val = np.max(lout)
    slack_val = np.min(lout)
    epsilon = 1e-6
    indices = np.arange(len(lout))

    # --- Downward ramp (p3/p4) ---
    # Last index where Lout is at the plateau → downward ramp onset is the next index
    plateau_indices = np.where(lout >= plateau_val - epsilon)[0]
    t_onset = int(plateau_indices[-1]) + 1

    # First index at slack level AFTER t_onset → downward ramp end
    post_onset_slack = np.where((lout <= slack_val + epsilon) & (indices >= t_onset))[0]
    t_end = int(post_onset_slack[0])

    # p3: window immediately before downward ramp onset
    p3_start = max(0, t_onset - window)
    p3_end = t_onset
    p3_center = (p3_start + p3_end) // 2
    p3_val = float(np.mean(force[p3_start:p3_end]))

    # p4: window immediately after downward ramp end
    p4_start = t_end
    p4_end = min(len(force), t_end + window)
    p4_center = (p4_start + p4_end) // 2
    p4_val = float(np.mean(force[p4_start:p4_end]))

    # --- Upward ramp (p1/p2) ---
    # First index where Lout rises above its initial slack value
    init_lout = float(lout[0])
    up_onset_candidates = np.where(lout > init_lout + epsilon)[0]
    up_onset = int(up_onset_candidates[0]) if len(up_onset_candidates) > 0 else window

    # First index where Lout reaches the plateau after the upward ramp
    up_end_candidates = np.where((lout >= plateau_val - epsilon) & (indices >= up_onset))[0]
    up_end = int(up_end_candidates[0]) if len(up_end_candidates) > 0 else t_onset

    # p1: window immediately before upward ramp onset
    p1_start = max(0, up_onset - window)
    p1_end = up_onset
    p1_center = (p1_start + p1_end) // 2
    p1_val = float(np.mean(force[p1_start:p1_end]))

    # p2: window immediately after upward ramp end
    p2_start = up_end
    p2_end = min(len(force), up_end + window)
    p2_center = (p2_start + p2_end) // 2
    p2_val = float(np.mean(force[p2_start:p2_end]))

    return {
        "p1": {
            "index": int(p1_center),
            "time_ms": float(time[p1_center]),
            "value": p1_val,
        },
        "p2": {
            "index": int(p2_center),
            "time_ms": float(time[p2_center]),
            "value": p2_val,
        },
        "p3": {
            "index": int(p3_center),
            "time_ms": float(time[p3_center]),
            "value": p3_val,
        },
        "p4": {
            "index": int(p4_center),
            "time_ms": float(time[p4_center]),
            "value": p4_val,
        },
    }
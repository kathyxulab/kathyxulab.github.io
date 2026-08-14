import pandas as pd, numpy as np
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(name):
    df = pd.read_csv(name, encoding='utf-8')
    df.columns = [c.strip() for c in df.columns]
    df['t'] = (df['Timestamp(ms)'] - df['Timestamp(ms)'].iloc[0]) / 1000.0
    df['gyro_mag'] = np.sqrt(df['Angular Velocity X']**2 + df['Angular Velocity Y']**2 + df['Angular Velocity Z']**2)
    return df

def cycle_periods(t, sig, min_period_s, max_period_s):
    dt = np.median(np.diff(t))
    distance = max(1, int(round(min_period_s/dt)))
    prominence = np.std(sig)*0.5
    peaks, _ = find_peaks(sig, distance=distance, prominence=prominence)
    if len(peaks) < 4:
        return None
    pt = t[peaks]
    periods = np.diff(pt)
    keep = (periods>=min_period_s)&(periods<=max_period_s)
    return periods[keep]

toy = load("fixed_rotation_toy_imu_data.csv")

toy_fine_segs = [(29.4,44.1),(45.8,56.8),(112.9,125.1),(164.3,179.0),(180.7,190.1),
                  (263.7,275.7),(320.0,408.8),(467.0,492.7),(505.4,525.1),(554.7,575.2),
                  (582.0,608.7),(619.6,691.7),(695.3,715.0),(722.4,734.7),(738.6,763.5)]

# --- Figure 1: full-session overview with active windows shaded ---
fig1, ax1 = plt.subplots(figsize=(13,4))
ax1.plot(toy['t'], toy['gyro_mag'], lw=0.4, color='tab:orange')
for t0,t1 in toy_fine_segs:
    ax1.axvspan(t0, t1, color='tab:green', alpha=0.15)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Gyro magnitude (deg/s)")
ax1.set_title("Toy device — full 871s recording, 15 active rotation windows highlighted (data completeness check)")
plt.tight_layout()
plt.savefig("toy_full_overview.png", dpi=120)
print("saved toy_full_overview.png")

# --- Figure 2: per-segment rotation rate + CV ---
rates=[]; cvs=[]; durs=[]
for t0,t1 in toy_fine_segs:
    m=(toy['t']>=t0)&(toy['t']<=t1)
    p = cycle_periods(toy.loc[m,'t'].values, toy.loc[m,'gyro_mag'].values, 0.3,1.5)
    if p is not None:
        rates.append(60/p.mean()); cvs.append(p.std()/p.mean()); durs.append(t1-t0)
    else:
        rates.append(np.nan); cvs.append(np.nan); durs.append(t1-t0)

x = np.arange(len(toy_fine_segs))
fig2, ax2 = plt.subplots(2,1, figsize=(12,7), sharex=True)
ax2[0].bar(x, rates, color='tab:green')
ax2[0].set_ylabel("Rotation rate (cycles/min)")
ax2[0].set_title("Toy: rotation rate and timing-consistency (CV) across the 15 active windows")
ax2[1].bar(x, cvs, color='tab:orange')
ax2[1].set_ylabel("Period CV (std/mean)")
ax2[1].set_xlabel("Active window #")
ax2[1].set_xticks(x)
plt.tight_layout()
plt.savefig("toy_rate_cv.png", dpi=120)
print("saved toy_rate_cv.png")
print("rate range:", np.nanmin(rates), np.nanmax(rates), "median:", np.nanmedian(rates))
print("cv range:", np.nanmin(cvs), np.nanmax(cvs), "median:", np.nanmedian(cvs))

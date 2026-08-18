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
    df['acc_mag'] = np.sqrt(df['Acceleration X']**2 + df['Acceleration Y']**2 + df['Acceleration Z']**2)
    return df

def sliding_stroke_stats(t, gyro, angle_x, win_s=60.0, stride_s=20.0,
                          min_period_s=0.4, max_period_s=3.0, min_strokes=8):
    dt_med = np.median(np.diff(t))
    starts = np.arange(t[0], t[-1] - win_s, stride_s)
    rows = []
    for s in starts:
        e = s + win_s
        m = (t >= s) & (t < e)
        if m.sum() < 20:
            continue
        tw, sw, aw = t[m], gyro[m], angle_x[m]
        prominence = np.std(sw) * 0.5
        distance = max(1, int(round(0.4 / dt_med)))
        peaks, _ = find_peaks(sw, distance=distance, prominence=prominence)
        rec = dict(t_mid=s + win_s/2, roll_range=np.percentile(aw, 95) - np.percentile(aw, 5))
        if len(peaks) >= 5:
            pt = tw[peaks]
            periods = np.diff(pt)
            periods = periods[(periods >= min_period_s) & (periods <= max_period_s)]
            if len(periods) >= min_strokes:
                rec['sr'] = 60 / periods.mean()
                rec['cv'] = periods.std() / periods.mean()
        rows.append(rec)
    return pd.DataFrame(rows)

def break_gaps(sub, xcol, max_gap):
    sub = sub.sort_values(xcol).reset_index(drop=True)
    gaps = sub[xcol].diff() > max_gap
    out = []
    for i, row in sub.iterrows():
        if i > 0 and gaps.iloc[i]:
            blank = row.copy(); blank[:] = np.nan
            out.append(blank)
        out.append(row)
    return pd.DataFrame(out)

human = load("real_swimmer_session_imu_data.csv")
t, gyro, acc, angle_x = human['t'].values, human['gyro_mag'].values, human['acc_mag'].values, human['Angle X'].values

wdf = sliding_stroke_stats(t, gyro, angle_x)
peaks_acc, _ = find_peaks(acc, height=3.0, distance=int(round(1.0 / np.median(np.diff(t)))))

# ---------- Figure Overview: full-session gyro + turn events, stroke rate, CV, roll range ----------
sr_full = break_gaps(wdf.dropna(subset=['sr']), 't_mid', max_gap=20.0 * 1.5)
roll_full = break_gaps(wdf.dropna(subset=['roll_range']), 't_mid', max_gap=20.0 * 1.5)

fig0, axes0 = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
axes0[0].plot(t, gyro, lw=0.3, color='gray')
axes0[0].scatter(t[peaks_acc], gyro[peaks_acc], color='red', s=20, zorder=5,
                  label=f'accel spike >3g (turn/push-off, n={len(peaks_acc)})')
axes0[0].set_title("Full session gyro magnitude with detected turn/push-off events")
axes0[0].legend(loc='upper right')

axes0[1].plot(sr_full['t_mid'], sr_full['sr'], marker='o', ms=4, color='tab:blue')
axes0[1].set_ylabel("Stroke rate (/min)")
axes0[1].set_title("Stroke rate over the session (60s sliding window)")

axes0[2].plot(sr_full['t_mid'], sr_full['cv'], marker='o', ms=4, color='tab:red')
axes0[2].set_ylabel("Stroke period CV")
axes0[2].set_title("Stroke rhythm consistency over the session")

axes0[3].plot(roll_full['t_mid'], roll_full['roll_range'], marker='o', ms=3, color='tab:green')
axes0[3].set_ylabel("Angle X 5-95pct range (deg)")
axes0[3].set_xlabel("Time (s)")
axes0[3].set_title("Body roll range over the session")

plt.tight_layout()
plt.savefig("session_overview.png", dpi=120)
print("saved session_overview.png")

# ---------- Figure A: Block A (0-1650s) work/rest interval structure ----------
mA = (wdf['t_mid'] >= 0) & (wdf['t_mid'] <= 1650)
sub = wdf[mA]
sr_b = break_gaps(sub.dropna(subset=['sr']), 't_mid', max_gap=60)
roll_b = break_gaps(sub.dropna(subset=['roll_range']), 't_mid', max_gap=60)

fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
axes[0].plot(sr_b['t_mid'], sr_b['sr'], color='tab:blue', marker='o', ms=4, label='Stroke rate (/min)')
ax0b = axes[0].twinx()
ax0b.plot(sr_b['t_mid'], sr_b['cv'], color='tab:red', marker='s', ms=3, alpha=0.6, label='Period CV')
axes[0].set_ylabel("Stroke rate (/min)", color='tab:blue')
ax0b.set_ylabel("Period CV", color='tab:red')
axes[0].set_title("Block A (0-1650s): stroke rate & rhythm CV")

axes[1].plot(roll_b['t_mid'], roll_b['roll_range'], color='tab:green', marker='o', ms=3)
axes[1].axhline(50, color='gray', ls='--', lw=1)
axes[1].set_ylabel("Body roll range (deg)")
axes[1].set_xlabel("Time (s)")
axes[1].set_title("Body roll range (dashed line = rest/idle threshold)")
plt.tight_layout()
plt.savefig("block_a_interval_structure.png", dpi=120)
print("saved block_a_interval_structure.png")

# ---------- Figure B: Block B (3400-3970s) fast-start-then-fade + turn events ----------
mB_raw = (t >= 3400) & (t <= 3970)
mB_w = (wdf['t_mid'] >= 3400) & (wdf['t_mid'] <= 3970)
subB = wdf[mB_w]
srB = break_gaps(subB.dropna(subset=['sr']), 't_mid', max_gap=60)

peaks_in_range = peaks_acc[(t[peaks_acc] >= 3400) & (t[peaks_acc] <= 3970)]

fig2, axes2 = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
axes2[0].plot(t[mB_raw], gyro[mB_raw], lw=0.4, color='gray')
axes2[0].scatter(t[peaks_in_range], gyro[peaks_in_range], color='red', s=25, zorder=5,
                  label=f'accel spike >3g (n={len(peaks_in_range)})')
axes2[0].set_title("Block B (3400-3970s): gyro magnitude with detected turn/push-off events")
axes2[0].legend()

axes2[1].plot(srB['t_mid'], srB['sr'], color='tab:blue', marker='o', ms=5)
axes2[1].set_ylabel("Stroke rate (/min)")
axes2[1].set_title("Stroke rate: fast start, fading later")

axes2[2].plot(srB['t_mid'], srB['cv'], color='tab:red', marker='o', ms=5)
axes2[2].set_ylabel("Period CV")
axes2[2].set_xlabel("Time (s)")
axes2[2].set_title("Rhythm consistency: most regular at the start, degrading later")
plt.tight_layout()
plt.savefig("block_b_fade_and_turns.png", dpi=120)
print("saved block_b_fade_and_turns.png")

print("\nn turn/push-off events in Block A (0-1650s):",
      ((t[peaks_acc] >= 0) & (t[peaks_acc] <= 1650)).sum())
print("n turn/push-off events in Block B (3400-3970s):", len(peaks_in_range))

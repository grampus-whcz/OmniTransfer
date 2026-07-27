import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime, timedelta
import numpy as np

# ===================== Configuration Parameters =====================
CSV_PATH = "/root/shared-nvme/data_set/OpenRCA/Bank/record.csv"
# CSV_PATH = "/root/shared-nvme/data_set/OpenRCA/Telecom/record.csv"
# CSV_PATH = "/root/shared-nvme/data_set/OpenRCA/Market/record_all.csv"
FIG_SAVE_PATH = "./bank_groundtruth_halfhour_dist.pdf"

# Disable Chinese font config, use default English font
plt.rcParams["axes.unicode_minus"] = False

# ===================== 1. Load and Process CSV Data =====================
df = pd.read_csv(CSV_PATH)
df["datetime"] = pd.to_datetime(df["datetime"])

def get_halfhour_window(dt: datetime):
    """
    Align fault timestamp to the nearest 30-min window start
    Return: window start datetime, window label string, offset minutes within window
    """
    minute = dt.minute
    if minute < 30:
        window_start = dt.replace(minute=0, second=0, microsecond=0)
    else:
        window_start = dt.replace(minute=30, second=0, microsecond=0)
    window_label = window_start.strftime("%Y-%m-%d %H:%M")
    offset_min = (dt - window_start).total_seconds() / 60.0
    return window_start, window_label, offset_min

# Batch compute window information for all fault records
window_info = df["datetime"].apply(get_halfhour_window)
df[["window_start", "window_label", "offset_min"]] = pd.DataFrame(window_info.tolist(), index=df.index)

# ===================== 2. Statistical Distribution within 30-min Window =====================
bins = np.arange(0, 31, 5)
counts, bin_edges = np.histogram(df["offset_min"], bins=bins)
print("===== Ground Truth Timestamp Offset Distribution (Minutes within 30-min Window) =====")
for i in range(len(counts)):
    left = bin_edges[i]
    right = bin_edges[i+1]
    print(f"[{left:2.0f}, {right:2.0f}) min: {counts[i]} fault records")
total_records = len(df)
print(f"Total fault records: {total_records}")

# ===================== 3. Plot Scatter Distribution Figure =====================
fig, ax = plt.subplots(figsize=(12, 8))

# Map unique time windows to y-axis coordinate
unique_windows = sorted(df["window_label"].unique())
window_to_y = {window: idx for idx, window in enumerate(unique_windows)}
df["y_pos"] = df["window_label"].map(window_to_y)

# Draw scatter points
scatter = ax.scatter(
    x=df["offset_min"],
    y=df["y_pos"],
    s=60,
    alpha=0.7,
    c="#2E86AB",
    edgecolors="black",
    linewidth=0.5
)

# Axis configuration
ax.set_xlim(0, 30)
ax.set_xticks(np.arange(0, 31, 5))
ax.set_xlabel("Offset Minutes of Fault within 30-Minute Window (0 ~ 30)", fontsize=12)

ax.set_yticks(range(len(unique_windows)))
# Reduce ytick label font size from 9 to 6
ax.set_yticklabels(unique_windows, fontsize=3)
ax.set_ylabel("30-Minute Time Windows", fontsize=12)

# ax.set_title("Distribution of $\\text{Telecom}$ Dataset Ground Truth Root Cause Timestamps in 30-Minute Windows", fontsize=14, pad=15)
ax.grid(axis="x", alpha=0.3, linestyle="--")

# Save output figure
plt.tight_layout()
plt.savefig(FIG_SAVE_PATH, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nDistribution figure saved to: {FIG_SAVE_PATH}")
# %%
import matplotlib.pyplot as plt
import wfdb

# %%
record = wfdb.rdrecord(
    "datasets/mit-bih-arrhythmia-database-1.0.0/100"
)  # Replace with the actual record name
signals = record.p_signal

# Plot each lead
num_leads = len(signals[0])
for lead in range(num_leads):
    plt.figure(figsize=(10, 4))
    plt.plot(signals[:1000, lead])
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.title(f"ECG Lead {lead+1}")
    plt.grid(True)
    plt.show()
# %%
annotation = wfdb.rdann(
    "datasets/mit-bih-arrhythmia-database-1.0.0/100", "atr"
)  # Replace with the actual record name

plt.figure(figsize=(10, 4))
plt.plot(
    signals[:1000, 0]
)  # Slice the signals array to only plot the first 1000 samples of the first lead
plt.scatter(
    annotation.sample[:1000],
    signals[annotation.sample[:1000], 0],
    c="red",
    marker="o",
    label="R-peaks",
)
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.title("ECG Lead 1 with R-peaks (First 1000 Samples)")
plt.legend()
plt.grid(True)
plt.show()


# %%
annotation.symbol
# %%

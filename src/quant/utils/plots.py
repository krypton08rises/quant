import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plots(df: pd.DataFrame):
    """
    :return:
    """
    percent_change = df[["percent_change"]].values

    # Compute statistics
    mean = np.mean(percent_change)
    std_dev = np.std(percent_change)
    plt.figure(figsize=(8, 5))
    sns.histplot(percent_change, kde=True, bins=30, color="blue", edgecolor="black", alpha=0.7)

    # Plot mean and standard deviation lines
    plt.axvline(mean, color="red", linestyle="dashed", linewidth=2, label="Mean")
    plt.axvline(mean + std_dev, color="green", linestyle="dashed", linewidth=1, label="+1σ")
    plt.axvline(mean - std_dev, color="green", linestyle="dashed", linewidth=1, label="-1σ")
    plt.axvline(mean + 2 * std_dev, color="orange", linestyle="dashed", linewidth=1, label="+2σ")
    plt.axvline(mean - 2 * std_dev, color="orange", linestyle="dashed", linewidth=1, label="-2σ")
    plt.axvline(mean + 3 * std_dev, color="purple", linestyle="dashed", linewidth=1, label="+3σ")
    plt.axvline(mean - 3 * std_dev, color="purple", linestyle="dashed", linewidth=1, label="-3σ")

    # Labels and legend
    plt.title("Distribution of Percent Change with Std Dev")
    plt.xlabel("Percent Change")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

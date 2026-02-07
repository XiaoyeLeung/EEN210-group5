import matplotlib.pyplot as plt
import numpy as np
import os
import platform
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# for mac
if platform.system() == 'Darwin':
    import matplotlib
    matplotlib.use('TkAgg')



def plot_raw_lines(dfs_dict, feature="acc_mag", ylabel="Acceleration (g)"):

    n = len(dfs_dict)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(12, 4*n), sharex=False)
    if n == 1: axes = [axes]
    
    for i, (label, df) in enumerate(dfs_dict.items()):
        ax = axes[i]
        

        ax.plot(df["t_s"], df[feature], color='tab:blue', lw=1.5, label=feature)
        
     
        if "acc" in feature and "Fall" in label:
            peak_idx = df[feature].idxmax()
            peak_time = df.loc[peak_idx, "t_s"]
            peak_val = df.loc[peak_idx, feature]
            ax.annotate('Impact', xy=(peak_time, peak_val), xytext=(peak_time+1, peak_val),
                        arrowprops=dict(facecolor='red', shrink=0.05))

        ax.set_title(f"{label}: Time Domain Trend", fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.3)
        if i == 0: ax.legend(loc="upper right")
    
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

def plot_raw_hists(dfs_dict, feature="acc_mag"):
   
    n = len(dfs_dict)
    fig, axes = plt.subplots(nrows=1, ncols=n, figsize=(6*n, 5))
    if n == 1: axes = [axes]

    for i, (label, df) in enumerate(dfs_dict.items()):
        ax = axes[i]
        

        sns.histplot(data=df, x=feature, kde=True, ax=ax, 
                     color='tab:blue', stat="density", element="step", alpha=0.3)

        mean_val = df[feature].mean()
        ax.axvline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.2f}')
        
        ax.set_title(f"{label}: Distribution", fontweight='bold')
        ax.set_xlabel(feature)
        ax.legend()
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()

def plot_raw_scatter(dfs_dict, x_col="acc_mag", y_col="gyro_mag"):
 
    n = len(dfs_dict)
    fig, axes = plt.subplots(nrows=1, ncols=n, figsize=(6*n, 6))
    if n == 1: axes = [axes]

    for i, (label, df) in enumerate(dfs_dict.items()):
        ax = axes[i]
        

        sc = ax.scatter(df[x_col], df[y_col], 
                        c=df["t_s"], cmap="viridis", alpha=0.5, s=20)
        
        ax.set_title(f"{label}: {x_col} vs {y_col}", fontweight='bold')
        ax.set_xlabel(x_col)
        if i == 0: ax.set_ylabel(y_col)
        ax.grid(True, linestyle='--', alpha=0.3)
        
     
        if i == n - 1:
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label("Time (s)")

    plt.tight_layout()
    plt.show()



def plot_session(df, title="Session", show_gaps=True, max_seconds=None):

    plot_df = df.copy()
    if max_seconds is not None:
        plot_df = plot_df[plot_df["t_s"] <= max_seconds]

    # Plot Accel
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(plot_df["t_s"], plot_df["acc_mag"], label="Acc Magnitude", color='tab:blue')
    if show_gaps and "is_gap" in plot_df.columns:
        gap_x = plot_df.loc[plot_df["is_gap"], "t_s"]
        for gt in gap_x:
            ax.axvline(gt, color='r', linestyle="--", alpha=0.5)
    ax.set_title(f"{title} - Accelerometer")
    ax.set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

    # Plot Gyro
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(plot_df["t_s"], plot_df["gyro_mag"], label="Gyro Magnitude", color='tab:orange')
    ax.set_title(f"{title} - Gyroscope")
    ax.set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

def plot_rolling(df, col_raw, col_mean, col_std, title="Rolling Stats"):

    plt.figure(figsize=(10, 4))
    plt.plot(df["t_s"], df[col_raw], alpha=0.3, label="Raw")
    plt.plot(df["t_s"], df[col_mean], linewidth=2, label="Rolling Mean")
    plt.plot(df["t_s"], df[col_std], linewidth=2, label="Rolling Std")
    plt.xlabel("Time (s)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# feature visualization



sns.set_style("whitegrid")
COLORS = sns.color_palette("husl", 5) 

def plot_hist_compare(df, feature, label_col="label", bins=30, density=True):
 
    plt.figure(figsize=(8, 5))
    
   
    labels = df[label_col].unique()
    
    for i, lab in enumerate(sorted(labels)):
        subset = df[df[label_col] == lab][feature].dropna()
        plt.hist(subset, bins=bins, alpha=0.5, density=density, 
                 label=f"{label_col}={lab}", color=COLORS[i % len(COLORS)])

    ylabel = "Density (Normalized)" if density else "Count"
    plt.xlabel(feature)
    plt.ylabel(ylabel)
    plt.title(f"Distribution of {feature} by {label_col}")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_boxplot_stat(df, feature, label_col="label"):
    
    # boxplot
    plt.figure(figsize=(6, 5))
    sns.boxplot(x=label_col, y=feature, data=df, palette="Set2")
    plt.title(f"Statistical Comparison: {feature}")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.show()

def scatter_features(df, x, y, group_col="label"):
    
    plt.figure(figsize=(8, 6))
    
    
    sns.scatterplot(data=df, x=x, y=y, hue=group_col, style=group_col, palette="deep", alpha=0.7)
    
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"Feature Space: {x} vs {y} (grouped by {group_col})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_3d_feature_space(df, x, y, z, label_col="activity"):
    """
    Plot 3D scatter of features to visualize class separation.
    
    Args:
        df: DataFrame containing features.
        x, y, z: Column names for the 3 axes.
        label_col: Column name to use for coloring points.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')


    # Red for Fall (alert), Blue for Safe (Non-Fall/ADL), others mixed
    color_map = {
        'Fall': '#d62728',       
        'Non-Fall': '#1f77b4',   
        'Walking': '#1f77b4',    
        'Sit-Stand': '#2ca02c',  
        'Standing': '#ff7f0e',  
    }
    

    unique_labels = sorted(df[label_col].unique())
    
    # Plot each category separately to handle legends correctly
    for label in unique_labels:
        subset = df[df[label_col] == label]
        
        # Determine styling based on label importance
        # "Fall" should be prominent (bigger, opaque)
        # "Non-Fall" can be background (smaller, transparent)
        is_fall = label == "Fall"
        
        ax.scatter(
            subset[x], subset[y], subset[z],
            c=color_map.get(label, 'gray'), # Default to gray if unknown
            label=label,
            alpha=0.8 if is_fall else 0.3,  # Transparency
            s=40 if is_fall else 20,        # Size
            edgecolors='w' if is_fall else None,
            linewidth=0.5
        )

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    ax.set_title(f"3D Feature Space: {x} vs {y} vs {z}", fontsize=14, fontweight='bold')
    
    ax.legend()
    plt.tight_layout()
    plt.show()



def plot_segmentation(df, events, title="Event Segmentation", signal_col="acc_mag"):
 
    plt.figure(figsize=(12, 4))
    
    plt.plot(df["t_s"], df[signal_col], label=signal_col, color='#1f77b4', linewidth=1)
    
    # the detected events as shaded regions
    for i, (start, end) in enumerate(events):
        plt.axvspan(start, end, color='green', alpha=0.3, label="Detected Event" if i==0 else "")
        
    plt.title(f"{title} (Found {len(events)} events)")
    plt.xlabel("Time (s)")
    plt.ylabel("Magnitude")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()
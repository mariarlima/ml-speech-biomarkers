import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from mlcog.utils import plotting


def plot_hist(results):
    with plotting.paper_theme():
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(
            data=results.groupby('pid').mean().reset_index(),
            y='prob',
            x='mmse',
            bins=(15, 10),
            ax=ax,
        )

        ax.axvline(x=26, linestyle='--', color='darkblue')
        ax.axvline(x=20, linestyle='--', color='darkblue')
        ax.axvline(x=10, linestyle='--', color='darkblue')

        # Calculate y position as a percentage of y-axis range
        y_min, y_max = ax.get_ylim()
        y_pos = y_max - 0.02 * (y_max - y_min)  # X% above the bottom of the y-axis

        ax.text(4, y_pos, 'Severe', fontsize=8, color='black')
        ax.text(10.5, y_pos, 'Moderate', fontsize=8, color='black')
        ax.text(20.5, y_pos, 'Mild', fontsize=8, color='black')
        ax.text(26.5, y_pos, 'CN', fontsize=8, color='black')

        ax.set_xlabel('MMSE Score')
        ax.set_ylabel('Positive Probability')

        fig.colorbar(ax.collections[0], ax=ax, label='Number of Participants', shrink=0.8)
        plt.show()


def plot_calib(df):
    prob_bin_freq = 0.1
    prob_bins = np.arange(0, 1.0001, prob_bin_freq)
    prob_bin_centres = np.arange(0, 1, prob_bin_freq)+prob_bin_freq/2

    binned_probs_mmse = (
        df
        .assign(
            binned_probs=lambda x: pd.cut(
                x.probs, 
                bins=np.arange(0, 1.0001, prob_bin_freq), 
                labels=np.arange(0, 1, prob_bin_freq)+prob_bin_freq/2
            )
        )
        .groupby(
            ['binned_probs', 'run']
        )
        [['mmse']]
        .mean()
        .reset_index()
        .dropna()
    )
    with plotting.paper_theme():
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.lineplot(
            data=binned_probs_mmse,
            x='binned_probs',
            y='mmse',
            palette='viridis',
            ax=ax,
            marker="o",
            err_style='bars',
            err_kws=dict(
                capsize=3
            ),
        )
        
        sns.lineplot(
            data=binned_probs_mmse,
            x='binned_probs',
            y='mmse',
            estimator=None,
            units='run',
            ax=ax,
            err_style='bars',
            color='grey',
            alpha=0.2,
            )

        ax.set_xlabel('Mean Positive Probability')
        ax.set_ylabel('Mean MMSE')
    
    with plotting.paper_theme():
        fig, ax = plt.subplots(figsize=(6, 4))

        mean_prob = binned_probs_mmse.groupby('binned_probs').mmse.mean()
        std_prob = binned_probs_mmse.groupby('binned_probs').mmse.std().fillna(0)

        na_idx = ~mean_prob.isna().values
    
        plt.bar(
            prob_bin_centres,
            mean_prob,
            fill=True,
            label='Mean',
            color='xkcd:lilac',
            alpha=0.75,
            edgecolor='xkcd:purple',
            width=prob_bin_freq,
            linewidth=0.5,
        )

        ax.set_xlabel('Mean Positive Probability')
        ax.set_ylabel('Mean MMSE')

        ax.errorbar(
            prob_bin_centres[na_idx],
            mean_prob.iloc[na_idx],
            yerr=std_prob[na_idx],
            fmt='o',
            capsize=3,
            label='Std',
            c='xkcd:violet',
        )
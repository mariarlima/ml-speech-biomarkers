import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib import transforms

# ---- Module constants -------------------------------------------------------

DEFAULT_POINT_ALPHA = 0.75
DEFAULT_CMAP = "winter"
FEATURE_VALUE_QUANTILES = (0.025, 0.975)

# Palette with slight transparency
COLORS = ['xkcd:cerulean', "#00B2EE", "#88CCEE", "mediumaquamarine"]
COLORS = [mcolors.to_rgba(c, alpha=0.97) for c in COLORS]


# ---- Utilities --------------------------------------------------------------

def update_with_defaults(kwargs_dict, default_dict):
    """Return a copy of kwargs_dict filled with missing keys from default_dict."""
    out = kwargs_dict.copy()
    for k, v in default_dict.items():
        if k not in out:
            out[k] = v
    return out


# ---- SHAP plotting ----------------------------------------------------------

class ShapDisplay:
    @staticmethod
    def from_shap_values(shap_values_df, n_features_plot=15, ax=None):
        """
        Create a scatter plot of SHAP values colored by (normalized) feature value.

        Expected columns in shap_values_df:
            - 'Feature'
            - 'SHAP Value'
            - 'Value' (normalized feature value for coloring)
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))

        # Order features by mean absolute SHAP value
        feature_order = (
            shap_values_df.groupby("Feature")["SHAP Value"]
            .apply(lambda x: np.abs(x).mean())
            .sort_values(ascending=False)
            .index
        )

        # Trim extreme feature values (winsorize by quantiles)
        v_lo, v_hi = FEATURE_VALUE_QUANTILES
        values = shap_values_df["Value"]
        lo = values.quantile(v_lo)
        hi = values.quantile(v_hi)
        plot_df = shap_values_df[(values > lo) & (values < hi)].copy()

        # Keep top features and sort by global importance order
        top_features = set(feature_order[:n_features_plot])
        plot_df = plot_df[plot_df["Feature"].isin(top_features)]
        plot_df = plot_df.sort_values(
            "Feature",
            key=lambda s: s.map({f: i for i, f in enumerate(feature_order)}),
        )

        # Scatter plot
        ax = sns.scatterplot(
            data=plot_df,
            x="SHAP Value",
            y="Feature",
            hue="Value",
            palette=DEFAULT_CMAP,
            legend=False,
            s=5,
            ax=ax,
            alpha=DEFAULT_POINT_ALPHA,
            edgecolor=None,
        )

        # Jitter vertically to reduce overplotting
        pts = ax.collections[0]
        offsets = pts.get_offsets()
        jitter = np.random.normal(0, 0.1, size=len(plot_df))
        pts.set_offsets(offsets + np.c_[np.zeros(len(plot_df)), jitter])

        # Axes & colorbar
        ax.margins(y=0.1)
        ax.autoscale_view()
        ax.yaxis.grid(True)
        ax.set_ylabel("")
        ax.set_xlabel("SHAP Value (%)")
        ax.axvline(0, color="black", linestyle="--", linewidth=1, zorder=0)

        norm = mcolors.Normalize(plot_df["Value"].min(), plot_df["Value"].max())
        sm = plt.cm.ScalarMappable(cmap=DEFAULT_CMAP, norm=norm)
        sm.set_array([])
        cbar = ax.figure.colorbar(sm, ax=ax, pad=0.01, shrink=0.66)
        cbar.outline.set_linewidth(0)
        cbar.set_alpha(DEFAULT_POINT_ALPHA)
        cbar.set_label("Normalised Feature Value")
        cbar.ax.tick_params(axis="both", which="major", pad=1, length=2, width=0.5, right=True)
        cbar.set_ticks([plot_df["Value"].min(), plot_df["Value"].max()])
        cbar.set_ticklabels(["Low", "High"])

        ax.tick_params(axis="both", which="major", pad=1, length=2, width=0.5, bottom=True, left=True)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=11)

        return ax


# ---- Waterfall plot ---------------------------------------------------------

def waterfallplot(
    data,
    x,
    y,
    order=None,
    base=0.0,
    orient="h",
    estimator="sum",
    cmap=None,
    alpha=0.75,
    positive_colour=COLORS[0],
    negative_colour=COLORS[3],
    width=0.8,
    bar_label=True,
    ax=None,
    arrow_kwargs=None,
    bar_kwargs=None,
    bar_label_kwargs=None,
):
    """
    Draw a waterfall-style bar/arrow plot for aggregated category contributions.

    Parameters mirror your original API; typing annotations removed per your style.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 8))

    arrow_kwargs = {} if arrow_kwargs is None else arrow_kwargs
    bar_kwargs = {} if bar_kwargs is None else bar_kwargs
    bar_label_kwargs = {} if bar_label_kwargs is None else bar_label_kwargs

    horizontal = orient == "h"
    categories = y if horizontal else x
    values = x if horizontal else y

    unique_categories = data[categories].unique()
    order = unique_categories if order is None else order

    # Aggregate values by category and align to order
    grouped = data.groupby(categories)[values].agg(estimator)
    bar_labels = grouped.loc[order].index.to_numpy()
    bar_values = grouped.loc[order].to_numpy()

    # Collect any categories missing from the specified order
    missing = unique_categories[~np.isin(unique_categories, order)]
    if len(missing) > 0:
        bar_labels = np.concatenate([bar_labels, ["Other"]])
        bar_values = np.concatenate([bar_values, [grouped.loc[missing].sum()]])

    pos_bar = np.arange(len(bar_labels))
    height = bar_values
    bottom = np.concatenate([[base], base + np.cumsum(bar_values)])[:-1]
    ends = bottom + bar_values

    if horizontal:
        height = height[::-1]
        bottom = bottom[::-1]
        ends = ends[::-1]
        bar_labels = bar_labels[::-1]

    # Invisible bars to compute geometry for arrows/labels
    if horizontal:
        bars = ax.barh(y=pos_bar, height=width, left=bottom, alpha=0, width=height, **bar_kwargs)
    else:
        bars = ax.bar(pos_bar, height=height, bottom=bottom, alpha=0, width=width, **bar_kwargs)

    # Axis limits with padding
    min_value = min(base, ends.min())
    max_value = max(base, ends.max())
    rng = max_value - min_value
    pad = 0.1 * rng
    if horizontal:
        ax.set_xlim(min_value - pad, max_value + pad)
    else:
        ax.set_ylim(min_value - pad, max_value + pad)

    # Determine colors: palette or +/- colors
    bar_boxes = [r.get_bbox().get_points() for r in bars.patches]
    if cmap is not None:
        colors = sns.color_palette(cmap, len(pos_bar))
    else:
        colors = []
        for b in bar_boxes:
            delta = (b[1, 0] - b[0, 0]) if horizontal else (b[1, 1] - b[0, 1])
            colors.append(positive_colour if delta > 0 else negative_colour)

    # Draw arrows
    arrow_defaults = {
        "head_width": width,
        "length_includes_head": True,
        "alpha": alpha,
        "linewidth": 0,
        "width": width,
        "head_length": max(rng * 0.025, 1e-6),
    }
    arrow_kwargs = update_with_defaults(arrow_kwargs, arrow_defaults)

    for nb, b in enumerate(bar_boxes):
        if horizontal:
            x_bar = b[0, 0]
            y_bar = pos_bar[nb]
            dx = b[1, 0] - b[0, 0]
            dy = 0
        else:
            x_bar = pos_bar[nb]
            y_bar = b[0, 1]
            dx = 0
            dy = b[1, 1] - b[0, 1]

        ax.arrow(x_bar, y_bar, dx, dy, color=colors[nb], **arrow_kwargs)

    # Labels & ticks
    if bar_label:
        bar_label_defaults = {"fmt": "%.2f", "label_type": "center", "padding": 0}
        ax.bar_label(bars, **update_with_defaults(bar_label_kwargs, bar_label_defaults))

    if horizontal:
        ax.set_yticks(pos_bar)
        ax.set_yticklabels(bar_labels)
    else:
        ax.set_xticks(pos_bar)
        ax.set_xticklabels(bar_labels)

    ax.set_ylabel(y)
    ax.set_xlabel(x)
    ax.grid(False, axis="y" if horizontal else "x")

    return ax


# ---- Risk feature plot ------------------------------------------------------

def risk_feature_plot(shap_values_point_df, base, ax=None):
    """
    Waterfall-style plot of top features' risk contributions plus final risk marker.

    Expected columns:
        - 'Feature', 'Risk', 'Value Norm'
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    order = (
        shap_values_point_df.groupby("Feature")["Risk"]
        .sum()
        .abs()
        .sort_values(ascending=False)
        .index[:5]
    )

    # Labels for bar annotations (reverse to match horizontal orientation)
    bar_labels = (
        shap_values_point_df.set_index("Feature")
        .loc[order]["Value Norm"]
        .values[::-1]
    )

    ax = waterfallplot(
        shap_values_point_df,
        y="Feature",
        x="Risk",
        order=order,
        base=base,
        orient="h",
        cmap=None,
        estimator="sum",
        positive_colour=COLORS[0],
        negative_colour=COLORS[1],
        width=0.8,
        bar_label=True,
        ax=ax,
        arrow_kwargs={},
        bar_kwargs={},
        bar_label_kwargs={
            "fmt": "%.0f %%",
            "labels": [""] + [f"+{x:.1f}" if x >= 0 else f"{x:.1f}" for x in bar_labels],
        },
    )

    proba_value = shap_values_point_df["Risk"].sum() + base
    ax.axvline(proba_value, color="black", linestyle="--", linewidth=0.5, zorder=0)

    text_trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(
        proba_value * 0.99,
        0.0,
        f"Final Risk: {proba_value:.0f}%",
        ha="right",
        va="center",
        transform=text_trans,
        bbox=dict(facecolor="white", edgecolor="black"),
        fontsize=12,
    )

    ax.grid(True, axis="y")
    ax.xaxis.label.set_size(13.5)
    ax.tick_params(axis="y", labelsize=13.5)
    ax.tick_params(axis="x", labelsize=11)
    ax.set_ylabel("")

    return ax


# ---- Data shaping helpers ---------------------------------------------------

def list_of_array_to_df_with_melt(x, feature_names, value_name):
    """
    Convert list of arrays (runs) into a long DataFrame with melt,
    including a 'run' and 'Point' index per row.
    """
    frames = []
    for r, x_r in enumerate(x):
        df = pd.DataFrame(x_r, columns=feature_names)
        df = df.assign(Point=lambda d: np.arange(d.shape[0]))
        df = df.melt(id_vars=["Point"], var_name="Feature", value_name=value_name)
        df = df.assign(run=r)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def list_of_array_to_df(x, feature_name):
    """
    Convert list of 1-D arrays (runs) into a stacked DataFrame with
    columns [feature_name, Point, run].
    """
    frames = []
    for r, x_r in enumerate(x):
        df = pd.DataFrame(x_r, columns=[feature_name])
        df = df.assign(Point=lambda d: np.arange(d.shape[0]), run=r)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

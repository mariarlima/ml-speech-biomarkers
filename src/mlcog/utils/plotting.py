import matplotlib.pyplot as plt
import seaborn as sns
import contextlib
import matplotlib
import numpy as np
import pandas as pd
from cycler import cycler
import typing as t


def save_fig(fig: plt.figure, file_name: str, **kwargs) -> None:
    """
    This function saves a pdf, png, and svg of the figure,
    with :code:`dpi=300`.


    Arguments
    ---------

    - fig: plt.figure:
        The figure to save.

    - file_name: str:
        The file name, including path, to save the figure at.
        This should not include the extension, which will
        be added when each file is saved.

    """

    fig.savefig(f"{file_name}.pdf", **kwargs)
    fig.savefig(f"{file_name}.png", dpi=300, **kwargs)
    fig.savefig(f"{file_name}.svg", **kwargs)


# colours
tol_muted = [
    "#332288",
    "#88CCEE",
    "#44AA99",
    "#117733",
    "#999933",
    "#DDCC77",
    "#CC6677",
    "#882255",
    "#AA4499",
]

ibm = [
    "#648fff",
    "#fe6100",
    "#dc267f",
    "#785ef0",
    "#ffb000",
]


# colour map
def set_colour_map(colours: list = tol_muted):
    """
    Sets the default colour map for all plots.



    Examples
    ---------

    The following sets the colourmap to :code:`tol_muted`:

    .. code-block::

        >>> set_colour_map(colours=tol_muted)


    Arguments
    ---------

    - colours: list, optional:
        Format that is accepted by
        :code:`cycler.cycler`.
        Defaults to :code:`tol_muted`.

    """
    custom_params = {"axes.prop_cycle": cycler(color=colours)}
    matplotlib.rcParams.update(**custom_params)


# context functions
@contextlib.contextmanager
def temp_colour_map(colours=tol_muted):
    """
    Temporarily sets the default colour map for all plots.


    Examples
    ---------

    The following sets the colourmap to :code:`tol_muted` for
    the plotting done within the context:

    .. code-block::

        >>> with set_colour_map(colours=tol_muted):
        ...     plt.plot(x,y)


    Arguments
    ---------

    - colours: list, optional:
        Format that is accepted by
        :code:`cycler.cycler`.
        Defaults to :code:`tol_muted`.

    """
    set_colour_map(colours=colours)


@contextlib.contextmanager
def paper_theme(colours: t.List[str] = ibm):
    """
    Temporarily sets the default theme for all plots.


    Examples
    ---------

    .. code-block::

        >>> with paper_theme():
        ...     plt.plot(x,y)


    Arguments
    ---------

    - colours: t.List[str], optional:
        Any acceptable list to :code:`cycler`.
        Defaults to :code:`ibm`.


    """
    with matplotlib.rc_context():
        sns.set(context="paper", style="whitegrid")
        custom_params = {
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.edgecolor": "black",
            "axes.linewidth": 1,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.prop_cycle": cycler(color=colours),
            "grid.alpha": 0.5,
            "grid.color": "#b0b0b0",
            "grid.linestyle": "--",
            "grid.linewidth": 1,
            "xtick.major.width": 1,
            "ytick.major.width": 1,
            "boxplot.whiskerprops.linestyle": "-",
            "boxplot.whiskerprops.linewidth": 1,
            "boxplot.whiskerprops.color": "black",
            "boxplot.boxprops.linestyle": "-",
            "boxplot.boxprops.linewidth": 1,
            "boxplot.boxprops.color": "black",
            "boxplot.meanprops.markeredgecolor": "black",
            "boxplot.capprops.color": "black",
            "boxplot.capprops.linestyle": "-",
            "boxplot.capprops.linewidth": 1.0,
        }

        matplotlib.rcParams.update(**custom_params)

        yield


class ReliabilityDisplay:
    """
    Class to plot a reliability diagram from predictions.


    Examples
    ---------

    This can be used to plot the reliability diagram from predictions
    made by a classifier, such as:

    .. code-block::

        >>> y_prob = clf.predict_proba(X)
        >>> ax = avt.ReliabilityDisplay.from_predictions(
        ...     y,
        ...     y_prob,
        ... )

    .. image:: figures/ReliabilityDisplay_from_predictions.png
        :width: 600
        :align: center
        :alt: Alternative text

    Or from the classifier directly:

    .. code-block::

        >>> ax = avt.ReliabilityDisplay.from_estimator(
        ...     clf, X, y
        ... )

    .. image:: figures/ReliabilityDisplay_from_estimator.png
        :width: 600
        :align: center
        :alt: Alternative text



    """

    def _bin_stats(
        y_true: np.ndarray, probas_pred: np.ndarray, n_bins: int = 15
    ) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """

        Calculate the some bin statistics that are
        useful for calibration error analysis.


        Arguments
        ---------
        - y_true : numpy.ndarray:
            True labels. Please note that this vector should only contain
            integers that correspond to the probabilities in the second
            dimension of :code:`probas_pred`.

        - probas_pred : numpy.ndarray:
            Predicted probabilities, as returned by a classifier's
            :code:`predict_proba` method. This array should have the shape
            :code:`(n_samples, n_classes)`.

        - n_bins : int:
            The number of bins to use when calculating ECE.
            Defaults to :code:`15`.


        Returns
        -------
        - bin_counts: numpy.ndarray:
            The number of predictions in each bin.

        - bin_acc: numpy.ndarray:
            The accuracy of the predictions in each bin.

        - bin_avg_confidence:
            The average confidence of the predictions in each bin.

        """
        # converting to numpy arrays
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        if isinstance(probas_pred, pd.Series):
            probas_pred = probas_pred.values

        try:
            y_true = np.array(y_true).astype(int).reshape(-1)
        except ValueError:
            raise ValueError(
                f"y_true should only contain integers that correspond "
                "to the probabilities in the second dimension of probas_pred."
            )

        n_classes = len(np.unique(y_true))

        # ensure that the probas_pred has the correct shape
        if probas_pred.ndim != 2:
            raise ValueError(f"probas_pred should be of dimension 2.")

        if probas_pred.shape[1] != n_classes:
            raise ValueError(
                f"probas_pred should contain probabilities for {n_classes} classes."
            )

        y_pred = np.argmax(probas_pred, axis=1).reshape(-1)
        y_confidence = probas_pred[np.arange(len(probas_pred)), y_pred].reshape(-1)

        # an array of the bin lower bounds
        # for example with n_bins=3, bins=[0.333..., 0.666..., 1.        ]
        bins = np.linspace(0, 1, n_bins + 1)[1:]

        # comparing predictions to bins to get their bin positions
        # for example: array([1, 2, 2, 1, 1, 2, 0, 2])
        probas_pred_bins = np.digitize(y_confidence, bins)

        y_correct = y_true == y_pred

        # getting the counts of correct predictions per bin
        bin_correct_counts = np.bincount(probas_pred_bins, weights=y_correct)

        # getting the counts of total predictions per bin
        bin_total_counts = np.bincount(probas_pred_bins)

        # getting the mean confidence per bin
        # if there are no predictions in a bin, the mean confidence is 0
        bin_avg_confidence = np.divide(
            np.bincount(probas_pred_bins, weights=y_confidence),
            bin_total_counts,
            out=np.zeros_like(bin_correct_counts),
            where=bin_total_counts != 0,
        )

        # getting the accuracy per bin
        # if there are no predictions in a bin, the accuracy is 0
        bin_acc = np.divide(
            bin_correct_counts,
            bin_total_counts,
            out=np.zeros_like(bin_correct_counts),
            where=bin_total_counts != 0,
        )

        # getting the number of predictions per bin
        bin_counts = np.bincount(probas_pred_bins)

        return bin_counts, bin_acc, bin_avg_confidence

    def _reliability_bars(
        bin_counts,
        bin_acc,
        bin_avg_confidence,
        legend,
        accuracy_color,
        gap_color,
        diagonal_color,
        bar_kwargs,
        line_kwargs,
        ax,
    ):

        n_bins = len(bin_counts)

        bins = np.linspace(0, 1, n_bins + 1, endpoint=True)[1:]

        x_values = np.arange(len(bins))

        if "edgecolor" not in bar_kwargs:
            bar_kwargs["edgecolor"] = "black"

        if "linewidth" not in bar_kwargs:
            bar_kwargs["linewidth"] = 1

        # if "linewidth" not in line_kwargs:
        #    line_kwargs["linewidth"] = 1

        if "linestyle" not in line_kwargs:
            line_kwargs["linestyle"] = "--"

        ax.bar(
            x=x_values * 1 / n_bins,
            height=bin_acc,
            align="edge",
            width=1 / n_bins,
            color=accuracy_color,
            label="Accuracy",
            **bar_kwargs,
        )

        ax.bar(
            x=x_values * 1 / n_bins,
            height=bin_avg_confidence - bin_acc,
            bottom=bin_acc,
            color=gap_color,
            align="edge",
            width=1 / n_bins,
            label="Gap",
            **bar_kwargs,
        )

        ax.axline((0, 0), (1, 1), color=diagonal_color, **line_kwargs)

        if legend:
            ax.legend()

        loc = matplotlib.ticker.MaxNLocator(
            nbins="auto", steps=[1, 2, 4, 10], integer=False
        )
        ax.xaxis.set_major_locator(loc)

        ax.set_xlim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Confidence")
        # ax.set_title("Reliability Diagram")

        return ax

    def _histogram(
        y_confidence,
        n_bins,
        accuracy_avg,
        confidence_avg,
        hist_color,
        histogram_kwargs,
        label_box_kwargs,
        label_ys,
        label_offset,
        accuracy_line_color,
        confidence_line_color,
        ax,
    ):

        ax.hist(
            y_confidence,
            bins=np.linspace(0, 1, n_bins + 1, endpoint=True),
            color=hist_color,
            **histogram_kwargs,
        )

        ax.set_xlim(0, 1)
        loc = matplotlib.ticker.MaxNLocator(
            nbins="auto", steps=[1, 2, 4, 10], integer=False
        )
        ax.xaxis.set_major_locator(loc)

        ax.set_ylabel("Count")
        ax.set_xlabel("Confidence")

        if "boxstyle" not in label_box_kwargs:
            label_box_kwargs["boxstyle"] = "square"

        if "fc" not in label_box_kwargs:
            label_box_kwargs["fc"] = "w"

        if "alpha" not in label_box_kwargs:
            label_box_kwargs["alpha"] = 0.8

        ax.axvline(
            x=accuracy_avg,
            color=accuracy_line_color,
            linestyle="-",
        )

        ax.text(
            accuracy_avg + label_offset,
            label_ys[0],
            f"Avg Accy: {accuracy_avg:.2f}",
            transform=ax.get_xaxis_transform(),
            ha="right",
            va="center",
            bbox=dict(ec=accuracy_line_color, **label_box_kwargs),
        )

        ax.axvline(
            x=confidence_avg,
            color=confidence_line_color,
            linestyle="--",
        )

        ax.text(
            confidence_avg + label_offset,
            label_ys[1],
            f"Avg Conf: {confidence_avg:.2f}",
            transform=ax.get_xaxis_transform(),
            ha="right",
            va="center",
            bbox=dict(
                ec=confidence_line_color,
                linestyle="--",
                **label_box_kwargs,
            ),
        )

        return ax

    def _set_same_xaxis(axes: t.Dict[str, plt.Axes]):

        # same ticks
        xticks = axes["A"].get_xticks()
        axes["B"].set_xticks(xticks)

        # remove ticks and labels from plot A
        axes["A"].set_xticklabels([])
        axes["A"].set_xlabel("")

        # same limits
        axes["A"].set_xlim(0, 1)
        axes["B"].set_xlim(0, 1)

        return axes

    def from_predictions(
        y_true: np.ndarray,
        probas_pred: np.ndarray,
        n_bins: int = 10,
        legend: bool = True,
        accuracy_color="xkcd:baby blue",
        gap_color="xkcd:rose",
        hist_color="xkcd:lilac",
        diagonal_color="xkcd:dark red",
        accuracy_line_color="black",
        confidence_line_color="black",
        bar_kwargs: dict = {},
        histogram_kwargs: dict = {},
        line_kwargs: dict = {},
        label_box_kwargs: dict = {},
        label_ys: t.Tuple[float, float] = (0.8, 0.6),
        label_offset: float = -0.01,
        ax1: t.Optional[plt.Axes] = None,
        ax2: t.Optional[plt.Axes] = None,
    ) -> np.ndarray:
        """
        Function to plot the figure from the predictions.

        Arguments
        ---------
        - y_true : numpy.ndarray:
            True labels. Please note that this vector should only contain
            integers that correspond to the probabilities in the second
            dimension of :code:`probas_pred`.

        - probas_pred : numpy.ndarray:
            Predicted probabilities, as returned by a classifier's
            :code:`predict_proba` method. This array should have the shape
            :code:`(n_samples, n_classes)`.

        - n_bins : int:
            The number of bins to use when calculating ECE.
            Defaults to :code:`15`.

        - legend : bool:
            Whether to plot the legend.
            Defaults to :code:`True`.

        - accuracy_color : str:
            The colour to use for the accuracy bars.
            Defaults to :code:`xkcd:baby blue`.

        - gap_color : str:
            The colour to use for the gap bars.
            Defaults to :code:`xkcd:rose`.

        - hist_color : str:
            The colour to use for the histogram.
            Defaults to :code:`xkcd:lilac`.

        - diagonal_color : str:
            The colour to use for the diagonal line.
            Defaults to :code:`xkcd:dark red`.

        - accuracy_line_color : str:
            The colour to use for the accuracy line.
            Defaults to :code:`black`.

        - confidence_line_color : str:
            The colour to use for the confidence line.
            Defaults to :code:`black`.

        - bar_kwargs : dict:
            Keyword arguments to pass to the bar plot.
            Defaults to :code:`{}`.

        - histogram_kwargs : dict:
            Keyword arguments to pass to the histogram plot.
            Defaults to :code:`{}`.

        - line_kwargs : dict:
            Keyword arguments to pass to the line plot.
            Defaults to :code:`{}`.

        - label_box_kwargs : dict:
            Keyword arguments to pass to the label :code:`bbox` in the
            histogram plot.
            Defaults to :code:`{}`.

        - label_ys : tuple:
            The y coordinates of the labels in the histogram plot.
            Defaults to :code:`(0.8, 0.6)`.

        - label_offset : float:
            The offset of the labels in the histogram plot.

        - ax1 : matplotlib.axes.Axes:
            The axes to plot the reliability diagram on.
            Defaults to :code:`None`.

        - ax2 : matplotlib.axes.Axes:
            The axes to plot the histogram on.
            Defaults to :code:`None`.


        Returns
        -------
        - axes : numpy.ndarray:
            The axes of the plots.

        """

        # getting the bin statistics
        bin_counts, bin_acc, bin_avg_confidence = ReliabilityDisplay._bin_stats(
            y_true, probas_pred, n_bins=n_bins
        )

        y_pred = np.argmax(probas_pred, axis=1).reshape(-1)
        y_confidence = probas_pred[np.arange(len(probas_pred)), y_pred].reshape(-1)

        if ax1 is None and ax2 is None:
            fig, axes = plt.subplot_mosaic(
                """
                AA
                AA
                BB
                """,
                figsize=(4, 6),
            )

        elif (ax1 is not None) and (ax2 is not None):
            axes = {"A": ax1, "B": ax2}

        else:
            raise ValueError("Either both or neither of ax1 and ax2 must be None.")

        ReliabilityDisplay._reliability_bars(
            bin_counts=bin_counts,
            bin_acc=bin_acc,
            bin_avg_confidence=bin_avg_confidence,
            legend=legend,
            accuracy_color=accuracy_color,
            gap_color=gap_color,
            diagonal_color=diagonal_color,
            bar_kwargs=bar_kwargs,
            line_kwargs=line_kwargs,
            ax=axes["A"],
        )

        ReliabilityDisplay._histogram(
            y_confidence=y_confidence,
            accuracy_avg=(bin_acc * bin_counts).sum() / np.sum(bin_counts),
            confidence_avg=(bin_avg_confidence * bin_counts).sum() / np.sum(bin_counts),
            n_bins=n_bins,
            hist_color='xkcd:cerulean',
            histogram_kwargs=histogram_kwargs,
            label_box_kwargs=label_box_kwargs,
            label_ys=label_ys,
            label_offset=label_offset,
            accuracy_line_color=accuracy_line_color,
            confidence_line_color=confidence_line_color,
            ax=axes["B"],
        )

        axes = ReliabilityDisplay._set_same_xaxis(axes)

        return np.array([axes["A"], axes["B"]])

    def from_estimator(estimator, X, y, *args, **kwargs) -> np.ndarray:
        """
        Function to plot the reliability figure from an estimator.

        Arguments
        ---------
        - estimator : sklearn.base.BaseEstimator:
            The estimator to use to make predictions.

        - X : numpy.ndarray:
            The data to use to make predictions.

        - y : numpy.ndarray:
            The true labels.

        - *args :
            Positional arguments to pass to the :code:`from_predictions` method.

        - **kwargs :
            Keyword arguments to pass to the :code:`from_predictions` method.


        Returns
        -------
        - axes : numpy.ndarray:
            The axes of the plots.

        """

        if hasattr(estimator, "predict_proba"):
            probas_pred = estimator.predict_proba(X)
        else:
            raise AttributeError(
                f"{estimator.__class__.__name__} does not have a predict_proba method."
            )

        return ReliabilityDisplay.from_predictions(y, probas_pred, *args, **kwargs)

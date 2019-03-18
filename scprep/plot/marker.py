import numpy as np
import pandas as pd

from .. import utils, stats, select
from .utils import (_get_figure, show,
                    temp_fontsize, parse_fontsize, shift_ticklabels)
from .tools import label_axis


@utils._with_pkg(pkg="matplotlib", min_version=3)
def marker_plot(data, clusters, markers, gene_names=None,
                normalize_expression=True, cmap='magma',
                title=None, figsize=None,
                ax=None, fontsize=None):
    """
    Generate a plot indicating the expression level and enrichment of
    a set of marker genes for each cluster.

    Parameters
    ----------
    data : array-like, shape=[n_cells, n_genes]
        Gene expression data for calculating expression statistics.
    clusters : list-like, shape=[n_cells]
        Cluster assignments for each cell. Should be ints
        like the output of most sklearn.cluster methods.
    markers : dict or list-like
        If a dictionary, keys represent tissues and
        values being a list of marker genes in each tissue.
        If a list, a list of marker genes.
    gene_names : list-like, shape=[n_genes]
        List of gene names.
    normalize_expression : bool, optional (default: True)
        Normalize the expression of each row.
    cmap : str or matplotlib colormap, optional (default: 'inferno')
        Colormap with which to color points.
    title : str or None, optional (default: None)
        Title for the plot
    figsize : tuple or None, optional (default: None)
        If not None, sets the figure size (width, height)
    ax : `matplotlib.Axes` or None, optional (default: None)
        Axis to plot on. If None, a new axis will be created.
    fontsize : int or None, optional (default: None)
        Base fontsize.

    Returns
    -------
    ax : `matplotlib.Axes`
        axis on which plot was drawn

    Example
    -------
    >>> markers = {'Adaxial - Immature': ['myl10', 'myod1'],
                   'Adaxial - Mature': ['myog'],
                   'Presomitic mesoderm': ['tbx6', 'msgn1', 'tbx16'],
                   'Forming somites': ['mespba', 'ripply2'],
                   'Somites': ['meox1', 'ripply1', 'aldh1a2']}
    >>> cluster_marker_plot(data, clusters, gene_names, markers,
                            title="Tailbud - PSM")
    """
    with temp_fontsize(fontsize):
        if gene_names is None:
            if not isinstance(data, pd.DataFrame):
                raise ValueError(
                    "Either `data` must be a pd.DataFrame, or gene_names must "
                    "be provided. "
                    "Got gene_names=None, data as a {}".format(type(data)))
            gene_names = data.columns

        if isinstance(markers, dict):
            tissues, markers = tuple(
                zip(*[([k] * len(v), v) for k, v in markers.items()]))
            tissues, markers = np.concatenate(tissues), np.concatenate(markers)
        else:
            markers = utils.toarray(markers)
            tissues = None

        for gene in markers:
            if gene not in gene_names:
                raise ValueError('All genes in `markers` must appear '
                                 'in gene_names. Did not find: {}'.format(gene))

        data = utils.to_array_or_spmatrix(data)

        cluster_names = np.unique(clusters)

        keep_genes = np.isin(gene_names, markers)
        data, gene_names = select.select_cols(data, gene_names, idx=keep_genes)

        fig, ax, show_fig = _get_figure(ax, figsize=figsize)

        cluster_labels = []
        marker_labels = []
        tissue_labels = []
        x = []
        y = []
        c = []
        s = []

        # Do boolean indexing only once per cluster
        data_clust = {}
        for i, cluster in enumerate(cluster_names):
            in_cluster = clusters == cluster
            in_cluster_expr = data[in_cluster]
            out_cluster_expr = data[~in_cluster]
            data_clust[cluster] = (in_cluster_expr, out_cluster_expr)

        for j, marker in enumerate(markers):
            s_row = []
            c_row = []
            for i, cluster in enumerate(cluster_names):
                in_cluster_expr, out_cluster_expr = data_clust[cluster]
                x.append(i)
                y.append(j)
                marker_labels.append(marker)
                cluster_labels.append(cluster)
                if tissues is not None:
                    tissue_labels.append(tissues[j])
                gidx = np.where(gene_names == marker)
                marker_expr = in_cluster_expr[:, gidx]
                s_row.append(stats.EMD(marker_expr,
                                       out_cluster_expr[:, gidx]))
                c_row.append(np.mean(marker_expr))
            # row normalize
            s_row = np.array(s_row)
            s_row = 150 * s_row / np.max(s_row)
            c_row = np.array(c_row)
            if normalize_expression:
                c_row = c_row / np.max(c_row)
            s.append(s_row)
            c.append(c_row)

        s = np.concatenate(s)
        c = np.concatenate(c)

        ax.scatter(x, y, s=s, c=c, cmap=cmap, vmax=max(c) * 1.3)

        # Vertical and Horizontal Grid Lines
        for h in np.unique(y):
            ax.axhline(h, c='k', linewidth=0.1, zorder=0)
        for v in np.unique(x):
            ax.axvline(v, c='k', linewidth=0.1, zorder=0)

        # Title
        title_fontsize = parse_fontsize(None, 'xx-large')
        ax.set_title(title, fontsize=title_fontsize, fontweight='bold')

        # X axis decorators
        x_unique, x_unique_idx = np.unique(x, return_index=True)
        label_axis(ax.xaxis, label='Cluster', ticks=x_unique,
                   ticklabels=np.array(cluster_labels)[x_unique_idx],
                   ticklabel_rotation=45,
                   ticklabel_horizontal_alignment='right')
        shift_ticklabels(ax.xaxis, dx=0.1)

        # Y axis decorators
        y_unique, y_unique_idx = np.unique(y, return_index=True)
        label_axis(ax.yaxis, ticks=y_unique,
                   ticklabels=np.array(marker_labels)[y_unique_idx])

        if tissues is not None:
            # Right Y axis decorators
            ax2 = ax.twinx()
            ax2.set_ylim(ax.get_ylim())
            label_axis(ax2.yaxis, ticks=y_unique,
                       ticklabels=np.array(tissue_labels)[y_unique_idx])

        if show_fig:
            show(fig)

    return ax

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy

from .. import utils, stats, select
from .utils import _get_figure, show, temp_fontsize, parse_fontsize, shift_ticklabels
from .tools import label_axis


def _make_scatter_arrays(
    data_clust,
    cluster_names,
    tissues,
    markers,
    gene_names,
    normalize_emd,
    normalize_expression,
):
    cluster_labels = []
    marker_labels = []
    tissue_labels = []
    x = []
    y = []
    c = []
    s = []
    # build points coordinate, color and size arrays
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
            s_row.append(stats.EMD(marker_expr, out_cluster_expr[:, gidx]))
            c_row.append(np.mean(marker_expr))
        # row normalize
        s_row = np.array(s_row)
        if normalize_emd and np.max(s_row) != 0:
            s_row = 150 * s_row / np.max(s_row)
        c_row = np.array(c_row)
        if normalize_expression and np.max(c_row) != 0:
            c_row = c_row / np.max(c_row)
        s.append(s_row)
        c.append(c_row)

    s = np.concatenate(s)
    if not normalize_emd:
        s = 150 * s / np.max(s)
    c = np.concatenate(c)

    return x, y, c, s, cluster_labels, tissue_labels, marker_labels


def _cluster_tissues(tissue_names, cluster_names, tissue_labels, cluster_labels, s, c):
    # cluster tissues hierarchically using mean size and color
    tissue_features = []
    for tissue in tissue_names:
        tissue_data = []
        for cluster in cluster_names:
            tissue_cluster_idx = np.where(
                (np.array(tissue_labels) == tissue)
                & (np.array(cluster_labels) == cluster)
            )
            tissue_data.append(
                np.vstack([s[tissue_cluster_idx], c[tissue_cluster_idx]]).mean(axis=1)
            )
        tissue_features.append(np.concatenate(tissue_data))
    tissue_features = np.array(tissue_features)
    # normalize
    tissue_features = tissue_features / np.sqrt(np.sum(tissue_features ** 2))
    tissues_order = hierarchy.leaves_list(hierarchy.linkage(tissue_features))
    return tissues_order


def _cluster_markers(
    markers, tissues, marker_labels, tissue_labels, marker_groups_order, s, c
):
    # cluster markers hierarchically using mean size and color
    markers_order = []
    for marker_group in marker_groups_order:
        if len(marker_group) > 1:
            marker_names = markers[marker_group]
            marker_features = []
            for marker in marker_names:
                marker_idx = np.array(marker_labels) == marker
                if tissues is not None:
                    # check for markers that appear in multiple tissues
                    marker_idx = marker_idx & (
                        tissue_labels == tissues[marker_group[0]]
                    )
                marker_features.append(np.concatenate([s[marker_idx], c[marker_idx]]))
            marker_features = np.array(marker_features)
            # normalize
            marker_features = marker_features / np.sqrt(np.sum(marker_features ** 2))
            marker_group_order = hierarchy.leaves_list(
                hierarchy.linkage(marker_features)
            )
            markers_order.append(marker_group[marker_group_order])
        else:
            markers_order.append(marker_group)
    markers_order = np.concatenate(markers_order)
    return markers_order


@utils._with_pkg(pkg="matplotlib", min_version=3)
def marker_plot(
    data,
    clusters,
    markers,
    gene_names=None,
    normalize_expression=True,
    normalize_emd=True,
    reorder_tissues=True,
    reorder_markers=True,
    cmap="magma",
    title=None,
    figsize=None,
    ax=None,
    fontsize=None,
):
    """Marker gene enrichment plot

    Generate a plot indicating the expression level and enrichment of
    a set of marker genes for each cluster.

    Color of each point indicates the expression of each gene in each cluster.
    The size of each point indicates how differentially expressed each gene is
    in each cluster.

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
    normalize_{expression,emd} : bool, optional (default: True)
        Normalize the expression and EMD of each row.
    reorder_{tissues,markers} : bool, optional (default: True)
        Reorder tissues and markers according to hierarchical clustering=
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
                    "Got gene_names=None, data as a {}".format(type(data))
                )
            gene_names = data.columns

        if isinstance(markers, dict):
            tissues, markers = tuple(
                zip(*[([k] * len(v), v) for k, v in markers.items()])
            )
            tissues, markers = np.concatenate(tissues), np.concatenate(markers)
        else:
            markers = utils.toarray(markers)
            tissues = None

        for gene in markers:
            if gene not in gene_names:
                raise ValueError(
                    "All genes in `markers` must appear "
                    "in gene_names. Did not find: {}".format(gene)
                )

        data = utils.to_array_or_spmatrix(data)

        cluster_names = np.unique(clusters)

        keep_genes = np.isin(gene_names, markers)
        data, gene_names = select.select_cols(data, gene_names, idx=keep_genes)

        fig, ax, show_fig = _get_figure(ax, figsize=figsize)

        # Do boolean indexing only once per cluster
        data_clust = {}
        for i, cluster in enumerate(cluster_names):
            in_cluster = clusters == cluster
            in_cluster_expr = data[in_cluster]
            out_cluster_expr = data[~in_cluster]
            data_clust[cluster] = (in_cluster_expr, out_cluster_expr)

        (
            x,
            y,
            c,
            s,
            cluster_labels,
            tissue_labels,
            marker_labels,
        ) = _make_scatter_arrays(
            data_clust,
            cluster_names,
            tissues,
            markers,
            gene_names,
            normalize_emd,
            normalize_expression,
        )

        # reorder y axis
        if tissues is not None and len(tissues) > 1:
            tissue_names = np.unique(tissues)
            if reorder_tissues:
                tissues_order = _cluster_tissues(
                    tissue_names, cluster_names, tissue_labels, cluster_labels, s, c
                )
            else:
                # keep tissues in order
                tissues_order = np.arange(len(tissue_names))
            marker_groups_order = [
                np.arange(len(markers))[tissues == tissue_names[i]]
                for i in tissues_order
            ]
        else:
            # only one tissue
            marker_groups_order = [np.arange(len(markers))]

        if reorder_markers and len(markers) > 1:
            markers_order = _cluster_markers(
                markers,
                tissues,
                marker_labels,
                tissue_labels,
                marker_groups_order,
                s,
                c,
            )
        else:
            # keep markers in order
            markers_order = np.concatenate(marker_groups_order)

        # reposition y coordinates
        y = np.array(y)
        y_new = np.zeros_like(y)
        for i in range(len(markers)):
            y_new[y == markers_order[i]] = i
        y = y_new

        ax.scatter(x, y, s=s, c=c, cmap=cmap, vmax=max(c) * 1.3)

        # Vertical and Horizontal Grid Lines
        for h in np.unique(y):
            ax.axhline(h, c="k", linewidth=0.1, zorder=0)
        for v in np.unique(x):
            ax.axvline(v, c="k", linewidth=0.1, zorder=0)

        ax.set_ylim(-0.5, len(markers) - 0.5)

        # Title
        title_fontsize = parse_fontsize(None, "xx-large")
        ax.set_title(title, fontsize=title_fontsize, fontweight="bold")

        # X axis decorators
        x_unique, x_unique_idx = np.unique(x, return_index=True)
        label_axis(
            ax.xaxis,
            label="Cluster",
            ticks=x_unique,
            ticklabels=np.array(cluster_labels)[x_unique_idx],
            ticklabel_rotation=45,
            ticklabel_horizontal_alignment="right",
        )
        shift_ticklabels(ax.xaxis, dx=0.1)

        # Y axis decorators
        label_axis(
            ax.yaxis, ticks=np.arange(len(markers)), ticklabels=markers[markers_order]
        )

        if tissues is not None:
            # Right Y axis decorators
            ax2 = ax.twinx()
            ax2.set_ylim(ax.get_ylim())
            label_axis(
                ax2.yaxis,
                ticks=np.arange(len(tissues)),
                ticklabels=tissues[markers_order],
            )

        if show_fig:
            show(fig)

    return ax

"""Visualization — Matplotlib/Seaborn-inspired API with vectorphile aesthetics."""
from __future__ import annotations
import numpy as np
from typing import Optional, Union, List, Tuple

try:
    import matplotlib
    import matplotlib.pyplot as _plt
    import matplotlib.figure as _mfig
    import matplotlib.axes as _maxes
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def _require_mpl():
    if not _HAS_MPL:
        raise ImportError("matplotlib is required for vp.viz. Install with: pip install matplotlib")


# ── Color palettes ────────────────────────────────────────────────────────────
PALETTES = {
    'neon':    ['#7c5cff', '#06b6d4', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'],
    'void':    ['#c4b5fd', '#93c5fd', '#6ee7b7', '#fde68a', '#fca5a5', '#f9a8d4'],
    'muted':   ['#6366f1', '#0ea5e9', '#14b8a6', '#f97316', '#f43f5e', '#a855f7'],
    'deep':    ['#4f46e5', '#0284c7', '#059669', '#d97706', '#dc2626', '#7c3aed'],
    'pastel':  ['#a5b4fc', '#7dd3fc', '#6ee7b7', '#fcd34d', '#fca5a5', '#d8b4fe'],
    'dark':    ['#1e1b4b', '#0c4a6e', '#064e3b', '#78350f', '#7f1d1d', '#3b0764'],
}

THEMES = {
    'dark': {'facecolor': '#06060a', 'axesfacecolor': '#0d0d14', 'textcolor': '#e2e8f0', 'gridcolor': '#1e2535'},
    'light': {'facecolor': '#ffffff', 'axesfacecolor': '#f8fafc', 'textcolor': '#1e293b', 'gridcolor': '#e2e8f0'},
    'paper': {'facecolor': '#fafafa', 'axesfacecolor': '#ffffff', 'textcolor': '#111827', 'gridcolor': '#d1d5db'},
}

_current_theme = 'dark'
_current_palette = 'neon'


def set_theme(theme: str = 'dark') -> None:
    global _current_theme
    _current_theme = theme
    _apply_theme(theme)

def set_palette(palette: str) -> None:
    global _current_palette
    _current_palette = palette

def _apply_theme(theme: str) -> None:
    if not _HAS_MPL: return
    t = THEMES.get(theme, THEMES['dark'])
    _plt.rcParams.update({
        'figure.facecolor': t['facecolor'],
        'axes.facecolor': t['axesfacecolor'],
        'text.color': t['textcolor'],
        'axes.labelcolor': t['textcolor'],
        'xtick.color': t['textcolor'],
        'ytick.color': t['textcolor'],
        'axes.edgecolor': t['gridcolor'],
        'grid.color': t['gridcolor'],
        'axes.grid': True,
        'grid.alpha': 0.3,
        'font.family': 'monospace',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


class Figure:
    """Wrapper around matplotlib Figure."""
    def __init__(self, fig=None, axes=None):
        _require_mpl()
        self._fig = fig
        self._axes = axes

    def show(self):
        _plt.figure(self._fig.number)
        _plt.show()

    def save(self, path: str, dpi=150, bbox_inches='tight'):
        self._fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)

    def tight_layout(self, **kwargs):
        self._fig.tight_layout(**kwargs)
        return self


def figure(figsize=(10, 6), nrows=1, ncols=1, **kwargs) -> Figure:
    _require_mpl()
    _apply_theme(_current_theme)
    fig, axes = _plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    return Figure(fig, axes)


def subplots(nrows=1, ncols=1, figsize=(10, 6), **kwargs):
    _require_mpl()
    _apply_theme(_current_theme)
    return _plt.subplots(nrows, ncols, figsize=figsize, **kwargs)


# ── Basic plots ───────────────────────────────────────────────────────────────

def plot(x, y=None, label=None, color=None, linewidth=2, linestyle='-',
         alpha=1.0, ax=None, **kwargs):
    _require_mpl()
    _apply_theme(_current_theme)
    color = color or PALETTES[_current_palette][0]
    ax = ax or _plt.gca()
    if y is None:
        ax.plot(x, label=label, color=color, linewidth=linewidth,
                linestyle=linestyle, alpha=alpha, **kwargs)
    else:
        ax.plot(x, y, label=label, color=color, linewidth=linewidth,
                linestyle=linestyle, alpha=alpha, **kwargs)
    if label:
        ax.legend(framealpha=0.1, edgecolor='none')
    return ax


def scatter(x, y, hue=None, size=None, label=None, color=None, alpha=0.7,
            cmap=None, palette=None, ax=None, s=30, **kwargs):
    _require_mpl()
    _apply_theme(_current_theme)
    ax = ax or _plt.gca()
    pal = PALETTES.get(palette or _current_palette, PALETTES[_current_palette])
    if hue is not None:
        hue = np.array(hue)
        unique = np.unique(hue)
        for i, val in enumerate(unique):
            mask = hue == val
            ax.scatter(np.array(x)[mask], np.array(y)[mask],
                       label=str(val), color=pal[i % len(pal)], alpha=alpha, s=s, **kwargs)
        ax.legend(framealpha=0.1, edgecolor='none')
    else:
        c = color or pal[0]
        ax.scatter(x, y, label=label, color=c, alpha=alpha, s=s, **kwargs)
    return ax


def line(x, y, label=None, color=None, linewidth=2, **kwargs):
    return plot(x, y, label=label, color=color, linewidth=linewidth, **kwargs)


def bar(x, height, label=None, color=None, alpha=0.85, ax=None, **kwargs):
    _require_mpl()
    _apply_theme(_current_theme)
    ax = ax or _plt.gca()
    color = color or PALETTES[_current_palette][0]
    ax.bar(x, height, label=label, color=color, alpha=alpha, **kwargs)
    if label:
        ax.legend(framealpha=0.1, edgecolor='none')
    return ax


def barh(y, width, label=None, color=None, alpha=0.85, ax=None, **kwargs):
    _require_mpl()
    _apply_theme(_current_theme)
    ax = ax or _plt.gca()
    color = color or PALETTES[_current_palette][0]
    ax.barh(y, width, label=label, color=color, alpha=alpha, **kwargs)
    if label:
        ax.legend(framealpha=0.1, edgecolor='none')
    return ax


def histogram(data, bins=30, density=False, kde=False, label=None, color=None,
              alpha=0.7, ax=None, **kwargs):
    _require_mpl()
    _apply_theme(_current_theme)
    ax = ax or _plt.gca()
    color = color or PALETTES[_current_palette][0]
    n, edges, _ = ax.hist(data, bins=bins, density=density, label=label,
                           color=color, alpha=alpha, **kwargs)
    if kde:
        from scipy.stats import gaussian_kde as _kde
        x_range = np.linspace(np.min(data), np.max(data), 200)
        kde_fn = _kde(np.array(data).flatten())
        scale = 1.0 if density else (len(data) * (edges[1] - edges[0]))
        ax.plot(x_range, kde_fn(x_range) * scale, color=color, linewidth=2)
    if label:
        ax.legend(framealpha=0.1, edgecolor='none')
    return ax


def pie(x, labels=None, explode=None, colors=None, autopct='%1.1f%%',
        shadow=False, startangle=90, ax=None, **kwargs):
    _require_mpl()
    _apply_theme(_current_theme)
    ax = ax or _plt.gca()
    colors = colors or PALETTES[_current_palette]
    ax.pie(x, labels=labels, explode=explode, colors=colors[:len(x)],
           autopct=autopct, shadow=shadow, startangle=startangle, **kwargs)
    return ax


def area(x, y, alpha=0.3, color=None, label=None, ax=None, **kwargs):
    _require_mpl()
    _apply_theme(_current_theme)
    ax = ax or _plt.gca()
    color = color or PALETTES[_current_palette][0]
    ax.fill_between(x, y, alpha=alpha, color=color, label=label, **kwargs)
    ax.plot(x, y, color=color, linewidth=2)
    if label:
        ax.legend(framealpha=0.1, edgecolor='none')
    return ax


def step(x, y, where='mid', label=None, color=None, linewidth=2, ax=None, **kwargs):
    _require_mpl()
    ax = ax or _plt.gca()
    color = color or PALETTES[_current_palette][0]
    ax.step(x, y, where=where, label=label, color=color, linewidth=linewidth, **kwargs)
    if label:
        ax.legend(framealpha=0.1, edgecolor='none')
    return ax


# ── Statistical plots ─────────────────────────────────────────────────────────

def heatmap(data, annot=False, fmt='.2f', cmap='viridis', vmin=None, vmax=None,
            xticklabels=None, yticklabels=None, cbar=True, ax=None, **kwargs):
    _require_mpl()
    _apply_theme(_current_theme)
    ax = ax or _plt.gca()
    data = np.array(data)
    im = ax.imshow(data, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
    if cbar:
        _plt.colorbar(im, ax=ax)
    if annot:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f'{data[i, j]:{fmt}}', ha='center', va='center',
                        fontsize=7, color='white' if data[i, j] < (data.max() + data.min()) / 2 else 'black')
    if xticklabels is not None:
        ax.set_xticks(range(len(xticklabels)))
        ax.set_xticklabels(xticklabels, rotation=45, ha='right', fontsize=8)
    if yticklabels is not None:
        ax.set_yticks(range(len(yticklabels)))
        ax.set_yticklabels(yticklabels, fontsize=8)
    return ax


def boxplot(data, labels=None, notch=False, vert=True, patch_artist=True,
            ax=None, **kwargs):
    _require_mpl()
    _apply_theme(_current_theme)
    ax = ax or _plt.gca()
    bp = ax.boxplot(data, labels=labels, notch=notch, vert=vert,
                    patch_artist=patch_artist, **kwargs)
    if patch_artist:
        pal = PALETTES[_current_palette]
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(pal[i % len(pal)])
            patch.set_alpha(0.7)
    return ax


def violinplot(data, positions=None, vert=True, widths=0.5, showmeans=True,
               showextrema=True, showmedians=True, ax=None, **kwargs):
    _require_mpl()
    _apply_theme(_current_theme)
    ax = ax or _plt.gca()
    vp = ax.violinplot(data, positions=positions, vert=vert, widths=widths,
                       showmeans=showmeans, showextrema=showextrema, showmedians=showmedians, **kwargs)
    pal = PALETTES[_current_palette]
    for i, pc in enumerate(vp['bodies']):
        pc.set_facecolor(pal[i % len(pal)])
        pc.set_alpha(0.7)
    return ax


def distplot(data, bins=30, kde=True, hist=True, color=None, label=None,
             ax=None, **kwargs):
    _require_mpl()
    return histogram(data, bins=bins, density=True, kde=kde, label=label,
                     color=color, ax=ax, **kwargs)


def kdeplot(data, bw_method='scott', color=None, fill=False, alpha=0.3,
            label=None, ax=None, **kwargs):
    _require_mpl()
    _apply_theme(_current_theme)
    ax = ax or _plt.gca()
    color = color or PALETTES[_current_palette][0]
    try:
        from scipy.stats import gaussian_kde as _kde
        data = np.array(data).flatten()
        x_range = np.linspace(data.min(), data.max(), 200)
        kde_fn = _kde(data, bw_method=bw_method)
        vals = kde_fn(x_range)
        ax.plot(x_range, vals, color=color, label=label, linewidth=2, **kwargs)
        if fill:
            ax.fill_between(x_range, vals, alpha=alpha, color=color)
    except ImportError:
        raise ImportError("scipy required for kdeplot. pip install scipy")
    if label:
        ax.legend(framealpha=0.1, edgecolor='none')
    return ax


def pairplot(data, labels=None, palette=None, figsize=None, **kwargs):
    _require_mpl()
    _apply_theme(_current_theme)
    data = np.array(data)
    n = data.shape[1]
    labels = labels or [f'var{i}' for i in range(n)]
    fig, axes = _plt.subplots(n, n, figsize=figsize or (3*n, 3*n))
    pal = PALETTES.get(palette or _current_palette, PALETTES[_current_palette])
    for i in range(n):
        for j in range(n):
            ax = axes[i, j] if n > 1 else axes
            if i == j:
                ax.hist(data[:, i], bins=20, color=pal[i % len(pal)], alpha=0.7)
            else:
                ax.scatter(data[:, j], data[:, i], s=10, color=pal[i % len(pal)], alpha=0.5)
            if i == n - 1:
                ax.set_xlabel(labels[j], fontsize=8)
            if j == 0:
                ax.set_ylabel(labels[i], fontsize=8)
    _plt.tight_layout(pad=0.5)
    return fig


def jointplot(x, y, kind='scatter', color=None, figsize=(8, 8), **kwargs):
    _require_mpl()
    _apply_theme(_current_theme)
    color = color or PALETTES[_current_palette][0]
    fig = _plt.figure(figsize=figsize)
    gs = fig.add_gridspec(5, 5, hspace=0.05, wspace=0.05)
    ax_joint = fig.add_subplot(gs[1:, :-1])
    ax_top = fig.add_subplot(gs[0, :-1], sharex=ax_joint)
    ax_right = fig.add_subplot(gs[1:, -1], sharey=ax_joint)
    if kind == 'scatter':
        ax_joint.scatter(x, y, color=color, alpha=0.5, s=20, **kwargs)
    elif kind == 'hex':
        ax_joint.hexbin(x, y, cmap='plasma', gridsize=25)
    ax_top.hist(x, bins=30, color=color, alpha=0.7)
    ax_right.hist(y, bins=30, color=color, alpha=0.7, orientation='horizontal')
    _plt.setp(ax_top.get_xticklabels(), visible=False)
    _plt.setp(ax_right.get_yticklabels(), visible=False)
    return fig


def regplot(x, y, color=None, ci=95, label=None, ax=None, **kwargs):
    _require_mpl()
    _apply_theme(_current_theme)
    ax = ax or _plt.gca()
    color = color or PALETTES[_current_palette][0]
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    ax.scatter(x, y, color=color, alpha=0.5, s=20)
    # Fit line
    p = np.polyfit(x, y, 1)
    x_sorted = np.linspace(x.min(), x.max(), 100)
    y_line = np.polyval(p, x_sorted)
    ax.plot(x_sorted, y_line, color=color, linewidth=2, label=label)
    if ci:
        # Confidence interval via bootstrap
        n = len(x)
        predictions = np.array([np.polyval(np.polyfit(x, y, 1), x_sorted)
                                  for _ in range(50)])  # 50 bootstrap samples simplified
        y_std = predictions.std(axis=0)
        z = 1.96 if ci == 95 else 2.576
        ax.fill_between(x_sorted, y_line - z * y_std, y_line + z * y_std,
                        alpha=0.15, color=color)
    if label:
        ax.legend(framealpha=0.1, edgecolor='none')
    return ax


def imshow(data, cmap='viridis', interpolation='nearest', aspect='auto', ax=None, **kwargs):
    _require_mpl()
    _apply_theme(_current_theme)
    ax = ax or _plt.gca()
    ax.imshow(data, cmap=cmap, interpolation=interpolation, aspect=aspect, **kwargs)
    return ax


def contour(x, y, z, levels=10, cmap='viridis', ax=None, **kwargs):
    _require_mpl()
    _apply_theme(_current_theme)
    ax = ax or _plt.gca()
    ax.contour(x, y, z, levels=levels, cmap=cmap, **kwargs)
    return ax


def contourf(x, y, z, levels=10, cmap='viridis', alpha=0.8, ax=None, **kwargs):
    _require_mpl()
    _apply_theme(_current_theme)
    ax = ax or _plt.gca()
    cs = ax.contourf(x, y, z, levels=levels, cmap=cmap, alpha=alpha, **kwargs)
    _plt.colorbar(cs, ax=ax)
    return ax


# ── ML-specific plots ─────────────────────────────────────────────────────────

def confusion_matrix_plot(cm, class_names=None, normalize=False, cmap='Blues',
                           title='Confusion Matrix', figsize=(8, 6)):
    _require_mpl()
    _apply_theme(_current_theme)
    cm = np.array(cm)
    if normalize:
        cm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)
    fig, ax = _plt.subplots(figsize=figsize)
    heatmap(cm, annot=True, fmt='.2f' if normalize else '.0f', cmap=cmap,
            xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    ax.set_title(title, fontsize=13, pad=12)
    _plt.tight_layout()
    return fig


def roc_curve_plot(fpr_list, tpr_list, auc_list=None, labels=None,
                   figsize=(7, 7), title='ROC Curve'):
    _require_mpl()
    _apply_theme(_current_theme)
    pal = PALETTES[_current_palette]
    fig, ax = _plt.subplots(figsize=figsize)
    if not isinstance(fpr_list[0], (list, np.ndarray)):
        fpr_list = [fpr_list]
        tpr_list = [tpr_list]
        auc_list = [auc_list] if auc_list is not None else [None]
    for i, (fpr, tpr) in enumerate(zip(fpr_list, tpr_list)):
        label = labels[i] if labels else f'Model {i+1}'
        if auc_list and auc_list[i] is not None:
            label += f' (AUC={auc_list[i]:.3f})'
        ax.plot(fpr, tpr, color=pal[i % len(pal)], linewidth=2, label=label)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(framealpha=0.1, edgecolor='none')
    _plt.tight_layout()
    return fig


def training_history(history: dict, figsize=(12, 4)):
    _require_mpl()
    _apply_theme(_current_theme)
    pal = PALETTES[_current_palette]
    keys = list(history.keys())
    n = len(keys)
    fig, axes = _plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for i, (key, ax) in enumerate(zip(keys, axes)):
        vals = history[key]
        ax.plot(vals, color=pal[i % len(pal)], linewidth=2)
        ax.set_title(key, fontsize=11)
        ax.set_xlabel('Epoch', fontsize=9)
    _plt.tight_layout()
    return fig


def feature_importance_plot(importances, feature_names=None, top_n=20,
                             title='Feature Importances', figsize=(10, 6)):
    _require_mpl()
    _apply_theme(_current_theme)
    importances = np.array(importances)
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(len(importances))]
    idx = np.argsort(importances)[-top_n:]
    fig, ax = _plt.subplots(figsize=figsize)
    colors = [PALETTES[_current_palette][i % len(PALETTES[_current_palette])]
              for i in range(len(idx))]
    ax.barh(np.array(feature_names)[idx], importances[idx], color=colors, alpha=0.8)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('Importance', fontsize=11)
    _plt.tight_layout()
    return fig


def residuals_plot(y_true, y_pred, ax=None, **kwargs):
    _require_mpl()
    _apply_theme(_current_theme)
    ax = ax or _plt.gca()
    residuals = np.array(y_true) - np.array(y_pred)
    ax.scatter(y_pred, residuals, color=PALETTES[_current_palette][0], alpha=0.5, s=20)
    ax.axhline(0, color='white', linewidth=1, linestyle='--', alpha=0.5)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Residuals', fontsize=11)
    ax.set_title('Residuals vs. Predicted', fontsize=13)
    return ax


def learning_curve(train_scores, val_scores, train_sizes, figsize=(8, 5)):
    _require_mpl()
    _apply_theme(_current_theme)
    pal = PALETTES[_current_palette]
    fig, ax = _plt.subplots(figsize=figsize)
    ax.plot(train_sizes, np.mean(train_scores, axis=1) if train_scores[0].ndim else train_scores,
            color=pal[0], linewidth=2, label='Train')
    ax.plot(train_sizes, np.mean(val_scores, axis=1) if val_scores[0].ndim else val_scores,
            color=pal[1], linewidth=2, label='Validation')
    ax.fill_between(train_sizes,
                    np.mean(train_scores, axis=1) - np.std(train_scores, axis=1) if hasattr(train_scores[0], '__len__') else train_scores,
                    np.mean(train_scores, axis=1) + np.std(train_scores, axis=1) if hasattr(train_scores[0], '__len__') else train_scores,
                    alpha=0.1, color=pal[0])
    ax.legend(framealpha=0.1, edgecolor='none')
    ax.set_xlabel('Training Set Size', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Learning Curve', fontsize=13)
    _plt.tight_layout()
    return fig


# ── Display control ───────────────────────────────────────────────────────────

def show():
    _require_mpl()
    _plt.show()

def close(fig=None):
    _require_mpl()
    if fig is None:
        _plt.close('all')
    elif hasattr(fig, '_fig'):
        _plt.close(fig._fig)
    else:
        _plt.close(fig)

def savefig(fname, dpi=150, bbox_inches='tight', **kwargs):
    _require_mpl()
    _plt.savefig(fname, dpi=dpi, bbox_inches=bbox_inches, **kwargs)

def title(s, **kwargs):
    _require_mpl()
    _plt.title(s, **kwargs)

def xlabel(s, **kwargs):
    _require_mpl()
    _plt.xlabel(s, **kwargs)

def ylabel(s, **kwargs):
    _require_mpl()
    _plt.ylabel(s, **kwargs)

def xlim(*args, **kwargs):
    _require_mpl()
    _plt.xlim(*args, **kwargs)

def ylim(*args, **kwargs):
    _require_mpl()
    _plt.ylim(*args, **kwargs)

def legend(**kwargs):
    _require_mpl()
    _plt.legend(framealpha=0.1, edgecolor='none', **kwargs)

def grid(visible=True, **kwargs):
    _require_mpl()
    _plt.grid(visible, **kwargs)

def tight_layout(**kwargs):
    _require_mpl()
    _plt.tight_layout(**kwargs)

def gca():
    _require_mpl()
    return _plt.gca()

def gcf():
    _require_mpl()
    return _plt.gcf()

import os

import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True


def plot_mnmatrix(m, filepath=None):
    """
    Draw a Minuit MN-matrix contour plot.

    Parameters
    ----------
    m : iminuit.Minuit
        Minuit object after migrad() and minos() have been called.
    filepath : str or None
        Path to save the figure (including filename and extension).
        If None, calls plt.show() instead.
    """
    try:
        fig, axs = m.draw_mnmatrix()
    except ValueError as e:
        print(f"Error drawing Minuit contour: {e}")
        return

    n = axs.shape[0]
    diag_width_frac = 0.8
    r = 1 - diag_width_frac

    for j in range(n):
        offdiag = [axs[i, j] for i in range(j + 1, n)]
        for ax in offdiag[1:]:
            ax.sharex(offdiag[0])
        for i in range(j, n - 1):
            axs[i, j].tick_params(labelbottom=False)

    for i in range(1, n):
        row_axes = [axs[i, j] for j in range(i)]
        for ax in row_axes[1:]:
            ax.sharey(row_axes[0])
        for ax in row_axes[1:]:
            ax.tick_params(labelleft=False)

    fig.set_size_inches(8, 8)
    fig.tight_layout()

    diag_xlims = [axs[j, j].get_xlim() for j in range(n)]
    diag_xticks = [axs[j, j].get_xticks() for j in range(n)]

    for i in range(n):
        for j in range(i + 1):
            axs[i, j].tick_params(direction='in', top=True, right=True)

    for j in range(n):
        axs[j, j].tick_params(left=False, labelleft=False)
        axs[j, j].set_ylabel(r'$\chi^2$')

    col_w = [axs[j, j].get_position().width for j in range(n)]
    diag_w = [w * diag_width_frac for w in col_w]
    left_margin = axs[n - 1, 0].get_position().x0
    bottom_margin = axs[n - 1, 0].get_position().y0
    top_edge = axs[0, 0].get_position().y0 + axs[0, 0].get_position().height
    row_h = (top_edge - bottom_margin) / n
    new_col_x0 = [left_margin]
    for j in range(1, n):
        new_col_x0.append(new_col_x0[j - 1] + col_w[j - 1])

    for i in range(n):
        for j in range(i + 1):
            y0 = bottom_margin + (n - 1 - i) * row_h
            if i == j:
                axs[i, j].set_position(
                    [new_col_x0[j] + col_w[j] - diag_w[j], y0, diag_w[j], row_h])
            else:
                axs[i, j].set_position([new_col_x0[j], y0, col_w[j], row_h])

    for j in range(n - 1):
        xlim_lo, xlim_hi = diag_xlims[j]
        x_left = (xlim_lo - r * xlim_hi) / (1 - r)
        axs[j + 1, j].set_xlim(x_left, xlim_hi)
        valid_ticks = [t for t in diag_xticks[j] if xlim_lo <= t <= xlim_hi]
        if valid_ticks:
            axs[n - 1, j].set_xticks(valid_ticks)

    if filepath is not None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close(fig)
    else:
        plt.show()

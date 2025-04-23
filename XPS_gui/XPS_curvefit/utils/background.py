import numpy as np


def shirley_bg(x, y, i1, i2, tol=1e-6, max_iter=300):
    """
    Very compact Shirley background between indices i1 and i2
    (adapted from common implementations).
    """
    bg = np.interp(x, (x[i1], x[i2]), (y[i1], y[i2]))
    for _ in range(max_iter):
        area = np.trapz(y - bg, x)
        new_bg = bg + (area / (x[i2] - x[i1]))
        if np.max(np.abs(new_bg - bg)) < tol:
            break
        bg = new_bg
    return bg

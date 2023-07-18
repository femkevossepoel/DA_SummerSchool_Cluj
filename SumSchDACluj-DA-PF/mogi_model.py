import numpy as np


def mogi(x, y, z, sources, strengths, nu=0.25):
    """Calculates surface deformation based on point sources.

    This is modified version of the following code:

        https://github.com/scottyhq/cov9/blob/master/mogi.py
        (Scott Henderson; 8/31/2012)

    The adjustments are particular for the purpose of the DA assignment. The
    principal changes are:
    - Accepts z-location of the grid.
    - Volume change is defined as a source strength factor, which is internally
      multiplied by dV=1e6 m3.
    - Recursion to compute multiple sources in one go. The sources are assumed
      to be independent and their effects are simply summed.

    References: Mogi 1958, Segall 2010 p.203

    Parameters
    ----------
    x, y, z : ndarray
        Coordinates of observation points [m]. Can have any shape as long as
        they are the same for x, y, and z; positive z upwards.

    sources : ndarray
        Source positions [m]; `Nx4`, where `N` is the number of sources.

            [[x1, y1, z1],
             [ ·,  ·,  ·],
             [xN, yN, zN]]

    strengths : ndarray
        Source strengths; array of length `N`; dimensionless (internally
        multiplied by 1e6 m3).

    nu: float; default: 0.25
        Poisson's ratio for medium.

    Returns
    -------
    ux, uy, uz: ndarray
        Deformations in x, y, z [m] (same shape as input x/y/z).

    """
    # Use recursion for multiple sources (assuming independent sources!).
    if sources.ndim == 2:
        uxyz = mogi(x, y, z, sources[0, :], strengths[0], nu)
        for i in range(1, sources.shape[0]):
            uxyz += mogi(x, y, z, sources[i, :], strengths[i], nu)
        return uxyz

    # Center coordinate grid on point source
    x = x - sources[0]
    y = y - sources[1]
    z = z - sources[2]

    # Convert to surface cylindrical coordinates
    theta = np.arctan2(y, x)
    rho = np.hypot(y, x)
    R = np.hypot(z, rho)

    # Mogi displacement calculation
    dV = strengths * 1e6  # Total volume change
    C = ((1-nu) / np.pi) * dV
    ur = C * rho / R**3
    uz = C * z / R**3

    ux = ur * np.cos(theta)
    uy = ur * np.sin(theta)

    return np.array([ux, uy, uz])

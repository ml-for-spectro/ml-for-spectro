# utils/fitting_helpers.py
from lmfit.models import VoigtModel


def build_voigt_model(x, peak_centers, pref_list=None):
    """
    Create a composite Voigt model.
    Parameters
    ----------
    x            : array-like (not used in model construction, but handy)
    peak_centers : list/array of initial center positions
    pref_list    : optional list of string prefixes ("A_", "B_", ...)
                   If None, defaults to ["v1_","v2_",...]
    Returns
    -------
    model  : lmfit.Model (sum of Voigt components)
    params : lmfit.Parameters with initial guesses
    """
    if pref_list is None:
        pref_list = [f"v{i+1}_" for i in range(len(peak_centers))]

    model, params = None, None

    for cen, pref in zip(peak_centers, pref_list):
        # print(cen)
        # print(type(cen))
        # print(pref)
        v = VoigtModel(prefix=pref)

        # add component to composite model
        if model is None:
            model = v
            params = v.make_params()
        else:
            model += v
            params.update(v.make_params())

        # sensible initial guesses / loose bounds
        params[pref + "center"].set(value=cen, min=cen - 1, max=cen + 1)
        params[pref + "sigma"].set(value=0.3, min=0.05, max=2.0)
        params[pref + "gamma"].set(value=0.3, min=0.05, max=2.0)
        params[pref + "amplitude"].set(value=1.0, min=0)

    return model, params

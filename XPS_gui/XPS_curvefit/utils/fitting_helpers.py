# utils/fitting_helpers.py
from lmfit.models import VoigtModel

def build_voigt_model(x, peak_centers):
    """
    Build a composite lmfit VoigtModel, one component per center in
    peak_centers.  Returns (model, params) ready for fitting.
    """
    model  = None
    params = None
    for i, cen in enumerate(peak_centers, 1):
        pref = f"v{i}_"
        v = VoigtModel(prefix=pref)

        # first component starts the composite
        if model is None:
            model = v
            params = v.make_params()
        else:
            model += v
            params.update(v.make_params())

        # reasonable initial guesses / bounds
        params[pref+"center"].set(value=cen, min=cen-1, max=cen+1)
        params[pref+"sigma" ].set(value=0.3, min=0.05, max=2)
        params[pref+"gamma" ].set(value=0.3, min=0.05, max=2)
        params[pref+"amplitude"].set(value=1, min=0)

    return model, params

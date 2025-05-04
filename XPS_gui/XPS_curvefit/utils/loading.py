import csv


def load_fit_parameters_csv(file_path):
    names, centers, sigmas, gammas, amplitudes = [], [], [], [], []
    sigma_exprs, gamma_exprs = [], []

    with open(file_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if not row or not row[0].strip():
                continue  # Skip empty lines
            names.append(row[0].strip())
            centers.append(float(row[1]))
            sigmas.append(float(row[2]))
            gammas.append(float(row[3]))
            amplitudes.append(float(row[4]))
            sigma_exprs.append(
                row[6].strip() if len(row) > 6 and row[6].strip() else None
            )
            gamma_exprs.append(
                row[7].strip() if len(row) > 7 and row[7].strip() else None
            )

    return {
        "names": names,
        "centers": centers,
        "sigmas": sigmas,
        "gammas": gammas,
        "amplitudes": amplitudes,
        "sigma_exprs": sigma_exprs,
        "gamma_exprs": gamma_exprs,
    }

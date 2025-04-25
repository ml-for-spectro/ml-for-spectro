# XPS CurveFit

**A lightweight XPS (X-ray Photoelectron Spectroscopy) analysis GUI for smoothing, background subtraction, and Voigt peak fitting.**

Developed with [PySide6](https://doc.qt.io/qtforpython/) for researchers who want a quick, no-frills data analysis workflow.

---

## 🚀 Features

- 📂 Load and view spectra from text/CSV files
- 🎛 Apply Gaussian smoothing
- 🔧 Subtract Shirley/Tougaard background
- 📈 Interactive Voigt peak fitting
- 🧮 Compare multiple spectra
- 🧠 Stores photon energy and background crop state between runs

---

## 🛠 Installation

You need **Python 3.8+**.

### ⚙️ Local editable install (for development)

git clone https://github.com/yourusername/xps-curvefit.git
cd xps-curvefit
pip install -e .

Then run it:
xps-curvefit


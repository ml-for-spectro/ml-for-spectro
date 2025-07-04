# ML for Spectro

Welcome to **ml-for-spectro**, a personal project led by a beamline scientist at Synchrotron SOLEIL, exploring the application of Machine Learning (ML) and AI in the nanoscale characterization of organic and hybrid photovoltaic materials.

---

## 🧠 Objective
To build ML and AI tools that:
- Analyze and classify spectroscopic and imaging data (e.g., XPEEM, STXM, GIWAXS)
- Enable smart visualization of large-scale datasets (e.g., HDF5, NeXus)
- Integrate GUI tools for research workflows

---

## 📁 Project Layout
```
ml-for-spectro/
├── data/                 # Raw and processed datasets (not tracked)
├── notebooks/            # Jupyter notebooks for exploration and prototyping
├── gui/                  # GUI apps (e.g., PySide6-based loaders and viewers)
├── models/               # Trained ML models, configs, checkpoints
├── utils/                # Helper scripts, plotting, data prep tools
├── environment.yml       # Anaconda environment setup
└── README.md             # This file
```

---

## 🔧 Environment Setup
```bash
conda env create -f environment.yml
conda activate ml-for-spectro
```

---

## 🚀 First Goal (Milestone 1)
Build a PySide6 GUI that can:
- Load and view `.hdf5` XPEEM datasets
- Display energy, sample_x, sample_y axes

Followed by:
- Preprocessing + ML model training notebooks (classification, dimensionality reduction)
- Eventually packaging useful tools for others

---

## 🧑‍💻 Author


---

Stay tuned for updates!

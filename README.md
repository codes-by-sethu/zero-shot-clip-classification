# Zero-Shot Image Classification using CLIP

Group:

Man Vijaybhai PATEL – DIA1

Luthfi Juneeda SHAJ – DIA1

Sethulakshmi KOCHUCHIRAYIL BABU – DIA1

Yasar THAJUDEEN – DIA1


This project demonstrates zero-shot image classification on a custom dataset using **OpenAI CLIP**. The pipeline compares zero-shot performance with few-shot / fine-tuned baselines and explores prompt engineering and visual reasoning.

---

## Project Overview

* **Goal:** Evaluate CLIP's ability to classify domain-specific images without training, and improve performance using prompt design and small labeled datasets.
* **Models:** CLIP (ViT-B/32, RN50), linear probe / fine-tuned classifier.
* **Key Features:**

  * Zero-shot classification with multiple prompts
  * Few-shot/fine-tuning experiments
  * Evaluation using Top-1, Top-5 accuracy, precision, recall, F1
  * Visualizations of misclassifications and Grad-CAM for analysis

---

## Repository Structure

```
zero-shot-clip-classification/
│
├── data/                # Dataset or scripts to download images
├── notebooks/           # Jupyter notebooks with experiments
├── src/                 # Python scripts (models, utils)
├── results/             # Evaluation metrics, plots, and figures
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Installation

1. Clone the repo:

```bash
git clone https://github.com/<your-username>/zero-shot-clip-classification.git
cd zero-shot-clip-classification
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Usage

1. Place dataset in `data/` folder or run dataset download scripts.
2. Open the notebook `notebooks/zero_shot_clip.ipynb` to run experiments.
3. Modify prompts, models, or datasets as needed.
4. Visualizations and metrics are saved in `results/`.

---

## Requirements

* Python 3.8+
* PyTorch 2.0+
* Jupyter Notebook
* open_clip_torch
* matplotlib, pandas, numpy

*(All dependencies listed in `requirements.txt`)*

---

## Project Timeline

* Week 1: Dataset setup & environment
* Week 2: Zero-shot CLIP pipeline
* Week 3: Prompt engineering experiments
* Week 4: Few-shot / fine-tuning experiments
* Week 5: Failure analysis & visualizations
* Week 6-8: Documentation, slides, and final report

---

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

---

## License

This project is licensed under the MIT License.

---

## Cont

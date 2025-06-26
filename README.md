# Phishing Website Detection using Machine Learning

This project aims to detect phishing websites using various machine learning algorithms. It analyzes website features and classifies them as either legitimate or phishing, helping users stay protected from online threats.

---

## Features

- Extracts important features from URLs
- Supports multiple ML models (Decision Tree, Random Forest, etc.)
- Evaluates model performance using metrics like accuracy, precision, recall, and F1-score
- Jupyter/Colab notebook implementation for easy testing

---

## Technologies Used

- Python 3.x
- Google Colab / Jupyter Notebook
- Libraries: `scikit-learn`, `pandas`, `matplotlib`, `seaborn`, `numpy`

---

## Machine Learning Models

- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Support Vector Machine (SVM)

---

## Dataset

The dataset used includes features extracted from website URLs. Some of the important features are:
- URL length
- Use of HTTPS
- Presence of "@" symbol
- Domain age
- DNS record availability

> You can find the dataset in the `dataset/` directory or import it in the notebook from a shared Google Drive/URL.

---

## Results

| Model            | Accuracy |
|------------------|----------|
| Random Forest    | 97.3%    |
| Decision Tree    | 94.5%    |
| Logistic Regression | 92.1% |
| SVM              | 90.4%    |

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/phishing-detection.git
   cd phishing-detection

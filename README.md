# Diabetes Prediction Using SVM

A machine learning project that predicts whether a person is **diabetic** from medical measurements, using a **Support Vector Machine (SVM)** classifier.

## Dataset

[PIMA Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database): 768 records, 8 features and 1 label.

| Feature | Description |
|---------|-------------|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-hour serum insulin |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Family history score |
| Age | Age in years |
| **Outcome** | 0 = not diabetic, 1 = diabetic |

## Workflow

1. Load and explore the data with Pandas (`describe`, class counts, group means)
2. Split features (X) and label (Y)
3. Standardize the features with `StandardScaler`
4. Split into train and test sets (80/20, stratified)
5. Train an SVM with a linear kernel
6. Measure accuracy
7. Build a small predictive system for new input

## Results

| Data | Accuracy |
|------|----------|
| Training | ~78.7% |
| Testing | ~77.3% |

## Tech Stack

- Python
- NumPy, Pandas
- scikit-learn (`SVC`, `StandardScaler`, `train_test_split`)
- Jupyter Notebook

## How to Run

```bash
git clone https://github.com/PRiNCeKUsHW/Diabetics-svm-ML.git
cd "Diabetics-svm-ML/Diabetes prediction project"
pip install numpy pandas scikit-learn jupyter
jupyter notebook "Diabetes of person SVM.ipynb"
```

To predict for a new person, change `input_data` in the last cell (8 values in the feature order above) and run it.

## Project Structure

```
Diabetes prediction project/
├── diabetes.csv                    # Dataset
└── Diabetes of person SVM.ipynb    # Analysis, training and prediction
```

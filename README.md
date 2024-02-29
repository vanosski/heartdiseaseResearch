# Heart Disease Prediction Using Machine Learning

This project aims to predict heart disease using machine learning techniques. The dataset used in this project contains various attributes related to heart health, such as age, sex, cholesterol levels, resting blood pressure, and other medical indicators. By training machine learning models on this dataset, we can predict whether a person is likely to have heart disease based on their health metrics.

## Dataset
The dataset used in this project is sourced from Kaggle and consists of the following columns:
- Age: Age of the patient
- Sex: Gender of the patient (0 for female, 1 for male)
- ChestPainType: Type of chest pain experienced by the patient
- RestingBP: Resting blood pressure of the patient
- Cholesterol: Cholesterol level of the patient
- RestingECG: Resting electrocardiographic results
- MaxHeartRate: Maximum heart rate achieved by the patient
- ExerciseAngina: Whether the patient experiences exercise-induced angina (0 for No, 1 for Yes)
- Oldpeak: ST depression induced by exercise relative to rest
- ST_Slope: Slope of the peak exercise ST segment

## Preprocessing
- Removed records with resting blood pressure (RestingBP) equal to 0.
- Replaced 0 values in the 'Cholesterol' column with the median value.
- Converted object columns ('Sex' and 'ExerciseAngina') to numeric values.
- Encoded categorical columns ('ChestPainType', 'RestingECG', 'ST_Slope') using label encoding.

## Machine Learning Models
The following machine learning models were trained and evaluated for heart disease prediction:
- Random Forest Classifier
- Support Vector Machine (SVM) Classifier
- XGBoost Classifier

## Training and Evaluation
- The dataset was split into training and testing sets using a 70-30 split.
- Power Transformer was applied to normalize features.
- GridSearchCV was used for hyperparameter tuning.
- Model performance was evaluated using accuracy score, classification report, and ROC curves.
- Feature importances were extracted to identify the most important features for prediction.

## Results
- Random Forest Classifier achieved the highest accuracy among the models.
- Selected top 5 features based on importance for prediction.
- Saved trained models as pickle files for future use.

## Usage
1. Clone the repository.
2. Install the required dependencies listed .
3. Run the `mid.py` script to train machine learning models and evaluate performance.
4. Trained models will be saved as pickle files for later use.

## Requirements
- Python 3.x
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- xgboost


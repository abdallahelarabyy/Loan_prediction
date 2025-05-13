# Loan Prediction

## Project Overview

The **Loan Prediction** project is a machine learning model designed to predict whether a loan application will be approved based on various factors like applicant income, loan amount, credit history, and more. The model is built using data preprocessing techniques, feature engineering, and classification algorithms. This project utilizes **Random Forest**, **Gradient Boosting**, and **Logistic Regression** classifiers, and the best model is selected based on performance metrics.

### Data

The dataset contains information about loan applicants, with columns such as:

* **Loan\_ID**: Unique identifier for the loan
* **Gender**: Gender of the applicant
* **Married**: Marital status of the applicant
* **Dependents**: Number of dependents
* **Education**: Education level of the applicant
* **Self\_Employed**: Whether the applicant is self-employed
* **ApplicantIncome**: Income of the applicant
* **CoapplicantIncome**: Income of the coapplicant (if any)
* **LoanAmount**: The amount of the loan requested
* **Loan\_Amount\_Term**: Term of the loan (in months)
* **Credit\_History**: Credit history of the applicant (1 means good, 0 means bad)
* **Property\_Area**: Area where the property is located (Urban, Semiurban, Rural)
* **Loan\_Status**: Outcome of the loan (Y for approved, N for rejected)

### Project Workflow

1. **Data Preprocessing**:

   * Handling missing values and encoding categorical variables.
   * Feature engineering, including scaling and one-hot encoding.
   * Splitting the data into training and testing sets.

2. **Model Training**:

   * Multiple classification models are trained, including **Random Forest**, **Gradient Boosting**, and **Logistic Regression**.
   * Hyperparameter tuning is performed using **GridSearchCV** to optimize the model performance.

3. **Model Evaluation**:

   * The model is evaluated using **accuracy**, **precision**, **recall**, and **F1 score** metrics.

4. **Prediction**:

   * Once trained, the model can predict the approval status of new loan applications, providing the likelihood of approval.

5. **Model Persistence**:

   * The best-performing model is saved as a `.pkl` file for future use and deployment.

### Requirements

* Python 3.x
* Pandas
* Numpy
* Scikit-learn
* Joblib

### How to Run the Project

1. Install the necessary packages:

   ```bash
   pip install pandas numpy scikit-learn joblib
   ```

2. Preprocess the data and train the model using the script `model.py`.

3. Use the trained model to make predictions by calling the `predict()` method from the `LoanPredictor` class.

4. Save the model to a file for future use by calling the `save_model()` method.

### Example Usage

```python
from model import LoanPredictor

# Create an instance of the predictor
predictor = LoanPredictor()

# Load the trained model
predictor.load_model("loan_model.pkl")

# Example input for prediction
input_data = {
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": 1,
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 1500,
    "LoanAmount": 120,
    "Loan_Amount_Term": 360,
    "Credit_History": 1.0,
    "Property_Area": "Urban"
}

# Make a prediction
result = predictor.predict(input_data)
print(result)
```


# model.py - Machine Learning Component

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LoanPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = None
        self.property_area_columns = None

    def preprocess_data(self, df):
        """Preprocess the data according to the pipeline"""
        # Make a copy to avoid modifying the original
        df_copy = df.copy()

        # Fill missing values
        df_copy["Gender"].fillna("Unknown", inplace=True)
        df_copy["Married"].fillna("Unknown", inplace=True)
        df_copy["Dependents"].fillna("Unknown", inplace=True)
        df_copy["Self_Employed"].fillna("Unknown", inplace=True)
        df_copy["LoanAmount"].fillna(
            df_copy["LoanAmount"].median(), inplace=True)
        df_copy["Loan_Amount_Term"].fillna(
            df_copy["Loan_Amount_Term"].median(), inplace=True)
        df_copy["Credit_History"].fillna(
            df_copy["Credit_History"].mode()[0], inplace=True)

        # Handle Loan_ID if present
        if "Loan_ID" in df_copy.columns:
            try:
                df_copy["Loan_ID"] = df_copy["Loan_ID"].str.extract(
                    "(\d+)").astype(int)
            except:
                # If conversion fails, just drop the column
                df_copy.drop("Loan_ID", axis=1, inplace=True)

        # Encode categorical variables
        categorical_cols = ["Gender", "Married", "Education", "Self_Employed"]
        for col in categorical_cols:
            if col in df_copy.columns:
                le = self.label_encoders.get(col)
                if le is None:
                    le = LabelEncoder()
                    le.fit(df_copy[col])
                    self.label_encoders[col] = le
                df_copy[col] = le.transform(df_copy[col])

        # Convert Dependents with mapping
        dependents_mapping = {"0": 0, "1": 1, "2": 2, "3+": 3, "Unknown": -1}
        df_copy["Dependents"] = df_copy["Dependents"].map(dependents_mapping)

        # One-Hot Encoding for Property_Area
        if "Property_Area" in df_copy.columns:
            property_area_dummies = pd.get_dummies(
                df_copy["Property_Area"], prefix="Property_Area")
            # If we're in training mode, save the columns
            if self.property_area_columns is None:
                self.property_area_columns = property_area_dummies.columns.tolist()
            # Otherwise, ensure we have the same columns
            else:
                for col in self.property_area_columns:
                    if col not in property_area_dummies.columns:
                        property_area_dummies[col] = 0
                # Reorder and select only known columns
                property_area_dummies = property_area_dummies[self.property_area_columns]

            # Drop the original column and join the dummies
            df_copy.drop("Property_Area", axis=1, inplace=True)
            df_copy = pd.concat([df_copy, property_area_dummies], axis=1)

        # Save feature names if in training mode
        if self.feature_names is None:
            self.feature_names = df_copy.columns.tolist()

        # Standard scaling
        if "Loan_Status" in df_copy.columns:
            # Training mode - fit scaler
            y = df_copy["Loan_Status"]
            X = df_copy.drop("Loan_Status", axis=1)

            if self.scaler is None:
                self.scaler = StandardScaler()
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = self.scaler.transform(X)

            return X_scaled, y
        else:
            # Prediction mode - use existing scaler
            if self.scaler is not None:
                X_scaled = self.scaler.transform(df_copy)
                return X_scaled
            else:
                raise ValueError("Scaler not fitted. Train the model first.")

    def train(self, df, model_type="random_forest"):
        """Train the model on the given dataframe"""
        logger.info(f"Starting model training with {model_type}")

        # First ensure Loan_Status is encoded if it's not already
        if "Loan_Status" in df.columns and df["Loan_Status"].dtype == 'object':
            loan_status_encoder = LabelEncoder()
            df["Loan_Status"] = loan_status_encoder.fit_transform(
                df["Loan_Status"])
            self.label_encoders["Loan_Status"] = loan_status_encoder

        # Preprocess data
        X, y = self.preprocess_data(df)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Select and train model
        if model_type == "random_forest":
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            base_model = RandomForestClassifier(random_state=42)

        elif model_type == "gradient_boosting":
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5],
                'min_samples_split': [2, 5]
            }
            base_model = GradientBoostingClassifier(random_state=42)

        elif model_type == "logistic_regression":
            param_grid = {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l2'],
                'solver': ['liblinear', 'saga']
            }
            base_model = LogisticRegression(random_state=42, max_iter=1000)

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Use GridSearch for hyperparameter tuning
        grid_search = GridSearchCV(
            base_model, param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Get best model
        self.model = grid_search.best_estimator_
        logger.info(f"Best parameters: {grid_search.best_params_}")

        # Evaluate on test set
        y_pred = self.model.predict(X_test)

        # Print evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        logger.info(f"Model Evaluation:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred))

        # Return metrics for comparison
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "model": self.model,
            "best_params": grid_search.best_params_
        }

    def predict(self, input_data):
        """Make a prediction for the given input data"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        # Convert to DataFrame if it's a dict
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])

        # Preprocess the input
        X_processed = self.preprocess_data(input_data)

        # Make prediction
        prediction = self.model.predict(X_processed)[0]
        probability = self.model.predict_proba(X_processed)[0][1]

        # Convert prediction back to original label if needed
        if "Loan_Status" in self.label_encoders:
            prediction = self.label_encoders["Loan_Status"].inverse_transform([prediction])[
                0]

        return {
            "prediction": int(prediction),
            "approval_probability": float(probability),
            "approval_status": "Approved" if prediction == 1 else "Rejected"
        }

    def save_model(self, model_path="loan_model.pkl"):
        """Save the trained model and preprocessing components"""
        if self.model is None:
            raise ValueError("No trained model to save")

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "label_encoders": self.label_encoders,
            "feature_names": self.feature_names,
            "property_area_columns": self.property_area_columns
        }

        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")

    def load_model(self, model_path="loan_model.pkl"):
        """Load a trained model and preprocessing components"""
        try:
            model_data = joblib.load(model_path)

            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            self.label_encoders = model_data["label_encoders"]
            self.feature_names = model_data["feature_names"]
            self.property_area_columns = model_data["property_area_columns"]

            logger.info(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False


def get_feature_importance(predictor):
    """Extract feature importance if available"""
    if predictor.model is None or not hasattr(predictor.model, 'feature_importances_'):
        return {}

    # Get feature names and importances
    importances = predictor.model.feature_importances_
    feature_names = predictor.feature_names

    # Remove Loan_Status if present
    if "Loan_Status" in feature_names:
        idx = feature_names.index("Loan_Status")
        feature_names.pop(idx)
        importances = np.delete(importances, idx)

    # Sort by importance
    indices = np.argsort(importances)[::-1]

    # Return sorted feature importances
    return {
        feature_names[i]: float(importances[i])
        for i in indices if i < len(feature_names)
    }


def train_best_model():
    """Train and select the best model"""
    # Load training data
    df = pd.read_csv("loan.csv")
    predictor = LoanPredictor()

    # Train models with different algorithms for comparison
    models = {}
    for model_type in ["random_forest", "gradient_boosting", "logistic_regression"]:
        logger.info(f"Training {model_type} model...")
        metrics = predictor.train(df, model_type=model_type)
        models[model_type] = metrics

    # Find best model based on F1 score
    best_model_type = max(models.items(), key=lambda x: x[1]["f1"])[0]
    logger.info(f"Best model: {best_model_type}")

    # Retrain with best model type
    predictor.train(df, model_type=best_model_type)
    predictor.save_model()

    return predictor


if __name__ == "__main__":
    # Example usage
    train_best_model()

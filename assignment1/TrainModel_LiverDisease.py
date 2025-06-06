import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score
import pickle
import mlflow
import mlflow.sklearn


# Function to load and preprocess the dataset
def load_and_preprocess_data(file_name):
    # Load the dataset
    data = pd.read_csv(file_name)

    # Preprocessing: Replace '?' with NaN and drop rows with missing values
    data.replace('?', pd.NA, inplace=True)
    data.dropna(inplace=True)

    # Encode the target variable: 'Yes' as 1 and 'No' as 0
    data['Dataset'] = data['Dataset'].map({'Yes': 1, 'No': 0})

    # Feature selection and target variable
    X = data.drop(columns=['Dataset'])
    y = data['Dataset']
    return X, y


# Function to train the model and log metrics
def train_and_log_model(X_train, X_test, y_train, y_test, params):
    # Tracking with MLflow
    with mlflow.start_run():
        # Define the Random Forest model
        model = RandomForestClassifier(random_state=42)

        # Set up the GridSearchCV
        grid_search = GridSearchCV(estimator=model, param_grid=params,
                                   scoring='f1', cv=5, verbose=2, n_jobs=-1)

        # Fit the grid search to the training data
        grid_search.fit(X_train, y_train)

        # Get the best parameters and best score
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        print("Best Parameters found by GridSearchCV:")
        print(best_params)
        print(f"Best F1 Score: {best_score:.2f}")

        # Log parameters and metrics to MLflow
        mlflow.log_params(best_params)
        mlflow.log_metric("best_f1_score", best_score)

        # Evaluate the best model on the test data
        best_model = grid_search.best_estimator_
        predictions = best_model.predict(X_test)

        # Calculate performance metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        # Print performance metrics
        print(f"\nModel accuracy: {accuracy:.2f}")
        print(f"Model precision: {precision:.2f}")
        print(f"Model recall: {recall:.2f}")
        print(f"Model F1 score: {f1:.2f}")

        # Log performance metrics to MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Save the trained model to a file using pickle
        with open('model.pkl', 'wb') as file:
            pickle.dump(best_model, file)

        # Save metrics to a dictionary and serialize with pickle
        metrics = {
            'Best Parameters found by GridSearchCV': best_params,
            'Best F1 Score': best_score,
            'Model Accuracy': accuracy,
            'Model Precision': precision,
            'Model Recall': recall,
            'Model F1 Score': f1
        }

        with open('metrics.pkl', 'wb') as file:
            pickle.dump(metrics, file)

        print("Model and metrics saved to model.pkl and metrics.pkl")


if __name__ == "__main__":
    X, y = load_and_preprocess_data('liver_disease.csv')

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
                                        X, y, test_size=0.2,
                                        random_state=42)

    # Define the parameter grid for tuning
    param_grid = [
        {'n_estimators': [50, 100], 'max_depth': [10, None],
         'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]},
        {'n_estimators': [200], 'max_depth': [20, 30],
         'min_samples_split': [5, 10], 'min_samples_leaf': [2, 4]}
    ]

    for params in param_grid:
        train_and_log_model(X_train, X_test, y_train, y_test, params)

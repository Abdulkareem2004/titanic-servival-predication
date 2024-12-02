Python 3.11.2 (tags/v3.11.2:878ead1, Feb  7 2023, 16:38:35) [MSC v.1934 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
data = pd.read_csv("C:/Users/XMG/Downloads/archive (9)/Titanic-Dataset.csv")  # Replace with your Titanic dataset file

# Data exploration
print(data.head())
print(data.info())
print(data.describe())

# Check for missing values
print("Missing values:\n", data.isnull().sum())

# Handle missing values
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
if 'Cabin' in data.columns:
    data.drop('Cabin', axis=1, inplace=True)

# Feature engineering: Extract titles from names
def extract_title(name):
    return name.split(",")[1].split(".")[0].strip()

data['Title'] = data['Name'].apply(extract_title)

# Replace rare titles with 'Rare' and map common titles
title_mapping = {
    "Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4,
    "Dr": 5, "Rev": 5, "Col": 5, "Major": 5, "Mlle": 2,
    "Mme": 3, "Ms": 2, "Capt": 5, "Countess": 5, "Don": 5,
    "Jonkheer": 5, "Lady": 5, "Sir": 5
}
data['Title'] = data['Title'].map(title_mapping)
data['Title'].fillna(0, inplace=True)
... 
... # Drop irrelevant columns
... data.drop(['Name', 'Ticket'], axis=1, inplace=True)
... 
... # Encode categorical variables
... le = LabelEncoder()
... data['Sex'] = le.fit_transform(data['Sex'])
... data['Embarked'] = le.fit_transform(data['Embarked'])
... 
... # Define features and target
... X = data.drop(['Survived', 'PassengerId'], axis=1, errors='ignore')  # Features
... y = data['Survived']  # Target variable
... 
... # Split the data
... X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
... 
... # Scale the features
... scaler = StandardScaler()
... X_train = scaler.fit_transform(X_train)
... X_test = scaler.transform(X_test)
... 
... # Train the model
... model = RandomForestClassifier(n_estimators=100, random_state=42)
... model.fit(X_train, y_train)
... 
... # Evaluate the model
... y_pred = model.predict(X_test)
... print("Accuracy:", accuracy_score(y_test, y_pred))
... print("Classification Report:\n", classification_report(y_test, y_pred))
... print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
... 
... # Example prediction
... example_passenger = [[3, 22.0, 0, 0, 7.25, 0, 1, 2]]  # Modify with actual feature values
... example_passenger_scaled = scaler.transform(example_passenger)
... example_prediction = model.predict(example_passenger_scaled)
... print("Example Prediction (0 = Did not survive, 1 = Survived):", example_prediction[0])

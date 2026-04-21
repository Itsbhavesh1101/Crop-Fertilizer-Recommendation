import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def train_crop_model():
    print("Training Crop Recommendation Model...")
    data_path = os.path.join("data", "Crop_recommendation.csv")
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    df = pd.read_csv(data_path)
    
    # Mapping dictionary based on original code
    crop_dict = {
        'rice':1, 'maize':2, 'jute':3, 'cotton':4, 'coconut':5,
        'papaya':6, 'orange':7, 'apple':8, 'muskmelon':9, 'watermelon':10,
        'grapes':11, 'mango':12, 'banana':13, 'pomegranate':14, 'lentil':15,
        'blackgram':16, 'mungbean':17, 'mothbeans':18, 'pigeonpeas':19,
        'kidneybeans':20, 'chickpea':21, 'coffee': 22
    }
    
    df['crop_no'] = df['label'].map(crop_dict)
    
    X = df.drop(['label', 'crop_no'], axis=1)
    y = df['crop_no']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)

    # Using Random Forest for a "High-End" project as it generally performs better
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)
    
    print(f"Crop Model - Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")

    os.makedirs("models", exist_ok=True)
    with open(os.path.join("models", "crop_model.sav"), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join("models", "crop_scaler.sav"), 'wb') as f:
        pickle.dump(sc, f)
    
    # Save the dictionary for inference mapping
    inv_crop_dict = {v: k.capitalize() for k, v in crop_dict.items()}
    with open(os.path.join("models", "crop_dict.pkl"), 'wb') as f:
        pickle.dump(inv_crop_dict, f)

    print("Crop models and scaler saved successfully.\n")

def train_fertilizer_model():
    print("Training Fertilizer Recommendation Model...")
    data_path = os.path.join("data", "Fertilizer Prediction.csv")
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    df = pd.read_csv(data_path)
    
    # Target Mapping
    fert_dict = {
        'Urea':1, 'DAP':2, '14-35-14':3, '28-28':4, '17-17-17':5,
        '20-20':6, '10-26-26':7
    }
    df['fert_no'] = df['Fertilizer Name'].map(fert_dict)
    df.drop('Fertilizer Name', axis=1, inplace=True)

    # Encoding Categorical Variables
    le_soil = LabelEncoder()
    df['Soil Type'] = le_soil.fit_transform(df['Soil Type'])
    
    le_crop = LabelEncoder()
    df['Crop Type'] = le_crop.fit_transform(df['Crop Type'])

    X = df.drop('fert_no', axis=1)
    y = df['fert_no']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)
    
    print(f"Fertilizer Model - Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")

    os.makedirs("models", exist_ok=True)
    with open(os.path.join("models", "fertilizer_model.sav"), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join("models", "fertilizer_scaler.sav"), 'wb') as f:
        pickle.dump(sc, f)
    
    # Save LabelEncoders and Target Dict for inference
    inv_fert_dict = {v: k for k, v in fert_dict.items()}
    inference_artifacts = {
        "le_soil": le_soil,
        "le_crop": le_crop,
        "inv_fert_dict": inv_fert_dict
    }
    with open(os.path.join("models", "fertilizer_artifacts.pkl"), 'wb') as f:
        pickle.dump(inference_artifacts, f)

    print("Fertilizer models, scaler, and artifacts saved successfully.\n")

if __name__ == "__main__":
    train_crop_model()
    train_fertilizer_model()
    print("All training complete.")

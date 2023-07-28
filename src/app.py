import json
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel

# Load the model from the .pkl file
with open("models/regressor_house.pkl", "rb") as model_file_house:
    decision_tree_model_house = pickle.load(model_file_house)
with open("models/regressor_apartment.pkl", "rb") as model_file_apartment:
    decision_tree_model_apartment = pickle.load(model_file_apartment)
# Check if the model is loaded successfully
print("Model Loaded Successfully!")

# Initialize the FastAPI app
app_apartment = FastAPI()

class Body(BaseModel):
     rooms_number: int =0
     land_area: Optional[int]=0
     garden_area: Optional[int]=0
     terrace_area: Optional[int]=0

# Initialize the path
@app_apartment.get("/")
def start():
    return "ImmoEliza_API_deployment"

# Varifying the types of inputs
@app_apartment.post("/input")
def inputing_data(data_apartment: Body):
    try:    
        return data_apartment
    except ValueError:
       
        return {"error": "Invalid input. Please provide valid data."}


# Define the prediction endpoint
@app.post("/predict/")
def predict(data_apartment: Body):
    # Convert the input data to a numpy array for prediction
    data_apartment= json.loads(data_apartment.model_dump_json())
    input_features_apartment = pd.DataFrame(data_apartment, index=[0])
    print(input_features_apartment.head())
    #rooms_number
    #land_area
    #garden_area
    #terrace_area
    #Make the prediction using the model
    prediction_apartment = decision_tree_model_apartment.predict(input_features_apartment)

    # Return the prediction as JSON
    return {"Prediction": float(prediction_apartment)}

# Load the model from the .pkl file
with open("models/regressor_house.pkl", "rb") as model_file_house:
    decision_tree_model_house = pickle.load(model_file_house)

# Check if the model is loaded successfully
print("Model Loaded Successfully!")

# Initialize the FastAPI app
app_house = FastAPI()

class Body(BaseModel):
     rooms_number: int =0
     land_area: Optional[int]=0
     garden_area: Optional[int]=0
     terrace_area: Optional[int]=0


# Varifying the types of inputs
@app_house.post("/input for house")
def inputing_data(data_house: Body):
    try:    
        return data_house
    except ValueError:
       
        return {"error": "Invalid input. Please provide valid data."}


# Define the prediction endpoint
@app_house.post("/predict/")
def predict(data_house: Body):
    # Convert the input data to a numpy array for prediction
    data_house= json.loads(data_house.model_dump_json())
    input_features_house = pd.DataFrame(data_house, index=[0])
    print(input_features_house.head())
    #rooms_number
    #land_area
    #garden_area
    #terrace_area
    #Make the prediction using the model
    prediction_house = decision_tree_model_house.predict(input_features_house)

    # Return the prediction as JSON
    return {"Prediction for house": float(prediction_house)}






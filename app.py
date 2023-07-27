import json
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel,constr
# Load the XGBoost model from the .pkl file
with open("models/regressor_house.pkl", "rb") as model_file:
    xgb_model = pickle.load(model_file)
# Check if the model is loaded successfully
print("Model Loaded Successfully!")
# Initialize the FastAPI app
app = FastAPI()

class Body(BaseModel):
     area: int
     property_type:  str=0
     rooms_number: int =0
     zip_code: int=0
     land_area: Optional[int]=0
     garden: Optional[bool]=0
     garden_area: Optional[int]=0
     equipped_kitchen: Optional[bool]=0
     full_address: Optional[str]=0
     swimming_pool: Optional[bool]=0
     furnished: Optional[bool]=0
     open_fire: Optional[bool]=0
     terrace: Optional[bool]=0
     terrace_area: Optional[int]=0
     facades_number: Optional[int]=0
     building_state: str =0
@app.get("/")
def start():
    return "ImmoEliza_API_deployment"


@app.post("/input")
def inputing_data(data: Body):
    try:
       
        return data
    except ValueError:
       
        return {"error": "Invalid input. Please provide valid data."}
class OutputDataModel(BaseModel):
    Prediction: float

# Define the prediction endpoint
@app.post("/predict/")
async def predict(data: Body):
    # Convert the input data to a numpy array for prediction
    data= json.loads(data.model_dump_json())
    input_features = pd.DataFrame(data, index=[0])
    print(input_features.head())
    #data.Construction_year, data.Total_surface, data.Habitable_surface,
    #data.Bedroom_count, data.Furnished, data.Fireplace, data.Terrace,
    #data.Terrace_surface, data.Garden_surface, data.Facades,
    #data.SwimmingPool, data.Kitchen_equipped, data.Condition_encoded
    #]).reshape(1, -1)
    # Make the prediction using the XGBoost model
    prediction = xgb_model.predict(input_features)

    # Return the prediction as JSON
    return {"Prediction": float(prediction)}

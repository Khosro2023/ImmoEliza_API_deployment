ImmoEliza_API_deployment
This repository contains code for a Home Price Prediction Model API using FastAPI. The model is trained on a dataset of home and apartment prices and saved as a pickle file to enable easy access and usage through the API.

Getting Started
To use the Home ImmoEliza_API_deployment, follow the instructions below:

Prerequisites
Docker: Ensure that you have Docker installed on your system. If you don't have Docker installed, you can download and install it from the official Docker website.
Installation
Clone this repository to your local machine:
git clone https://github.com/Khosro2023/ImmoEliza_API_deployment
cd ImmoEliza_API_deployment

Install the required dependencies:
pip install -r requirements.txt

Start the FastAPI server:
uvicorn main:app --reload

The API should now be accessible at http://127.0.0.1:8000.

API Endpoints
The Home Price Prediction API exposes a single POST endpoint:

/predict
This endpoint allows you to estimate the price of a home or apartment based on input features. The required input features are:

rooms_number: int =0
land_area: Optional[int]=0
garden_area: Optional[int]=0
terrace_area: Optional[int]=0
Example input JSON:

json
Copy code
{
  "rooms_number": 3,
  "land_area": 300,
  "garden_area": 20,
  "terrace_area": 20,
  }
Example output JSON:

json
Copy code
{
  "predicted_price": 1120000
}
Usage
To make predictions using the API, you can send a POST request with the input features to the /predict endpoint. You can use any HTTP client, such as curl, requests in Python, or tools like Postman.

Here's an example using curl:

bash
Copy code
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "rooms_number": 3,
  "land_area": 300,
  "garden_area": 20,
  "terrace_area": 20,
}'
The API will respond with the predicted price in the JSON format.

Customization
If you want to modify the model or any other part of the API, you can do so by updating the main.py file. Make sure to test your changes thoroughly before deploying the API.

Contributing
Contributions to this project are welcome. If you find any issues or want to add new features, feel free to open a pull request.

License
This project is licensed under the MIT License - see the LICENSE file for details.


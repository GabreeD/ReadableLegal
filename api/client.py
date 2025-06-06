# import requests
# from config import API_KEY

# BASE_URL = "https://api.legiscan.com/"

# def call_legiscan_api(operation, **params):
#     # print("Using API key:", API_KEY)

#     query = {
#         "key": API_KEY,
#         "op": operation,
#         **params
#     }
#     response = requests.get(BASE_URL, params=query)
#     response.raise_for_status()
#     return response.json()
from fastapi import FastAPI
import requests
from config import API_KEY

app = FastAPI()
BASE_URL = "https://api.legiscan.com/"

def call_legiscan_api(operation: str, **params):
    query = {
        "key": API_KEY,
        "op": operation,
        **params
    }
    response = requests.get(BASE_URL, params=query)
    response.raise_for_status()
    return response.json()

@app.get("/")
def root():
    return {"message": "FastAPI is working"}
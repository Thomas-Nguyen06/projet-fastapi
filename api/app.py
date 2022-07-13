from fastapi import FastAPI
import json
import joblib


def load_model():
    return joblib.load('api/regression.joblib')


app = FastAPI(
    title="Real estate estimator",
    version="0.0.1",
)

model = load_model()

@app.get(
    "/",
    name="Index",
    summary="Returns a welcome message",
)
def read_root():
    return "Welcome"


@app.get(
    "/status",
    name="Status",
    summary="Returns OK",
)
def read_status():
    return "OK"

@app.get(
    "/predict"
)
def get_predict(taille: int,
                nb_rooms: int,
                garden: int):
    X = [[taille, nb_rooms, garden]]
    prediction = model.predict(X)
    ## afficher la prediction
    return json.dumps({"price_predicted": round(prediction[0])})

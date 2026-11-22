from fastapi import FastAPI, HTTPException, Response
from src.api.schemas import PredictionRequest, PredictionResponse
from src.api.service import PredictionService
import gradio as gr
from src.ui import demo

# Create the FastAPI application
app = FastAPI(title="Bike Rental ML API")

# Initialize the prediction service and load the model
service = PredictionService()

# Root endpoint (Provide a message to indicate that the API is running)
@app.get("/")
def root():
    return {"message": "Bike Rental ML API is running"}

# Health check endpoint (Check if the API is healthy and ready to receive requests)
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

# Prediction endpoint (Receive input features and return the predicted rental count)
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:

        # Convert request to dictionary format for the service
        request_data = {
            "date": request.date,
            "hour": request.hour,
            "temperature": request.temperature,
            "humidity": request.humidity,
            "windspeed": request.windspeed,
            "visibility": request.visibility,
            "dewpoint": request.dewpoint,
            "seasons": request.seasons,
            "holiday": request.holiday,
            "functioning_day": request.functioning_day,
            "previous_7_rented_bikes_per_hour": request.previous_7_rented_bikes_per_hour,
            "rented_bikes_same_hour_yesterday": request.rented_bikes_same_hour_yesterday,
        }

        # Run the prediction using the input features
        prediction = service.predict(request_data)

    # Return 500 if the model file is missing
    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail=str(error))

    # Return 400 if the input data is invalid
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    # Return the prediction in the expected response format
    return PredictionResponse(prediction=prediction)

# Handle manifest requests to avoid 404 errors.
# The application is a regular website and does not use PWA.
@app.get("/manifest.json", include_in_schema=False)
def manifest():
    return Response(status_code=204)

# add the made ui to the app 
app = gr.mount_gradio_app(app, demo, theme=gr.themes.Soft(), path="/ui")
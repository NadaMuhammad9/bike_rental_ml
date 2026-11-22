import pandas as pd
import gradio as gr
from src.api.service import PredictionService


# Create prediction service
prediction_service = PredictionService()


def predict_bikes(
    date,
    hour,
    temperature,
    humidity,
    windspeed,
    visibility,
    dewpoint,
    seasons,
    holiday,
    functioning_day,
    previous_7_rented_bikes_per_hour,
    rented_bikes_same_hour_yesterday,
):
    try:
        # Convert the 7 previous rented bikes into a list of numbers
        rented_bikes = [
            int(x.strip())
            for x in previous_7_rented_bikes_per_hour.split(",")
            if x.strip()
        ]

        # Validate exactly 7 values
        if len(rented_bikes) != 7:
            raise ValueError(
                "Please enter exactly 7 previous rented bikes values."
            )

        # Prepare input data
        data = {
            "date": pd.to_datetime(date),
            "hour": int(hour),
            "temperature": float(temperature),
            "humidity": float(humidity),
            "windspeed": float(windspeed),
            "visibility": float(visibility),
            "dewpoint": float(dewpoint),
            "seasons": seasons,
            "holiday": holiday,
            "functioning_day": functioning_day,

            "previous_7_rented_bikes_per_hour": rented_bikes,

            "rented_bikes_same_hour_yesterday": int(
                rented_bikes_same_hour_yesterday
            ),
        }

        # Make prediction
        prediction = prediction_service.predict(data)

        return prediction

    except Exception as e:
        return f"Error: {str(e)}"



with gr.Blocks(title="Bike Rental Prediction") as demo:

    gr.Markdown(
        """
        # 🚲 Bike Rental Prediction

        Enter the weather, time, and previous rental information
        to predict the number of rented bikes per hour.

        (previous rental information is used to calculate lag/rolling)
        
        (the model more responsive to periods of high demand, helping ensure sufficient bike availability and reduce the risk of stockouts, and may slightly overestimate demand on typical days)
        """
    )

    with gr.Row():

        date = gr.DateTime(
            label="Date",
            include_time=False,
            type="string",
            value="2024-06-03",
        )

        hour = gr.Slider(
            minimum=0,
            maximum=23,
            step=1,
            value=18,
            label="Hour of the Day (0-23)",
        )

        temperature = gr.Number(
            value=23,
            label="Temperature (°C)",
        )

        humidity = gr.Number(
            value=55,
            label="Humidity (%)",
        )

    with gr.Row():
        windspeed = gr.Number(
            value=2,
            label="Wind Speed (m/s)",
        )

        visibility = gr.Number(
            value=1800,
            label="Visibility (10m)",
        )

        dewpoint = gr.Number(
            value=13,
            label="Dew Point Temperature (°C)",
        )

    with gr.Row():
        seasons = gr.Dropdown(
            choices=[
                "Spring",
                "Summer",
                "Autumn",
                "Winter",
            ],
            value="Summer",
            label="Season",
        )

        holiday = gr.Dropdown(
            choices=[
                "Holiday",
                "No Holiday",
            ],
            value="No Holiday",
            label="Holiday",
        )

        functioning_day = gr.Dropdown(
            choices=[
                "Yes",
                "No",
            ],
            value="Yes",
            label="Functioning Day",
        )

    gr.Markdown(
        """
        ### Previous 7 Hours

        Enter the rented bikes Count for the previous 7 hours,
        separated by commas.

        Example:
        `300, 340, 390, 450, 500, 560, 600`
        """
    )

    previous_7_rented_bikes_per_hour = gr.Textbox(
        label="Previous 7 Rented Bikes Count Per Hour",
        placeholder="300, 340, 390, 450, 500, 560, 600",
        value="300, 340, 390, 450, 500, 560, 600",
    )

    rented_bikes_same_hour_yesterday = gr.Number(
        value=630,
        label="Rented Bikes at Same Hour Yesterday",
    )

    predict_button = gr.Button(
        "Predict 🚲",
        variant="primary",
    )

    prediction_output = gr.Textbox(
        label="Predicted Rented Bikes Per Hour",
    )

    predict_button.click(
        fn=predict_bikes,
        inputs=[
            date,
            hour,
            temperature,
            humidity,
            windspeed,
            visibility,
            dewpoint,
            seasons,
            holiday,
            functioning_day,
            previous_7_rented_bikes_per_hour,
            rented_bikes_same_hour_yesterday,
        ],
        outputs=prediction_output,
    )


if __name__ == "__main__":
    demo.launch()
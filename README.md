# Bike Rental ML

## project Overview
An end-to-end machine learning project for hourly bike demand forecasting ,including feature engineering, hyperparameter optimization, experiment tracking, API ,UI 

## Project Approach 

Although time-series forecasting can be handled effectively using **time-series neural networks**, such models often require large amounts of data, higher computational resources, and longer training times. For this project, I chose a **classical machine learning approach combined with feature engineering** to build an efficient and practical forecasting model.

Since traditional machine learning models do not automatically understand the temporal relationships within the data, I created several time-based features to help the model learn recurring patterns.

 These features include :
 - **lag features** : which provide information about previous bike demand
 - **rolling features** : which capture recent demand trends and averages
 - **date features** :  date and time features such as day, month, year and week of the year are extracted from data column
- **cyclical features** : periodic representations of hour

By combining these engineered time-series features with a classical machine learning algorithm, the model can learn important  temporal patterns.

Overall, the project demonstrates how effective feature engineering can transform time-series data into a format suitable for classical machine learning, providing a balance between forecasting performance, computational efficiency, and model simplicity.


## Project Workflow

- **Reproducible ML pipeline**: Built an end-to-end training pipeline for the Seoul Bike Sharing Demand dataset, with configuration managed through `config/config.yaml`.

- **Chronological preprocessing**: Parsed date and time information, removed unused precipitation and radiation columns, encoded binary fields, one-hot encoded seasons, and sorted records chronologically by date and hour.

- **Time-series feature engineering**: Created calendar features, cyclical hour encoding, demand lag features, and rolling statistics to capture temporal patterns and recent demand trends.

- **LightGBM model**: Trained a LightGBM regression model using a chronological time-based split, ensuring that training observations always precede test observations and preventing data leakage from future observations. The model uses a fixed random seed of `42` and tuned values for estimators, learning rate, depth, leaves, row subsampling, and feature subsampling.

- **Hyperparameter tuning**: Implemented an optional Optuna tuner in `src/models/tune.py` that searches LightGBM hyperparameters using `TimeSeriesSplit` cross-validation and early stopping. The optimization minimizes mean validation RMSE and starts from parameters identified during exploratory analysis.

- **Model evaluation**: Evaluated model performance using MAE, MSE, RMSE, and R², with non-negative predictions.

- **MLflow experiment tracking**: Recorded training runs in the local SQLite tracking store at `mlruns/mlruns.db`, including parameters, metrics, processed data, and the trained model. Each training result also reports the model path, processed data path, evaluation metrics, and MLflow run ID.

- **FastAPI service**: Built a prediction API with health checks, request validation, automatic OpenAPI documentation, and prediction responses.

- **Gradio UI**: Added an interactive forecasting interface.

- **Testing**: Added unit tests covering preprocessing, feature engineering, model training, and API health and validation behavior.

- **Docker supported**: Containerize the application for consistent execution. Exposes port 8000 for the FastAPI service, starting Uvicorn on 0.0.0.0:8000 when the container runs.

- **GitHub Actions CI**: Automates the CI pipeline by automatically running tests whenever code is pushed or a pull request is created.


## Model Evaluation Results

Evaluation metrics: **{'mae': 77.7272830927696, 'mse': 14298.340212838728, 'rmse': 119.57566731086524, 'r2': 0.9619444696923234}**

The model explains approximately `96%` of the variance in rental demand on test data
  



## A problem and a solution
Time-series features such as lag and rolling features require historical demand values to capture temporal patterns.
In a real-time system, these values would normally come from sources such as sensors, databases, or APIs. Since this project uses a specific dataset. the prediction API requires these historical inputs to be passed directly in the request:


{
  "previous_7_rented_bikes_per_hour": [145, 278, 320, 295, 342, 286, 330],
  "rented_bikes_same_hour_yesterday": 305
}

(numbers are for example)



## Useful commands (or take a look at the Makefile)

Install pinned Python dependencies 
```bash
python -m pip install -r requirements.txt
```

Handles the complete model training pipeline, including data preprocessing, feature engineering, training, evaluation, model saving, and MLflow experiment tracking
```bash
python -m src.models.train
```

Run the pytest suite quietly
```bash
python -m pytest -q
```

Start FastAPI with Uvicorn reload enabled
```bash
python -m uvicorn src.app.main:app --reload
```

Build the Docker image
```bash
docker build -t bike-rental .
```

Run the Docker image
```bash
docker run -p 8000:8000	bike-rental
```
## Dataset & Attribution

This project uses the **Seoul Bike Sharing Demand** dataset, obtained from the **UCI Machine Learning Repository**.
**UCI dataset license:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)


@misc{seoul_bike_sharing_demand_560,
  author       = {Easwaramoorthy, Sathishkumar Veerappampalayam },
  title        = {{Seoul Bike Sharing Demand}},
  year         = {2020},
  howpublished = {UCI Machine Learning Repository},
  note         = {{DOI}: https://doi.org/10.24432/C5F62R}
}

Please cite as the Introductory Papers: Sathishkumar, V. E., Park, J., & Cho, Y. (2020). Using data mining techniques for bike sharing demand prediction in metropolitan city. Computer Communications, 153, 353–366. https://doi.org/10.1016/j.comcom.2020.02.007 Sathishkumar, V. E., & Cho, Y. (2020). A rule-based model for Seoul Bike sharing demand prediction using weather data. European Journal of Remote Sensing, 53(sup1), 166–183. https://doi.org/10.1080/22797254.2020.1725789

**Modifications**: The dataset was cleaned and transformed for machine learning. Unused columns were dropped, categorical and binary features were encoded, records were sorted chronologically, and additional calendar, cyclical, lag, and rolling features were derived from the data.

 Original data sources:
* **Seoul Open Data Plaza:** https://data.seoul.go.kr 
* **Korea Meteorological Administration (KMA):** https://data.kma.go.kr


> This project is not affiliated with or endorsed by the UCI Machine Learning Repository, Seoul Open Data Plaza, or Korea Meteorological Administration.

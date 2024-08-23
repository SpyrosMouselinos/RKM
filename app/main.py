import json

import pandas as pd
import torch
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import os
import shutil

from app.inference import inference
from app.model_manager import ModelServer
from app.modeling.SimpleForcaster import SimpleForcaster
from app.utils.data_processing import RealTimeTimeSeriesDataset
from training import train_model

app = FastAPI()

# Global variable to keep track of the loaded model
active_model = None

# Global variable to keep track of the current inference data processing
inference_data_processor = RealTimeTimeSeriesDataset(None,
                                                     None,
                                                     0,
                                                     '1Y')

# Mount the static directory to serve static files like CSS
app.mount("/static", StaticFiles(directory="./static"), name="static")

# Set up template directory
templates = Jinja2Templates(directory="./templates")

# Define the directories for models and active model
MODEL_DIR = "./models"
ACTIVE_MODEL_DIR = "./active_model"


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # List all models
    models = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pth')]

    # Initialize an empty dictionary for model metrics
    model_metrics = {}

    # Iterate through the models and read their respective JSON files
    for model in models:
        json_path = os.path.join(MODEL_DIR, model.replace('.pth', '.json'))
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                metrics = json.load(f)
                model_metrics[model] = {
                    "mape": metrics.get("mape", "N/A"),
                    "loss": metrics.get("loss", "N/A")
                }
        else:
            model_metrics[model] = {
                "mape": "N/A",
                "loss": "N/A"
            }

    # Identify the active model and its metrics
    active_model_path = os.path.join(ACTIVE_MODEL_DIR, "best_model.pth")
    active_model = os.path.basename(active_model_path) if os.path.exists(active_model_path) else None
    active_model_metrics = {}

    if active_model:
        json_path = os.path.join(ACTIVE_MODEL_DIR, "best_model.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                active_model_metrics = json.load(f)

    # Render the index.html template with the model data
    return templates.TemplateResponse("index.html", {
        "request": request,
        "models": models,
        "model_metrics": model_metrics,
        "active_model": active_model,
        "active_model_metrics": active_model_metrics
    })


@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...),
                     m_name: str = Form(...),
                     mode: str = Form(...),
                     sequence_length: int = Form(...),
                     target_offset: int = Form(...),
                     batch_size: int = Form(...),
                     num_epochs: int = Form(...),
                     learning_rate: float = Form(...),
                     impute_backward: int = Form(...),
                     group_by: str = Form(...),
                     eval_every: int = Form(...),
                     early_stopping_patience: int = Form(...)):
    # Save the uploaded CSV file
    file_path = os.path.join("./data", file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Start training the model with the provided parameters
    model_save_path = os.path.join(MODEL_DIR, m_name)
    train_model(csv_file=file_path,
                mode=mode,
                model_save_path=model_save_path,
                sequence_length=sequence_length,
                target_offset=target_offset,
                batch_size=batch_size,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                impute_backward=impute_backward,
                group_by=group_by,
                eval_every=eval_every,
                early_stopping_patience=early_stopping_patience)

    # Redirect back to the home page
    return RedirectResponse("/", status_code=303)


@app.post("/model/{model_name}/activate")
async def activate_model(model_name: str):
    global active_model

    # Ensure there is no other active model before activation
    if active_model is not None:
        return {"error": "Another model is already active."}

    # Load the model and move it to the active model directory
    model_path = os.path.join(MODEL_DIR, model_name)
    json_path = model_path.replace('.pth', '.json')

    if not os.path.exists(ACTIVE_MODEL_DIR):
        os.makedirs(ACTIVE_MODEL_DIR)

    shutil.copy(model_path, os.path.join(ACTIVE_MODEL_DIR, "best_model.pth"))
    shutil.copy(json_path, os.path.join(ACTIVE_MODEL_DIR, "best_model.json"))

    # Redirect back to the home page
    return RedirectResponse("/", status_code=303)


@app.post("/model/{model_name}/start")
async def start_model():
    global inference_data_processor
    # Load the model from the active model directory
    model_path = os.path.join(ACTIVE_MODEL_DIR, "best_model.pth")
    model = SimpleForcaster.load_from_checkpoint(model_path, device=torch.device("cpu"))

    # Load its configuration json to get the features and their mean / variance values
    json_path = os.path.join(ACTIVE_MODEL_DIR, "best_model.json")
    with open(json_path, 'r') as f:
        config = json.load(f)

    # List of feature names
    feautre_names = []

    # New configuration file, containing only the features and their mean / variance values
    feature_stats_config = {}

    # Go over the items in the config, if they contain _mean or _std, load them
    for k, v in config.items():
        if "_mean" in k or "_std" in k:
            # Take the key name before _mean or _std
            k = k.split("_")[0]
            # Add it to the list of feature names, if it does not already exist
            if k not in feautre_names:
                feautre_names.append(k)
            # Make an dictionary entry with this name
            feature_stats_config[k] = torch.tensor(v)

    # Signature of a fixed cpu input #
    fixed_input = torch.randn(1, 10, model.model.input_size, device=torch.device("cpu"))

    # ONNX Conversion Process
    model_server = ModelServer(model=model, device=torch.device("cpu"))
    model_server.convert_to_onnx(fixed_input)

    # Set up the inference data processor
    inference_data_processor = RealTimeTimeSeriesDataset(feature_names=feautre_names,
                                                         feature_stats=feature_stats_config)

    # Return the response to update the frontend
    return {"status": "online"}


@app.post("/model/{model_name}/stop")
async def stop_model(model_name: str):
    # Logic to stop the model (e.g., remove it from the GPU or stop serving it)
    # In this placeholder, we simply redirect back to the home page.
    return RedirectResponse("/", status_code=303)


@app.post("/model/{model_name}/deactivate")
async def deactivate_model(model_name: str):
    global active_model

    # Deactivate the active model and delete it from the directory
    model_path = os.path.join(ACTIVE_MODEL_DIR, "best_model.pth")
    json_path = os.path.join(ACTIVE_MODEL_DIR, "best_model.json")

    if os.path.exists(model_path):
        os.remove(model_path)
    if os.path.exists(json_path):
        os.remove(json_path)

    # Clear the active model from memory
    active_model = None

    # Redirect back to the home page
    return RedirectResponse("/", status_code=303)


@app.post("/model/infer")
async def infer(request: Request):
    global active_model, inference_data_processor

    # Check if there is an active model
    if active_model is None:
        return {"error": "No active model for inference."}

    # Parse the JSON payload from the request
    request_data = await request.json()

    # Impute missing data in the incoming data point
    request_data = inference_data_processor.impute_missing(request_data)

    # Update the buffer with the scaled data point and remove old data
    inference_data_processor.update_buffer(request_data)

    # Get sequences ready for prediction
    sequences, needed_points = inference_data_processor.get_current_sequences()

    if sequences is None:
        return {"error": f"Not enough data for inference. Need {needed_points} more points."}

    # Perform inference for each sequence
    predictions = []
    for sequence in sequences:
        pass
    return {"predictions": predictions}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)

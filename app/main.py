# FastAPI backend and API routes
import json

import torch
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import os
import shutil
from training import train_model

app = FastAPI()

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
                    "accuracy": metrics.get("accuracy", "N/A"),
                    "loss": metrics.get("loss", "N/A")
                }
        else:
            model_metrics[model] = {
                "accuracy": "N/A",
                "loss": "N/A"
            }

    # Identify the active model
    active_model_path = os.path.join(ACTIVE_MODEL_DIR, "best_model.pth")
    active_model = os.path.basename(active_model_path) if os.path.exists(active_model_path) else None

    # Render the index.html template with the model data
    return templates.TemplateResponse("index.html", {
        "request": request,
        "models": models,
        "model_metrics": model_metrics,
        "active_model": active_model
    })


@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...), m_name: str = Form(...)):
    # Save the uploaded CSV file
    file_path = os.path.join("./data", file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Start training the model
    model_save_path = os.path.join(MODEL_DIR, m_name)
    train_model(csv_file=file_path, model_save_path=model_save_path)

    # Redirect back to the home page
    return RedirectResponse("/", status_code=303)


@app.post("/model/{model_name}/activate")
async def activate_model(model_name: str):
    # Ensure there is no other active model before activation
    active_model_path = os.path.join(ACTIVE_MODEL_DIR, "best_model.pth")
    if os.path.exists(active_model_path):
        return RedirectResponse("/", status_code=303)

    # Quantize the model (if needed) and move it to the active model directory
    model_path = os.path.join(MODEL_DIR, model_name)
    json_path = model_path.replace('.pth', '.json')

    # Copy the model and JSON file to the active model directory
    if not os.path.exists(ACTIVE_MODEL_DIR):
        os.makedirs(ACTIVE_MODEL_DIR)

    shutil.copy(model_path, os.path.join(ACTIVE_MODEL_DIR, "best_model.pth"))
    shutil.copy(json_path, os.path.join(ACTIVE_MODEL_DIR, "best_model.json"))

    # Redirect back to the home page
    return RedirectResponse("/", status_code=303)


@app.post("/model/{model_name}/start")
async def start_model(model_name: str):
    # Load the model onto the GPU (if available)
    model_path = os.path.join(ACTIVE_MODEL_DIR, "best_model.pth")
    model = torch.load(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # You can add more logic here to start serving the model, if needed.

    # Redirect back to the home page
    return RedirectResponse("/", status_code=303)


@app.post("/model/{model_name}/stop")
async def stop_model(model_name: str):
    # Logic to stop the model (e.g., remove it from the GPU or stop serving it)
    # In this placeholder, we simply redirect back to the home page.
    return RedirectResponse("/", status_code=303)


@app.post("/model/{model_name}/deactivate")
async def deactivate_model(model_name: str):
    # Deactivate the active model and delete it from the directory
    model_path = os.path.join(ACTIVE_MODEL_DIR, "best_model.pth")
    json_path = os.path.join(ACTIVE_MODEL_DIR, "best_model.json")

    if os.path.exists(model_path):
        os.remove(model_path)
    if os.path.exists(json_path):
        os.remove(json_path)

    # Redirect back to the home page
    return RedirectResponse("/", status_code=303)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)

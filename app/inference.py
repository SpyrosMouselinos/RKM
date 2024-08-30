# Python script for inference with active model
import onnxruntime
import numpy as np
from typing import Dict
from app.utils.data_processing import RealTimeTimeSeriesDataset


def inference(incoming_data: Dict,
              rttsd: RealTimeTimeSeriesDataset,
              onnx_session: onnxruntime.InferenceSession):
    """
    A function that performs the following:

    1. Load the active model
    2. Accumulate enough datapoints in the RealTimeTimeSeriesDataset
    3. Get the sequences ready for prediction
    4. Make predictions with onnx model
    5. Return the predictions
    6. Make post-prediction sanity checks
    7. Format results as a Json-style payload to be sent back to the sender
    :param incoming_data: The Json-Style incoming data as taken from the POST request
    :param rttsd: Instance of the RealTimeTimeSeriesDataset
    :param onnx_session: Instance of the active onnx model
    :return: Json-Style results
    """

    # Impute missing data in the incoming data point
    incoming_data = rttsd.impute_missing(incoming_data)

    # Update the buffer with the scaled data point and remove old data
    rttsd.update_buffer(incoming_data)

    # Get sequences ready for prediction
    sequences, needed_points = rttsd.get_current_sequences()

    if sequences is None:
        return {"error": f"Not enough data for inference. Need {needed_points} more points."}

    # Perform inference for each sequence
    predictions = []
    for seq in sequences:
        inputs = {onnx_session.get_inputs()[0].name: seq.astype(np.float32)}
        pred_onnx = onnx_session.run(None, inputs)
        predictions.append(pred_onnx[0])

    # Post-prediction sanity checks
    for pred in predictions:
        # Check for NaNs
        if np.isnan(pred).any():
            return {"error": "Prediction contains NaNs."}

        # Check for violent differences
        if np.max(pred) - np.min(pred) > 100:  # Adjust the threshold as needed
            return {"error": "Prediction values differ too violently."}

    # Format results as a JSON-style payload to be sent back to the sender
    return {"predictions": predictions}
import numpy as np
import torch
import onnxruntime
import gc

class ModelServer:
    """
    A class to manage and serve a PyTorch model converted to ONNX format for inference.

    Attributes:
        model (torch.nn.Module): The PyTorch model to be converted and served.
        target_offset (int): The target offset for the model, used in many-to-many mode.
        mode (str): The mode of the model ('many_to_one' or 'many_to_many').
        model_path (str): The file path where the ONNX model will be saved.
        device (torch.device): The device to which the model and data should be moved.
        execution_provider (str): The ONNX runtime execution provider ('CPUExecutionProvider' or 'CUDAExecutionProvider').
        onnx_session (onnxruntime.InferenceSession): The ONNX runtime inference session.
    """

    def __init__(self, model, model_path="./active_model/active_model.onnx", device=None):
        """
        Initializes the ModelServer with a given PyTorch model and device.

        @param model: The PyTorch model to be managed.
        @param model_path: Path where the ONNX model will be saved.
        @param device: The device to move the model to. Defaults to CUDA if available, otherwise CPU.
        """
        self.model = model
        self.target_offset = self.model.target_offset
        self.mode = model.mode
        self.model_path = model_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.execution_provider = "CPUExecutionProvider" if self.device == torch.device("cpu") else "CUDAExecutionProvider"
        self.onnx_session = None

        # Move model to the appropriate device
        self.model.to(self.device)

    def convert_to_onnx(self, input_tensor):
        """
        Converts the PyTorch model to ONNX format and saves it to the specified path.

        @param input_tensor: A tensor that will be used to trace the model for ONNX export.
        """
        internal_model = self.model.model
        internal_model.eval()
        # Ensure input tensor is on the same device as the model
        input_tensor = input_tensor.to(self.device)
        if self.mode == 'many_to_one':
            print("Assuming many_to_one model conversion...\n")
            args = (input_tensor, None)
        elif self.mode == 'many_to_many':
            target_offset = self.target_offset
            print("Assuming many_to_many model conversion...\n")
            args = (input_tensor, None, target_offset)
        else:
            raise ValueError('mode must be one of "many_to_one" or "many_to_many"')

        torch.onnx.export(internal_model,
                          args,
                          self.model_path,
                          export_params=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=['input'],
                          output_names=['output'])

        print(f"Model converted to ONNX and saved to {self.model_path}")

        # Unload the Pytorch Model
        del self.model
        if self.device == torch.device("cuda"):
            torch.cuda.empty_cache()
        gc.collect()
        self.model = None

        # Load ONNX model into memory with GPU support
        self.onnx_session = onnxruntime.InferenceSession(self.model_path,
                                                         providers=[self.execution_provider])

        # Print inputs for debugging
        for input_meta in self.onnx_session.get_inputs():
            print("Input Name:", input_meta.name)
            print("Input Type:", input_meta.type)
            print("Input Shape:", input_meta.shape)

    def load_onnx_model(self):
        """
        Loads the ONNX model from the file into the ONNX runtime inference session.

        If the ONNX model is already loaded, this method does nothing.
        """
        if self.onnx_session is None:
            # Load ONNX model with GPU support
            self.onnx_session = onnxruntime.InferenceSession(self.model_path,
                                                             providers=[self.execution_provider])

    def infer(self, input_tensor, future_steps=None):
        """
        Performs inference using the ONNX model on the given input tensor.

        @param input_tensor: The input tensor for which to perform inference.
        @param future_steps: Number of future steps to predict in 'many_to_many' mode. Defaults to the model's target offset.
        @return: The tensor containing the inference results.
        """
        # Ensure the input tensor is on the appropriate device (GPU if available)
        input_tensor = input_tensor.to(self.device)

        # Prepare inputs for ONNX model
        onnx_inputs = {
            self.onnx_session.get_inputs()[0].name: input_tensor.cpu().numpy(),
        }

        if self.mode == 'many_to_many':
            # Many_to_many model input
            if future_steps is None:
                future_steps = self.target_offset
            onnx_inputs[self.onnx_session.get_inputs()[1].name] = future_steps

        # Perform inference using the GPU (if CUDAExecutionProvider is set)
        onnx_outputs = self.onnx_session.run(None, onnx_inputs)

        # Convert output back to CPU
        output_tensor = torch.tensor(onnx_outputs[0]).to('cpu')
        return output_tensor


def test_initialize_and_convert():
    """
    Tests the initialization, conversion of a PyTorch model to ONNX, and compares the performance between
    unoptimized GPU inference and optimized CPU inference using ONNX.

    This function benchmarks the average time for inference on GPU (without ONNX) and on CPU (with ONNX).
    """
    # Set a fixed seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Test the model initialization and conversion
    import time
    from app.modeling.SimpleForcaster import SimpleForcaster

    INPUT_SIZE = 25
    HIDDEN_SIZE = 256
    OUTPUT_SIZE = 10
    BATCH_SIZE = 1
    SEQUENCE_LENGTH = 256
    LOOKAHEAD = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the unoptimized model
    unoptimized_model = SimpleForcaster(mode='many_to_one',
                                        input_size=INPUT_SIZE,
                                        hidden_size=HIDDEN_SIZE,
                                        output_size=OUTPUT_SIZE,
                                        device=device)

    # Create a fixed input tensor
    fixed_input = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE, device=device)

    # Time the GPU inference (unoptimized, no ONNX)
    start = time.time()
    for i in range(100):
        output_gpu = unoptimized_model(fixed_input)
    end = time.time()
    print("\nUnoptimized (GPU) Average time per result: ", (end - start) / 100)

    # Convert model to ONNX and load it for CPU inference
    device = torch.device("cpu")
    unoptimized_model.to(device)
    model_server = ModelServer(unoptimized_model, device=device)
    model_server.convert_to_onnx(input_tensor=fixed_input)

    # Time the CPU inference (with ONNX)
    start = time.time()
    for i in range(100):
        output_cpu = model_server.infer(fixed_input.to('cpu'), LOOKAHEAD)
    end = time.time()
    print("\nOptimized (CPU with ONNX) Average time per result: ", (end - start) / 100)

    # Compare the outputs from GPU (non-ONNX) and CPU (ONNX)
    output_gpu = output_gpu.to(device)
    if torch.allclose(output_gpu, output_cpu, atol=1e-5):
        print("\nThe outputs from GPU and CPU are very close!")
    else:
        print("\nThe outputs from GPU and CPU differ.")


def test_equivalent_outputs():
    """
    Tests the equivalence of outputs from GPU inference (non-ONNX) and CPU inference (ONNX).
    It compares the results of a pretrained model on a validation dataset using both methods.

    This function verifies that the outputs of the GPU-based model and the ONNX model are similar within a tolerance.
    """
    import json
    import pandas as pd
    from app.utils.data_processing import find_and_convert_date_column, TimeSeriesImputationDataset
    from app.modeling.SimpleForcaster import SimpleForcaster
    from torch.utils.data import DataLoader

    # Assumes a pretrained model is available
    PRETRAINED_MODEL_PATH = "./models/model_4.pth"
    PRETRAINED_MODEL_CONFIG = PRETRAINED_MODEL_PATH.replace(".pth", ".json")
    DEVICE = 'cuda'

    # Load the model
    model = SimpleForcaster.load_from_checkpoint(checkpoint_path=PRETRAINED_MODEL_PATH, device=DEVICE)

    # Assumes the existence of the training dataset in the /data/ folder
    TRAINING_DATA_PATH = "./data/yahoo_stock.csv"

    # Load the data from the CSV
    data = find_and_convert_date_column(pd.read_csv(TRAINING_DATA_PATH))

    # Extract the features and timestamps
    timestamps = data["date"].values
    features = data.drop("date", axis=1)

    # Write a function that sorts the names of the columns alphabetically
    features = features.reindex(sorted(features.columns), axis=1)

    # Save the order of the columns in a list
    column_order = list(features.columns)

    # Convert it to numpy
    features = features.values

    # Initialize the dataset
    dataset = TimeSeriesImputationDataset(features,
                                          timestamps,
                                          10,
                                          column_names=column_order,
                                          group_by='24H',
                                          target_offset=2,
                                          impute_backward=1)

    # Split data into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Evaluate it on the validation set
    outputs_gpu = []
    outputs_onnx = []
    model.eval()
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(DEVICE)
            output = model(inputs).to('cpu').numpy()
            # Output shape: (1, 2, 1)
            # Reshape a flat list of batch_size items and append each one to the outputs list
            outputs_gpu.extend(output.reshape(-1))

    # Now convert it to ONNX as above and compare the outputs
    device = torch.device("cpu")
    model.to(device)
    model_server = ModelServer(model, device=device)
    model_server.convert_to_onnx(input_tensor=inputs)

    for inputs, targets in val_loader:
        output = model_server.infer(inputs.to('cpu'))
        outputs_onnx.extend(output.reshape(-1))

    # Convert the lists into numpy arrays
    outputs_gpu = np.array(outputs_gpu)
    outputs_onnx = np.array(outputs_onnx)

    # Compare the outputs from GPU (non-ONNX) and CPU (ONNX) in numpy
    if np.allclose(outputs_gpu, outputs_onnx, atol=1e-5):
        print("\nThe outputs from GPU inference and ONNX inference are very close!")
    else:
        print("\nThe outputs from GPU inference and ONNX inference differ.")

    print("GPU Outputs: {}".format(outputs_gpu[:10]))
    print("ONNX Outputs: {}".format(outputs_onnx[:10]))

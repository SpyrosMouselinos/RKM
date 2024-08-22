import numpy as np
import torch
import onnxruntime
import gc


class ModelServer:
    def __init__(self, model, model_path="./active_model/active_model.onnx", device=None):
        self.model = model
        self.future_steps = self.model.future_steps
        self.mode = model.mode
        self.model_path = model_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.execution_provider = "CPUExecutionProvider" if self.device == torch.device(
            "cpu") else "CUDAExecutionProvider"
        self.onnx_session = None

        # Move model to the appropriate device
        self.model.to(self.device)

    def convert_to_onnx(self, input_tensor):
        internal_model = self.model.model
        internal_model.eval()
        # Ensure input tensor is on the same device as the model
        input_tensor = input_tensor.to(self.device)
        if self.mode == 'many_to_one':
            print("Assuming many_to_one model conversion...\n")
            args = (input_tensor, None)
        elif self.mode == 'many_to_many':
            target_offset = self.model.target_offset
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

        # Unload the Pytorch Model #
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
        if self.onnx_session is None:
            # Load ONNX model with GPU support
            self.onnx_session = onnxruntime.InferenceSession(self.model_path,
                                                             providers=[self.execution_provider])

    def infer(self, input_tensor, future_steps=None):

        # Ensure the input tensor is on the appropriate device (GPU if available)
        input_tensor = input_tensor.to(self.device)

        # Here its tricky, remember we have 2 inputs for Many to Many model
        # ONNX requires numpy arrays as inputs

        # Default Many_to_one model input
        onnx_inputs = {
            self.onnx_session.get_inputs()[0].name: input_tensor.cpu().numpy(),
        }

        if self.mode == 'many_to_many':
            # Many_to_many model input
            if future_steps is None:
                future_steps = self.model.future_steps
            onnx_inputs[self.onnx_session.get_inputs()[1].name] = future_steps

        # Perform inference using the GPU (if CUDAExecutionProvider is set)
        onnx_outputs = self.onnx_session.run(None, onnx_inputs)

        # Convert output back to CPU
        output_tensor = torch.tensor(onnx_outputs[0]).to('cpu')
        return output_tensor


def test_initialize_and_convert():
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

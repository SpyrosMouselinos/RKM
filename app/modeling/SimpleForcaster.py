from typing import Iterator
import torch
from torch.nn import Parameter


class RecurrentBase(torch.nn.Module):
    """
    @brief Base class for recurrent neural network modules.

    This class defines a recurrent neural network architecture that should be inherited and where the forward
     method should be implemented or overridden by subclasses.
    """

    def __init__(self, input_size=3, hidden_size=256, output_size=1):
        """
        Initializes the RecurrentBase module with specified input, hidden, and output sizes.

        @param input_size: The number of input features per timestep.
        @param hidden_size: The size of the hidden layer.
        @param output_size: The size of the output layer.
        """
        super(RecurrentBase, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.recurrent = torch.nn.GRU(input_size=input_size,
                                      hidden_size=hidden_size,
                                      num_layers=1,
                                      bias=True,
                                      batch_first=True,
                                      bidirectional=False)
        self.output = torch.nn.Linear(in_features=hidden_size,
                                      out_features=output_size,
                                      bias=True)

    def forward(self, *args, **kwargs):
        """
        Forward pass for the RecurrentBase module, to be implemented in subclasses.

        @return: Outputs from the final layer of the network.
        """
        pass


class ManyToOneRecurrentBase(RecurrentBase):
    """
    @brief Specialized class for a many-to-one recurrent network.

    This class is designed to predict a single output from a sequence of inputs.
    """

    def __init__(self, input_size=3, hidden_size=256, output_size=1):
        """
        Initializes the ManyToOneRecurrentBase module.

        @param input_size: The number of input features per timestep.
        @param hidden_size: The size of the hidden layer.
        @param output_size: The size of the output layer.
        """
        super(ManyToOneRecurrentBase, self).__init__(input_size=input_size,
                                                     hidden_size=hidden_size,
                                                     output_size=output_size)

    def forward(self, x, init_state=None):
        """
        Processes input data through the GRU layer and extracts the last timestep's output for prediction.

        @param x: Input data, a tensor of shape (batch_size, sequence_length, input_size).
        @param init_state: Initial hidden state for the GRU layer.
        @return: Output tensor for the last timestep.
        """
        out, _ = self.recurrent(x, init_state)
        return self.output(out[:, -1:, :])

    def to(self, device):
        """
        Moves all model parameters and buffers to the specified device.

        @param device: The device (e.g., "cpu", "cuda") to move the tensors to.
        """
        super(ManyToOneRecurrentBase, self).to(device)


class ManyToManyRecurrentBase(RecurrentBase):
    """
    @brief Specialized class for a many-to-many recurrent network.

    This class is designed to predict multiple future timesteps from a sequence of inputs.
    """

    def __init__(self, input_size=3, hidden_size=256, output_size=1, trainable_dummy_token=False):
        """
        Initializes the ManyToManyRecurrentBase module with an option for a trainable dummy token.

        @param input_size: The number of input features per timestep.
        @param hidden_size: The size of the hidden layer.
        @param output_size: The size of the output layer.
        @param trainable_dummy_token: A boolean flag that if true, initializes a trainable dummy token.
        """
        super(ManyToManyRecurrentBase, self).__init__(input_size=input_size,
                                                      hidden_size=hidden_size,
                                                      output_size=output_size)
        if trainable_dummy_token:
            self.dummy_token = torch.nn.Parameter(torch.randn(1, 1, self.input_size))
        else:
            self.dummy_token = torch.zeros(1, 1, self.input_size)

    def forward(self, x, init_state=None, future_steps=2):
        """
        Processes input data through the GRU layer for multiple steps into the future.

        @param x: Input data, a tensor of shape (batch_size, sequence_length, input_size).
        @param init_state: Initial hidden state for the GRU layer.
        @param future_steps: Number of future steps to predict.
        @return: Output tensor for multiple future steps.
        """
        batch_size = x.size(0)
        if init_state is None:
            h0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        else:
            h0 = init_state
        outputs = []
        out, h = self.recurrent(x, h0)
        out = self.output(out[:, -1:, :])
        outputs.append(out)
        out, h = self.recurrent(self.dummy_token.repeat(batch_size, future_steps - 1, 1), h)
        out = self.output(out)
        outputs.append(out)
        outputs = torch.cat(outputs, dim=1)
        return outputs

    def to(self, device):
        """
        Moves all model parameters and buffers to the specified device, including the dummy token if it's trainable.

        @param device: The device (e.g., "cpu", "cuda") to move the tensors to.
        """
        super(ManyToManyRecurrentBase, self).to(device)
        self.dummy_token = self.dummy_token.to(device)


class SimpleForcaster(torch.nn.Module):
    """
    @brief A simple forecasting model for time series data.

    Configurable for either many-to-one or many-to-many predictions based on recurrent neural network architectures.
    """

    def __init__(self,
                 mode='many_to_one',
                 input_size=3,
                 hidden_size=256,
                 output_size=1,
                 trainable_dummy_token=False,
                 target_offset=1,
                 device=None):
        """
        Initializes the SimpleForcaster with specified configuration settings.

        @param mode: Forecasting mode, either 'many_to_one' or 'many_to_many'.
        @param input_size: The number of input features per timestep.
        @param hidden_size: The size of the hidden layer.
        @param output_size: The size of the output layer.
        @param trainable_dummy_token: A boolean flag that if true, uses a trainable dummy token for many-to-many predictions.
        @param target_offset: The number of timesteps ahead to predict; relevant in many-to-one mode.
        @param device: The device on which to run the model. If None, it selects CUDA if available, otherwise CPU.
        """
        super(SimpleForcaster, self).__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if mode not in ['many_to_one', 'many_to_many']:
            raise ValueError('mode must be one of "many_to_one" or "many_to_many"')
        self.mode = mode
        self.target_offset = target_offset
        if mode == 'many_to_one':
            self.model = ManyToOneRecurrentBase(input_size=input_size,
                                                hidden_size=hidden_size,
                                                output_size=output_size)
        elif mode == 'many_to_many':
            self.model = ManyToManyRecurrentBase(input_size=input_size,
                                                 hidden_size=hidden_size,
                                                 output_size=output_size,
                                                 trainable_dummy_token=trainable_dummy_token)
        self.to(self.device)

    def forward(self, *args, **kwargs):
        """
        Delegates the forward pass to the underlying model.

        @return: The output from the underlying model.
        """
        return self.model(*args, **kwargs)

    def to(self, device):
        """
        Moves all model parameters and buffers to the specified device.

        @param device: The device (e.g., "cpu", "cuda") to move the tensors to.
        """
        super(SimpleForcaster, self).to(device)
        self.model.to(device)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """
        Returns an iterator over the model parameters.

        @param recurse: Whether to recursively return parameters.
        @return: An iterator over the parameters of the model.
        """
        return self.model.parameters()

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, device=None):
        """
        @brief Loads the model from a checkpoint file.

        This class method creates an instance of SimpleForcaster and initializes it with the state from the checkpoint.

        @param checkpoint_path: Path to the checkpoint file.
        @param device: Device on which the model should be loaded. If None, it uses the default device setting.
        @return: An instance of SimpleForcaster with its state loaded from the specified checkpoint.
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        mode = checkpoint['mode']
        input_size = checkpoint['input_size']
        hidden_size = checkpoint['hidden_size']
        output_size = checkpoint['output_size']
        target_offset = checkpoint['target_offset']
        trainable_dummy_token = checkpoint.get('trainable_dummy_token', False)

        model = cls(mode=mode,
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=output_size,
                    trainable_dummy_token=trainable_dummy_token,
                    target_offset=target_offset,
                    device=device)
        model.model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def save_to_checkpoint(self, epoch=0, loss=0.0, metrics=None, model_save_path=None):
        """
        @brief Saves the model's current state to a checkpoint file.

        This method serializes the current state of the model along with additional training metadata.

        @param epoch: Current epoch number of training.
        @param loss: The loss value at the time of saving.
        @param metrics: Additional metrics to save with the checkpoint.
        @param model_save_path: Path where the checkpoint file will be saved.
        @return: None, but saves the state to a file.
        """
        checkpoint = {
            'epoch': epoch,
            'loss': loss,
            'mode': self.mode,
            'target_offset': self.target_offset,
            'input_size': self.model.input_size,
            'hidden_size': self.model.hidden_size,
            'output_size': self.model.output_size,
            'model_state_dict': self.model.state_dict(),
        }
        if hasattr(self.model, 'trainable_dummy_token'):
            checkpoint['trainable_dummy_token'] = True

        if metrics:
            for k, v in metrics.items():
                checkpoint[k] = v.item() if hasattr(v, 'item') else v

        torch.save(checkpoint, model_save_path)


def test_many_to_one():
    INPUT_SIZE = 3
    HIDDEN_SIZE = 256
    OUTPUT_SIZE = 2
    BATCH_SIZE = 32
    SEQUENCE_LENGTH = 10
    model = ManyToOneRecurrentBase(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE)
    dummy_input = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE)
    output = model(dummy_input)
    assert output.shape == (BATCH_SIZE, 1, OUTPUT_SIZE)


def test_many_to_many():
    INPUT_SIZE = 3
    HIDDEN_SIZE = 256
    OUTPUT_SIZE = 5
    BATCH_SIZE = 32
    SEQUENCE_LENGTH = 10
    LOOKAHEAD = 2
    model = ManyToManyRecurrentBase(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE,
                                    trainable_dummy_token=True)
    dummy_input = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE)
    output = model(dummy_input, future_steps=LOOKAHEAD)
    assert output.shape == (BATCH_SIZE, LOOKAHEAD, OUTPUT_SIZE)


def test_simple_forcaster():
    INPUT_SIZE = 3
    HIDDEN_SIZE = 256
    OUTPUT_SIZE = 5
    BATCH_SIZE = 32
    SEQUENCE_LENGTH = 10
    LOOKAHEAD = 2
    model = SimpleForcaster(mode='many_to_many', input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE,
                            output_size=OUTPUT_SIZE, device='cuda')
    dummy_input = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE, device='cuda')
    output = model(dummy_input, future_steps=LOOKAHEAD)
    assert output.shape == (BATCH_SIZE, LOOKAHEAD, OUTPUT_SIZE)


def test_single_batch_timetest():
    import time
    INPUT_SIZE = 25
    HIDDEN_SIZE = 256
    OUTPUT_SIZE = 10
    BATCH_SIZE = 1
    SEQUENCE_LENGTH = 64
    LOOKAHEAD = 32
    model = SimpleForcaster(mode='many_to_many', input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE,
                            output_size=OUTPUT_SIZE, device='cuda')
    start = time.time()
    for i in range(1000):
        dummy_input = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE, device='cuda')
        _ = model(dummy_input, future_steps=LOOKAHEAD)
    end = time.time()
    print("\nAverage time per result: ", (end - start) / 25)
    start_anchor = time.time()
    time.sleep(1)
    end_anchor = time.time()
    print("\n1 Second reference time anchor: ", (end_anchor - start_anchor))

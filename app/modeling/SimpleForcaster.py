from typing import Iterator

import torch
from torch.nn import Parameter


class RecurrentBase(torch.nn.Module):
    """
        Base class, forward method is meant to me inherited and overloaded.
    """

    def __init__(self, input_size=3, hidden_size=256, output_size=1):
        super(RecurrentBase, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Recurrent component - Takes care of timebased features #
        self.recurrent = torch.nn.GRU(input_size=input_size,
                                      hidden_size=hidden_size,
                                      num_layers=1,
                                      bias=True,
                                      batch_first=True,
                                      bidirectional=False)

        # Output layer - Downscale features to the requested output size #
        self.output = torch.nn.Linear(in_features=hidden_size,
                                      out_features=output_size,
                                      bias=True)

    def forward(self, *args, **kwargs):
        pass


class ManyToOneRecurrentBase(RecurrentBase):
    """
        Base class able to predict 1 timestep ahead in the future.
    """

    def __init__(self, input_size=3, hidden_size=256, output_size=1):
        super(ManyToOneRecurrentBase, self).__init__(input_size=input_size,
                                                     hidden_size=hidden_size,
                                                     output_size=output_size)

    def forward(self, x, init_state=None):
        out, _ = self.recurrent(x, init_state)
        return self.output(out[:, -1, :])

    def to(self, device):
        super(ManyToOneRecurrentBase, self).to(device)


class ManyToManyRecurrentBase(RecurrentBase):
    """
        Base class able to predict multiple timesteps ahead in the future.
    """

    def __init__(self, input_size=3, hidden_size=256, output_size=1, trainable_dummy_token=False):
        super(ManyToManyRecurrentBase, self).__init__(input_size=input_size,
                                                      hidden_size=hidden_size,
                                                      output_size=output_size)
        if trainable_dummy_token:
            self.dummy_token = torch.nn.Parameter(torch.randn(1, 1, self.input_size))
        else:
            self.dummy_token = torch.zeros(1, 1, self.input_size)

    def forward(self, x, init_state=None, future_steps=2):
        batch_size = x.size(0)

        # Initialize hidden state - batch first - 1 stands for num_layers
        if init_state is None:
            h0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        else:
            h0 = init_state

        # Initialize outputs list
        outputs = []

        # Pass the initial sequence through the GRU and keep the last output
        # This represents 1 step into the future #
        out, h = self.recurrent(x, h0)
        out = self.output(out[:, -1:, :])
        outputs.append(out)

        # Teacher-Forced Autoregressive loop for future steps #
        # Enable this code only for training purposes if teacher forcing is available #
        # for _ in range(future_steps - 1):
        #     # Use the output as the next input #
        #     out, h = self.recurrent(teacher_force_input, h)
        #     out = self.output(out[:, -1:, :])
        #     outputs.append(out)

        # Non-Teacher-Forced Autoregressive loop for future steps #
        out, h = self.recurrent(self.dummy_token.repeat(batch_size, future_steps - 1, 1), h)
        out = self.output(out)
        outputs.append(out)

        # Concatenate the outputs along the time dimension
        outputs = torch.cat(outputs, dim=1)

        return outputs

    def to(self, device):
        super(ManyToManyRecurrentBase, self).to(device)
        self.dummy_token = self.dummy_token.to(device)


class SimpleForcaster(torch.nn.Module):

    def __init__(self, mode='many_to_one',
                 input_size=3,
                 hidden_size=256,
                 output_size=1,
                 trainable_dummy_token=False,
                 device=None):
        super(SimpleForcaster, self).__init__()

        self.device = device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if mode not in ['many_to_one', 'many_to_many']:
            raise ValueError('mode must be one of "many_to_one" or "many_to_many"')
        self.mode = mode
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
        return self.model(*args, **kwargs)

    def to(self, device):
        self.model.to(device)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self.model.parameters()

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, device=None):
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Extract model arguments from the checkpoint
        mode = checkpoint['mode']
        input_size = checkpoint['input_size']
        hidden_size = checkpoint['hidden_size']
        output_size = checkpoint['output_size']
        trainable_dummy_token = checkpoint.get('trainable_dummy_token', False)

        # Create the model instance
        model = cls(mode=mode,
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=output_size,
                    trainable_dummy_token=trainable_dummy_token,
                    device=device)

        # Load the state dictionary
        model.model.load_state_dict(checkpoint['model_state_dict'])

        return model


def test_many_to_one():
    INPUT_SIZE = 3
    HIDDEN_SIZE = 256
    OUTPUT_SIZE = 2
    BATCH_SIZE = 32
    SEQUENCE_LENGTH = 10
    model = ManyToOneRecurrentBase(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE)
    dummy_input = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE)
    output = model(dummy_input)
    assert output.shape == (BATCH_SIZE, OUTPUT_SIZE)


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

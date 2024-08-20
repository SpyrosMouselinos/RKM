import torch


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
                                      dropout=0.1,
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

        # Autoregressive loop for future steps #
        # Here we need a dummy input since we dont have any future data #
        for _ in range(future_steps - 1):
            # Use the output as the next input #
            out, h = self.recurrent(self.dummy_token.repeat(batch_size, 1, 1), h)
            out = self.output(out[:, -1:, :])
            outputs.append(out)

            print(out.size())
        # Concatenate the outputs along the time dimension
        outputs = torch.cat(outputs, dim=1)

        return outputs


class SimpleForcaster(torch.nn.Module):

    def __init__(self, mode='many_to_one',
                 input_size=3,
                 hidden_size=256,
                 output_size=1,
                 trainable_dummy_token=False, device=None):
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

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def set_device(self, device):
        if device == self.device:
            return
        else:
            try:
                pass
            except Exception as e:
                print(e)


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
                            output_size=OUTPUT_SIZE)
    dummy_input = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE)
    output = model(dummy_input, future_steps=LOOKAHEAD)
    assert output.shape == (BATCH_SIZE, LOOKAHEAD, OUTPUT_SIZE)

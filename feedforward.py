from namedtensor import ntorch

"""
A feed-forward neural network.
"""
class FeedForward(ntorch.nn.Module):
    """
    This ``Module`` is a feed-forward neural network, just a sequence of ``Linear`` layers with
    activation functions in between.

    Parameters
    ----------
    input_dim : ``int``
        The dimensionality of the input.  We assume the input has shape ``(batch_size, input_dim)``.
    num_layers : ``int``
        The number of ``Linear`` layers to apply to the input.
    hidden_dims : ``Union[int, Sequence[int]]``
        The output dimension of each of the ``Linear`` layers.  If this is a single ``int``, we use
        it for all ``Linear`` layers.  If it is a ``Sequence[int]``, ``len(hidden_dims)`` must be
        ``num_layers``.
    activations : ``Union[Callable, Sequence[Callable]]``
        The activation function to use after each ``Linear`` layer.  If this is a single function,
        we use it after all ``Linear`` layers.  If it is a ``Sequence[Callable]``,
        ``len(activations)`` must be ``num_layers``.
    dropout : ``Union[float, Sequence[float]]``, optional
        If given, we will apply this amount of dropout after each layer.  Semantics of ``float``
        versus ``Sequence[float]`` is the same as with other parameters.
    """
    def __init__(self, input_dim, num_layers, hidden_dims, activations,
                 dropout = 0.0, out_name = "out"):
        super(FeedForward, self).__init__()
        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims] * num_layers  # type: ignore
        if not isinstance(activations, list):
            activations = [activations] * num_layers  # type: ignore
        if not isinstance(dropout, list):
            dropout = [dropout] * num_layers  # type: ignore
        if len(hidden_dims) != num_layers:
            raise ConfigurationError("len(hidden_dims) (%d) != num_layers (%d)" %
                                     (len(hidden_dims), num_layers))
        if len(activations) != num_layers:
            raise ConfigurationError("len(activations) (%d) != num_layers (%d)" %
                                     (len(activations), num_layers))
        if len(dropout) != num_layers:
            raise ConfigurationError("len(dropout) (%d) != num_layers (%d)" %
                                     (len(dropout), num_layers))
        self._activations = activations
        input_dims = [input_dim] + hidden_dims[:-1]
        linear_layers = []
        for layer_input_dim, layer_output_dim in zip(input_dims, hidden_dims):
            linear_layers.append(ntorch.nn.Linear(layer_input_dim, layer_output_dim).spec(out_name))
        self._linear_layers = ntorch.nn.ModuleList(linear_layers)
        dropout_layers = [ntorch.nn.Dropout(p=value) for value in dropout]
        self._dropout = ntorch.nn.ModuleList(dropout_layers)
        self._output_dim = hidden_dims[-1]
        self.input_dim = input_dim

    def get_output_dim(self):
        return self._output_dim

    def get_input_dim(self):
        return self.input_dim

    def forward(self, inputs):
        # pylint: disable=arguments-differ
        output = inputs
        for layer, activation, dropout in zip(self._linear_layers, self._activations, self._dropout):
            output = dropout(activation(layer(output)))
        return output
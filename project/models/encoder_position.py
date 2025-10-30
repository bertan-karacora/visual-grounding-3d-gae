import logging

import torch

from project.models.mlp import MLP


_LOGGER = logging.getLogger(__name__)


class EncoderPositionMLP(torch.nn.Module):
    def __init__(
        self,
        num_channels_in,
        nums_channels_hidden,
        num_channels_out,
        prob_dropout=None,
    ):
        super().__init__()

        self.mlp = None
        self.num_channels_in = num_channels_in
        self.num_channels_out = num_channels_out
        self.nums_channels_hidden = nums_channels_hidden
        self.prob_dropout = prob_dropout

        self._init()

    def _init(self):
        self.mlp = MLP(
            num_channels_in=self.num_channels_in,
            nums_channels_hidden=self.nums_channels_hidden,
            num_channels_out=self.num_channels_out,
            name_layer_norm="LayerNorm",
            prob_dropout=self.prob_dropout,
        )

    def forward(self, input):
        output = self.mlp(input)
        output = torch.nn.functional.layer_norm(output, (self.num_channels_out,))
        return output

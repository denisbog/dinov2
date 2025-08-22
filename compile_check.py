import logging
import torch
from torch import nn

from collections import OrderedDict
from collections import namedtuple

from dataclasses import dataclass

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ModelOutput(OrderedDict):
    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)


@dataclass
class BackboneOutput(ModelOutput):
    hidden_states: torch.FloatTensor


# BackboneOutput = namedtuple("BackboneOutput", ["hidden_states"])


class Dinov2Embeddings(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.ModuleList([nn.LayerNorm(10) for _ in range(12)])

    def forward(self, pixel_values: torch.FloatTensor):
        for layer in self.layer:
            pixel_values = layer(pixel_values)

        return BackboneOutput(hidden_states=pixel_values)


class Dinov2Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = Dinov2Embeddings()

    def forward(self, pixel_values: torch.Tensor):
        embedding_output = self.embeddings(pixel_values)
        embedding_output = embedding_output.hidden_states
        return embedding_output


def check():
    torch.manual_seed(13)
    model = Dinov2Backbone()
    model.eval()

    model = torch.compile(model)

    input = torch.randn(1, 10)
    logging.info(f"input: {input}")
    with torch.no_grad():
        output = model(input)
    logging.info(f"output: {output}")


if __name__ == "__main__":
    check()

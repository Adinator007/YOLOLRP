"""Class for layer-wise relevance propagation.

Layer-wise relevance propagation for VGG-like networks from PyTorch's Model Zoo.
Implementation can be adapted to work with other architectures as well by adding the corresponding operations.

    Typical usage example:

        model = torchvision.models.vgg16(pretrained=True)
        lrp_model = LRPModel(model)
        r = lrp_model.forward(x)

"""
import torch
from torch import nn
from copy import deepcopy

from torch.nn import ModuleList

from YoloLRP.lrp_layers import *
from model import ResidualBlock


def layers_lookup() -> dict:
    """Lookup table to map network layer to associated LRP operation.

    Returns:
        Dictionary holding class mappings.
    """
    lookup_table = {
        torch.nn.modules.Upsample: RelevancePropagationUpsample,
        torch.nn.modules.linear.Linear: RelevancePropagationLinear,
        torch.nn.modules.conv.Conv2d: RelevancePropagationConv2d,
        torch.nn.modules.activation.ReLU: RelevancePropagationNonLinearity,
        torch.nn.modules.activation.LeakyReLU: RelevancePropagationNonLinearity,
        torch.nn.BatchNorm2d: RelevancePropagationBatchNorm,
        torch.nn.modules.dropout.Dropout: RelevancePropagationDropout,
        torch.nn.modules.flatten.Flatten: RelevancePropagationFlatten,
        torch.nn.modules.pooling.AvgPool2d: RelevancePropagationAvgPool2d,
        torch.nn.modules.pooling.MaxPool2d: RelevancePropagationMaxPool2d,
        torch.nn.modules.pooling.AdaptiveAvgPool2d: RelevancePropagationAdaptiveAvgPool2d
    }
    return lookup_table



class LRPModel(nn.Module):
    """Class wraps PyTorch model to perform layer-wise relevance propagation."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self.model.eval()  # self.model.train() activates dropout / batch normalization etc.!

        # Parse network
        self.layers = self._get_layer_operations()

        # Create LRP network
        self.lrp_layers = self._create_lrp_model()

    def _create_lrp_model(self) -> torch.nn.ModuleList:
        """Method builds the model for layer-wise relevance propagation.

        Returns:
            LRP-model as module list.

        """
        torch.tensor([1, 2, 3]).backward()
        # Clone layers from original model. This is necessary as we might modify the weights.
        layers = deepcopy(self.layers)
        lookup_table = layers_lookup()

        # Run backwards through layers
        # becsapos, mert i az elejerol, layer a vegerol jon
        # Az egesznek az az eredmenye, hogy lesz egy olyan modulunk amin 0... ilyen iranyban mehetunk de ez az igazi modellen visszafele haladt volna
        for i, layer in enumerate(layers[::-1]):
            l = self.unpack_layer(layer)

        return layers

    def unpack_layer(self, layer: nn.Module) -> nn.Module:
        if len([f for f in layer.named_modules()]) == 1:
            return self.processLayer(layer)
        else:
            # unpack
            print(layer.__class__)
            result = ModuleList()
            for name, module in [f for f in layer.named_modules()][1:]: # minden kiveve az elso elem, mert az maga a blokk. Vegtelen ciklus lenne
                result.append(self.unpack_layer(module))
            return result

    def processLayer(self, layer):
        # print([f for f in layer.named_modules()][0][1].__class__)
        lookup_table = layers_lookup()
        try:  # itt van megforditva, visszaterunk layers-vel, de ugyanez a nev van hasznalva maguknak a layer eknek a megvalositasaban is

            result = lookup_table[layer.__class__](layer=layer)
        except KeyError:
            message = f"Layer-wise relevance propagation not implemented for " \
                      f"{layer.__class__.__name__} layer."
            raise NotImplementedError(message)

    def _get_layer_operations(self) -> torch.nn.ModuleList:
        """Get all network operations and store them in a list.

        This method is adapted to VGG networks from PyTorch's Model Zoo.
        Modify this method to work also for other networks.

        Returns:
            Layers of original model stored in module list.

        """
        # activation-ok legyenek
        layers = torch.nn.ModuleList()

        # Parse VGG-16
        for _, module in self.model.named_modules():
            layers.append(module)

        return layers

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Forward method that first performs standard inference followed by layer-wise relevance propagation.

        Args:
            x: Input tensor representing an image / images (N, C, H, W).

        Returns:
            Tensor holding relevance scores with dimensions (N, 1, H, W).

        """
        activations = list()

        # Run inference and collect activations.
        with torch.no_grad():
            # Replace image with ones avoids using image information for relevance computation.
            activations.append(torch.ones_like(x)) # nem full 1-eseket futtatunk vegig a normal halon self.layers
            # ez csak az elso reteg elotti aktivacio, ami logikus, mert amikor visszafele propagalunk, akkor ez igazabol azt adja meg, hogy az egyes pixelek mennyire
            # fontosak. Es az elso retegben mindenki egyenlo fontossagu, talan ki lehetne venni..
            # rekurzivva kell tenni
            # hasznalni kell a stack-et
            activations = self.get_activations()

        '''
        activations:
        elso tenzor: (1, 1, 1, 1, ...)
        2. tenzor:   ()
        3. 
        4. 
        ...
        utolso: (feature embedding) -> softmax -> (0.1, 0.2, 0.3, 0. 4)
        '''

        # Reverse order of activations to run backwards through model
        activations = activations[::-1]
        activations = [a.data.requires_grad_(True) for a in activations] # a data reszet minden aktivacionak modosithatova tesszuk. A belsoknek miert??

        # Initial relevance scores are the network's output activations
        relevance = torch.softmax(activations.pop(0), dim=-1)  # Unsupervised

        # Perform relevance propagation
        # activations -> az utolso eleme a csupa 1-es
        # az activations nek a merete 1-el nagyobb mint az lrp layers nek a merete, mert az elso csupa 1-es tenzort beleszamitva 1-el tobb activation van mint layer
        for i, layer in enumerate(self.lrp_layers): # az lrp layers nek az elejen van a classification valamiert, meg van forditva
            relevance = layer.forward(activations.pop(0), relevance) # a relevance propagal hatrafele es mindig az aktualis aktivaciot kuldjuk be

        return relevance.permute(0, 2, 3, 1).sum(dim=-1).squeeze().detach().cpu()

    def get_activation(self, layer: nn.Module, x: torch.Tensor):
        activations = []

        if len([f for f in layer.named_modules()]) > 1: # vannak submodule-ok, azoknak is kellenek az aktivacioi
            # unpack
            for name, sublayer in [f for f in layer.named_modules()][1:]:
                activations.append(self.get_activation(sublayer, x))

        x = layer.forward(x)
        activations.append(x)

        return activations
"""Layers for layer-wise relevance propagation.

Layers for layer-wise relevance propagation can be modified.

"""
import torch
from torch import nn
from torch.autograd import Variable

from YoloLRP.filter import relevance_filter
from model import CNNBlock

top_k_percent = 0.04  # Proportion of relevance scores that are allowed to pass.


class RelevancePropagationAdaptiveAvgPool2d(nn.Module):
    """Layer-wise relevance propagation for 2D adaptive average pooling.

    Attributes:
        layer: 2D adaptive average pooling layer.
        eps: a value added to the denominator for numerical stability.

    """

    def __init__(self, layer: torch.nn.AdaptiveAvgPool2d, eps: float = 1.0e-05) -> None:
        super().__init__()
        self.layer = layer
        self.eps = eps

    def forward(self, x):
        return self.layer(x)

    @staticmethod
    def backward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        z = self.layer.forward(a) + self.eps
        s = (r / z).data
        (z * s).sum().backward()
        c = a.grad
        r = (a * c).data
        return r


class RelevancePropagationAvgPool2d(nn.Module):
    """Layer-wise relevance propagation for 2D average pooling.

    Attributes:
        layer: 2D average pooling layer.
        eps: a value added to the denominator for numerical stability.

    """

    def __init__(self, layer: torch.nn.AvgPool2d, eps: float = 1.0e-05) -> None:
        super().__init__()
        self.layer = layer
        self.eps = eps

    def forward(self, x):
        self.a = self.layer(x) # always saving last activation
        return self.a

    @staticmethod
    def backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None, relevance=None):
        assert relevance is not None
        r = relevance
        a = self.a # az elozo forward call activation-je
        z = self.layer.forward(a) + self.eps
        s = (r / z).data
        (z * s).sum().backward()
        c = a.grad
        r = (a * c).data
        relevance = r
        super().backward(gradient, retain_graph, create_graph, inputs, relevance)



class RelevancePropagationCNNBlock(nn.Module):

    def __init__(self, layer: CNNBlock, eps: float = 1.0e-05) -> None:
        super().__init__()

        self.layer = layer
        self.convLrp = RelevancePropagationConv2d(layer.conv)
        self.bnLrp = RelevancePropagationBatchNorm(layer.conv)
        self.leakyLrp = RelevancePropagationNonLinearity(layer.conv)

        self.eps = eps

    def forward(self, x):
        self.a = self.layer(x)  # always saving last activation
        return self.a

    @staticmethod
    def backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None, relevance=None):
        r = relevance
        a = self.a  # az elozo forward call activation-je
        # a -> torch.Size([1, 512, 14, 14])
        # z -> torch.Size([1, 512, 7, 7])
        z = self.layer.forward(a) + self.eps
        s = (r / z).data
        # s -> torch.Size([1, 512, 7, 7])
        (z * s).sum().backward()
        c = a.grad
        r = (a * c).data
        relevance = r
        super().backward(gradient, retain_graph, create_graph, inputs, relevance)


class RelevancePropagationMaxPool2d(nn.Module):
    """Layer-wise relevance propagation for 2D max pooling.

    Optionally substitutes max pooling by average pooling layers.

    Attributes:
        layer: 2D max pooling layer.
        eps: a value added to the denominator for numerical stability.

    """

    def __init__(self, layer: torch.nn.MaxPool2d, mode: str = "avg", eps: float = 1.0e-05) -> None:
        super().__init__()

        if mode == "avg":
            self.layer = torch.nn.AvgPool2d(kernel_size=(2, 2))
        elif mode == "max":
            self.layer = layer

        self.eps = eps


    def forward(self, x):
        self.a = self.layer(x)  # always saving last activation
        return self.a

    @staticmethod
    def backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None, relevance=None):
        a = self.a

        # a -> torch.Size([1, 512, 14, 14])
        # z -> torch.Size([1, 512, 7, 7])
        z = self.layer.forward(a) + self.eps
        s = (r / z).data
        # s -> torch.Size([1, 512, 7, 7])
        (z * s).sum().backward()
        c = a.grad
        r = (a * c).data
        relevance = r


class RelevancePropagationUpsample(nn.Module):

    def __init__(self, layer: torch.nn.Upsample, eps: float = 1.0e-05) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        # a -> torch.Size([1, 512, 14, 14])
        # z -> torch.Size([1, 512, 7, 7])
        z = self.layer.forward(a) + self.eps
        s = (r / z).data
        # s -> torch.Size([1, 512, 7, 7])
        (z * s).sum().backward()
        c = a.grad
        r = (a * c).data
        return r



class RelevancePropagationConv2d(nn.Module):
    """Layer-wise relevance propagation for 2D convolution.

    Optionally modifies layer weights according to propagation rule. Here z^+-rule # top 4%

    Attributes:
        layer: 2D convolutional layer.
        eps: a value added to the denominator for numerical stability.

    """

    def __init__(self, layer: torch.nn.Conv2d, mode: str = "z_plus", eps: float = 1.0e-05) -> None:
        super().__init__()

        self.layer = layer

        if mode == "z_plus":
            self.layer.weight = torch.nn.Parameter(self.layer.weight.clamp(min=0.0))
            if self.layer.bias is not None:
                self.layer.bias = torch.nn.Parameter(torch.zeros_like(self.layer.bias))

        self.eps = eps

    # ide sem telejsen epsegben erkezik, mert elotte linear okon propagalt visszafele
    # a relevance folyton csokken az epsilon es top k miatt (szerintem)
    # activation, relevance
    # az elso korben az r az utolso layer nek az activation-ja softmax olva, hogy 1 legyen a summa
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor: # az activation a layer bemeneten levo activation
        # torch.Size([1, 512, 14, 14])
        r = relevance_filter(r, top_k_percent=top_k_percent) # kiveszi a 0.04 edet a relevance nak
        # torch.Size([1, 512, 14, 14]) nem valtozik r nek a merete
        # ANNYI APPROXIMACIO VAN ITT, hogy z nem a[a+1], erted, ez nem a kovi approximacio, de az r(elevance) pontosan van leosztva es visszaszorozva
        z = self.layer.forward(a) + self.eps # atkuldjuk ujra a normal halon az adatot
        # ezzel megkapjuk az elozo activation-t, ami ott volt. Nem kellett volna ujra szamolni, csak hozza kell adni epsilon-t
        # 1.0e-05
        # r es z nem lesznek ugyanazok? Nem mert az r az elejen egy softmax val inditott es onnan propagal vissza, azt leszamitva szerintem ugyanazok az epsilon on kivul
        s = (r / z).data

        '''
        Requires grad
        z -> True
        s -> False
        r -> False
        s/z -> True
        a -> True, vagyis ebbe az iranyba fog elpropagalni a relevance
        (r / z), azert igy tesszuk bele, mert ez attepi a gad ot True ba a z miatt es z nek az iranyaba fog elmenni a gradiens, a masikra ugy tekint mintha egy konstans lenne
        mert az a relevance
        z -> az elozo activation atkuldese ujra a retegen majd hozzaadva epsilon-t. Ennek a backward() ja azt fgoja csinalni, hogy az elozo reteg aktivacioja 
        fog modosulni
        '''
        # s = r/z             \>
        # a -> ConvLayer -> z -> r
        # r.sum().backward()
        # z * s kb = az elozo relevance val
        (z * s).sum().backward() # pontosan az r.
        # a.grad nak a shape je ugyanakkora, mint a nak. torch.Size([1, 512, 14, 14])
        c = a.grad # hogyan kell megvaltoztatni az aktivaciokat, hogy modosuljon a kimeneti relevance score
        # ha nem lenne az az eps es sor, akkor azt kapnank, hogy a bemeneti aktivacio mennyire erzekeny a kimenetre
        r = (a * c).data
        return r # kovetkezo layer relevance score ja


class RelevancePropagationBatchNorm(nn.Module):

    def __init__(self, layer: torch.nn.BatchNorm2d, mode: str = "z_plus", eps: float = 1.0e-05) -> None:
        super().__init__()

        self.layer = layer

        # ez igy szerintem azert fair, mert ez amugy is egy nn.Linear-nak felel meg
        if mode == "z_plus":
            self.layer.weight = torch.nn.Parameter(self.layer.weight.clamp(min=0.0)) # csak a pozitiv osztalyok erdekelnek
            self.layer.bias = torch.nn.Parameter(torch.zeros_like(self.layer.bias))

        self.running_mean = self.layer.running_mean
        self.running_var = self.layer.running_var

        self.eps = eps

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        # relevance score ok megszurve
        r = relevance_filter(r, top_k_percent=top_k_percent)
        # elozo aktivaciok ujboli atkuldese es eps vel megtoldasa
        z = self.layer.forward(a) + self.eps
        # ez miert disperse-eli el a relevance-t?
        s = r / z # mostani relevance / mostani activation eps vel megtolva
        # megszorozzuk a mostani layer.weight vel

        '''
        z = Variable(z, requires_grad=True)
        helper = Variable((z * s).sum(), requires_grad=True)
        # helper.requires_grad = True
        helper.backward()

        c = a.grad
        r = (a * c).data
        return r
        '''

        c = torch.mm(s, self.layer.weight)
        r = (a * c).data


        r = r * torch.rsqrt(self.running_var + self.eps)
        # r = r + self.running_mean # ezzel nem kell foglalkozni, mert a bias nem modositja a reteg erzekenyseget a kimenetre


        return r

class RelevancePropagationLinear(nn.Module):
    """Layer-wise relevance propagation for linear transformation.

    Optionally modifies layer weights according to propagation rule. Here z^+-rule

    Attributes:
        layer: linear transformation layer.
        eps: a value added to the denominator for numerical stability.

    """

    def __init__(self, layer: torch.nn.Linear, mode: str = "z_plus", eps: float = 1.0e-05) -> None:
        super().__init__()

        self.layer = layer

        if mode == "z_plus":
            self.layer.weight = torch.nn.Parameter(self.layer.weight.clamp(min=0.0)) # csak a pozitiv osztalyok erdekelnek
            self.layer.bias = torch.nn.Parameter(torch.zeros_like(self.layer.bias))

        self.eps = eps

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        # relevance score ok megszurve
        r = relevance_filter(r, top_k_percent=top_k_percent)
        # elozo aktivaciok ujboli atkuldese es eps vel megtoldasa
        z = self.layer.forward(a) + self.eps
        # ez miert disperse-eli el a relevance-t?
        s = r / z # mostani relevance / mostani activation eps vel megtolva
        # megszorozzuk a mostani layer.weight vel

        '''
        z = Variable(z, requires_grad=True)
        helper = Variable((z * s).sum(), requires_grad=True)
        # helper.requires_grad = True
        helper.backward()

        c = a.grad
        r = (a * c).data
        return r
        '''

        c = torch.mm(s, self.layer.weight)
        r = (a * c).data
        return r


class RelevancePropagationFlatten(nn.Module):
    """Layer-wise relevance propagation for flatten operation.

    Attributes:
        layer: flatten layer.

    """

    def __init__(self, layer: torch.nn.Flatten) -> None:
        super().__init__()
        self.layer = layer

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        r = r.view(size=a.shape)
        return r


class RelevancePropagationNonLinearity(nn.Module):
    """Layer-wise relevance propagation for ReLU activation.

    Passes the relevance scores without modification. Might be of use later.

    """

    def __init__(self, layer: torch.nn.ReLU) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        return r


class RelevancePropagationDropout(nn.Module):
    """Layer-wise relevance propagation for dropout layer.

    Passes the relevance scores without modification. Might be of use later.

    """

    def __init__(self, layer: torch.nn.Dropout) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        return r


class RelevancePropagationIdentity(nn.Module):
    """Identity layer for relevance propagation.

    Passes relevance scores without modifying them.

    """

    def __init__(self, layer) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        return r

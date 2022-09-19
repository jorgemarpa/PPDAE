"""
This python file contains the Autoencoder models as classes
per model. Architectures include linear, convolution, transpose
convolution, upampling, and ResNet type of NN/layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_out(l0, k, st):
    """
    return the output size after applying a convolution:
    Parameters
    ----------
    l0 : int
        initial size
    k  : int
        kernel size
    st : int
        stride size
    Returns
    -------
    int
        output size
    """
    return int((l0 - k) / st + 1)


def pool_out(l0, k, st):
    """
    return the output size after applying a convolution:
    Parameters
    ----------
    l0 : int
        initial size
    k  : int
        kernel size
    st : int
        stride size
    Returns
    -------
    int
        output size
    """
    return int((l0 - k) / st + 1)


class ConvLinTrans_AE(nn.Module):
    """
    Autoencoder class with user defined latent dimension, image size,
    and number of image channels. The encoder is constructed with
    sets of [2Dconv + Act_fn + MaxPooling] blocks, user defined,
    with a final linear layer to return the latent code.
    The decoder is build using transpose convolution and normal convolution layers.
    ...
    Attributes
    ----------
    latent_dim : int
        size of latent space
    img_width  : int
        width size of image
    img_height : int
        height size of image
    img_size   : float
        total numer of pixels in image
    in_ch      : int
        number of image channels
    enc_conv_blocks   : pytorch sequential
        encoder layers organized in a sequential module
    enc_linear : pytorch sequential
        encoder linear output layer
    dec_linear  : pytorch sequential
        decoder layers organized in a sequential module
    Methods
    -------
    encoder(self, x)
        Encoder module
    decoder(self, z)
        Decoder module
    forward(self, x)
        AE forward pass
    """

    def __init__(
        self,
        latent_dim=32,
        img_dim=28,
        dropout=0.2,
        in_ch=1,
        kernel=3,
        n_conv_blocks=5,
        phy_dim=0,
        feed_phy=True,
    ):
        """
        Parameters
        ----------
        latent_dim : int
            size of the dimensilatent space
        img_dim    : int
            image size, only one dimension, assuming square ratio.
        dropout    : float
            dropout probability
        in_ch      : int
            number of channels in input/output image
        kernel     : int
            size of the convolving kernel
        n_conv_blocks : int
            number of [conv + relu + maxpooling] blocks
        """
        super(ConvLinTrans_AE, self).__init__()
        self.latent_dim = latent_dim
        self.img_width = self.img_height = img_dim
        self.img_size = self.img_width * self.img_height
        self.in_ch = in_ch
        self.phy_dim = phy_dim
        self.feed_phy = feed_phy

        # Encoder specification
        self.enc_conv_blocks = nn.Sequential()
        h_ch = in_ch
        for i in range(n_conv_blocks):
            self.enc_conv_blocks.add_module(
                "conv2d_%i1" % (i + 1),
                nn.Conv2d(h_ch, h_ch * 2, kernel_size=kernel, bias=False),
            )
            self.enc_conv_blocks.add_module(
                "bn_%i1" % (i + 1), nn.BatchNorm2d(h_ch * 2, momentum=0.005)
            )
            self.enc_conv_blocks.add_module("relu_%i1" % (i + 1), nn.ReLU())
            self.enc_conv_blocks.add_module(
                "conv2d_%i2" % (i + 1),
                nn.Conv2d(h_ch * 2, h_ch * 2, kernel_size=kernel, bias=False),
            )
            self.enc_conv_blocks.add_module(
                "bn_%i2" % (i + 1), nn.BatchNorm2d(h_ch * 2, momentum=0.005)
            )
            self.enc_conv_blocks.add_module("relu_%i2" % (i + 1), nn.ReLU())
            self.enc_conv_blocks.add_module(
                "maxpool_%i" % (i + 1), nn.MaxPool2d(2, stride=2)
            )
            h_ch *= 2
            img_dim = conv_out(img_dim, kernel, 1)
            img_dim = conv_out(img_dim, kernel, 1)
            img_dim = pool_out(img_dim, 2, 2)

        self.enc_linear = nn.Sequential(
            nn.Linear(h_ch * img_dim ** 2 + phy_dim, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, self.latent_dim),
        )

        # Decoder specification
        self.dec_linear = nn.Sequential(
            nn.Linear(self.latent_dim + (phy_dim if feed_phy else 0), 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 16 * 4 * 4, bias=False),
            nn.BatchNorm1d(16 * 4 * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16 * 4 * 4, 16 * 8 * 8, bias=False),
            nn.BatchNorm1d(16 * 8 * 8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16 * 8 * 8, 16 * 16 * 16, bias=False),
            nn.ReLU(),
        )

        self.dec_transconv = nn.Sequential(
            nn.ConvTranspose2d(
                16, 16, 4, stride=2, bias=False, output_padding=1, padding=0
            ),
            nn.Conv2d(16, 16, 4, bias=False),
            nn.BatchNorm2d(16, momentum=0.005),
            nn.ReLU(),
            nn.Conv2d(16, 8, 4, bias=False),
            nn.BatchNorm2d(8, momentum=0.005),
            nn.ReLU(),
            nn.ConvTranspose2d(
                8, 8, 4, stride=2, bias=False, output_padding=1, padding=0
            ),
            nn.Conv2d(8, 8, 4, bias=False),
            nn.BatchNorm2d(8, momentum=0.005),
            nn.ReLU(),
            nn.Conv2d(8, 4, 4, bias=False),
            nn.BatchNorm2d(4, momentum=0.005),
            nn.ReLU(),
            nn.ConvTranspose2d(
                4, 4, 4, stride=2, bias=False, output_padding=1, padding=0
            ),
            nn.Conv2d(4, 4, 4, bias=False),
            nn.BatchNorm2d(4, momentum=0.005),
            nn.ReLU(),
            nn.Conv2d(4, 4, 4, bias=False),
            nn.BatchNorm2d(4, momentum=0.005),
            nn.ReLU(),
            nn.ConvTranspose2d(
                4, 4, 4, stride=2, bias=False, output_padding=1, padding=0
            ),
            nn.Conv2d(4, 4, 4, bias=False),
            nn.BatchNorm2d(4, momentum=0.005),
            nn.ReLU(),
            nn.Conv2d(4, in_ch, 7),
            nn.Sigmoid(),
        )

    def encode(self, x, phy=None):
        """
        Encoder side of autoencoder.
        Parameters
        ----------
        x : tensor
            input image with shape [N, C, H, W]
        Returns
        -------
            latent code
        """
        x = self.enc_conv_blocks(x)
        x = x.flatten(1)
        if self.phy_dim > 0 and phy is not None:
            x = torch.cat([x, phy], dim=1)
        x = self.enc_linear(x)
        return x

    def decode(self, z, phy=None):
        """
        Decoder side of autoencoder.
        Parameters
        ----------
        z : tensor
            latent code [N, latent_dim]
        Returns
        -------
            reconstructed image [N, C, H, W]
        """
        if self.phy_dim > 0 and self.feed_phy and phy is not None:
            z = torch.cat([z, phy], dim=1)
        z = self.dec_linear(z)
        z = z.view(-1, 16, 16, 16)
        z = self.dec_transconv(z)

        z = F.interpolate(z, size=(self.img_width, self.img_height), mode="nearest")
        return z

    def forward(self, x, phy=None):
        """
        Autoencoder forward pass.
        Parameters
        ----------
        x : tensor
            input image with shape [N, C, H, W]
        Returns
        -------
        xhat : tensor
            reconstructe image [N, C, H, W]
        z    : tensor
            latent code [N, latent_dim]
        """
        z = self.encode(x, phy=phy)
        xhat = self.decode(z, phy=phy)
        return xhat, z

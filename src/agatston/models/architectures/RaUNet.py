from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import (
    Convolution,
    ResidualUnit,
)
from monai.networks.layers.factories import Norm

__all__ = ['AttentionResidualUNet']


class ConvBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        strides: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = [
            ResidualUnit(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=None,
                adn_ordering='NDA',
                act='PRELU',
                norm=Norm.BATCH,
                dropout=dropout,
            ),
            ResidualUnit(
                spatial_dims=spatial_dims,
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=1,
                padding=None,
                adn_ordering='NDA',
                act='PRELU',
                norm=Norm.BATCH,
                dropout=dropout,
            ),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c: torch.Tensor = self.conv(x)
        return x_c


class UpConv(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        strides: int = 2,
        dropout=0.0,
    ) -> torch.Tensor:
        super().__init__()
        self.up_sample = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv = Convolution(
            spatial_dims,
            in_channels,
            out_channels,
            strides=1,
            kernel_size=kernel_size,
            act='PRELU',
            adn_ordering='NDA',
            norm=Norm.BATCH,
            dropout=dropout,
            is_transposed=False,
        )
        self.ru = ResidualUnit(
            spatial_dims,
            out_channels,
            out_channels,
            strides=1,
            kernel_size=kernel_size,
            subunits=1,
            act='PRELU',
            norm=Norm.BATCH,
            dropout=dropout,
            adn_ordering='NDA',
        )
        self.up = nn.Sequential(self.up_sample, self.conv, self.ru)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_up: torch.Tensor = self.up(x)
        return x_up


class AttentionBlock(nn.Module):
    def __init__(self, spatial_dims: int, f_int: int, f_g: int, f_l: int, dropout=0.0):
        super().__init__()
        self.W_g = nn.Sequential(
            ResidualUnit(
                spatial_dims=spatial_dims,
                in_channels=f_g,
                out_channels=f_int,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
            ),
            Norm[Norm.BATCH, spatial_dims](f_int),
        )
        self.W_x = nn.Sequential(
            ResidualUnit(
                spatial_dims=spatial_dims,
                in_channels=f_l,
                out_channels=f_int,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
            ),
            Norm[Norm.BATCH, spatial_dims](f_int),
        )

        self.psi = nn.Sequential(
            ResidualUnit(
                spatial_dims=spatial_dims,
                in_channels=f_int,
                out_channels=1,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
            ),
            Norm[Norm.BATCH, spatial_dims](1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU()

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi: torch.Tensor = self.relu(g1 + x1)
        return x * self.psi(psi)


class AttentionLayer(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        submodule: nn.Module,
        up_kernel_size: int = 3,
        strides: int = 2,
        dropout=0.0,
    ):
        super().__init__()
        self.attention = AttentionBlock(
            spatial_dims=spatial_dims, f_g=in_channels, f_l=in_channels, f_int=in_channels // 2
        )
        self.upconv = UpConv(
            spatial_dims=spatial_dims,
            in_channels=out_channels,
            out_channels=in_channels,
            strides=strides,
            kernel_size=up_kernel_size,
        )
        self.merge = ResidualUnit(
            spatial_dims=spatial_dims,
            in_channels=in_channels * 2,
            out_channels=in_channels,
            dropout=dropout,
        )
        self.submodule = submodule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from_lower = self.upconv(self.submodule(x))
        attention = self.attention(g=from_lower, x=x)
        attention_cat = self.merge(torch.cat((attention, from_lower), dim=1))
        return attention_cat


class AttentionResidualUNet(nn.Module):
    """
    Attention Unet based on
    Oktay, O., Schlemper, J., Folgoc, L. L., Lee, M. C. H., Heinrich, M. P., Misawa, K., ... & Rueckert, D. (2018).
    Attention U-Net: Learning Where to Look for the Pancreas. arXiv preprint arXiv:1804.03999.

    Args:
        spatial_dims (int): spatial dimensions of the input data.
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
        channels (Sequence[int]): sequence of channels. The length of the list is the number of layers.
        strides (Sequence[int]): sequence of strides. The length of the list is the number of layers.
        kernel_size (int): convolution kernel size.
        up_kernel_size (int): upconvolution kernel size.
        dropout (float): dropout probability.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Sequence[int] = 3,
        up_kernel_size: Sequence[int] = 3,
        num_res_units: int = 0,
        dropout: float = 0.0,
    ):
        super(AttentionResidualUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.spatial_dims = spatial_dims
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.num_res_units = num_res_units
        self.up_kernel_size = up_kernel_size

        head = ConvBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=channels[0],
            dropout=dropout,
        )

        reduce_channels = ResidualUnit(
            spatial_dims=spatial_dims,
            in_channels=channels[0],
            out_channels=out_channels,
            kernel_size=1,
            strides=1,
            padding=0,
        )
        self.up_kernel_size = up_kernel_size

        def _create_block(channels: Sequence[int], strides: Sequence[int]) -> nn.Module:
            if len(channels) > 2:
                subblock = _create_block(channels[1:], strides[1:])
                return AttentionLayer(
                    spatial_dims=spatial_dims,
                    in_channels=channels[0],
                    out_channels=channels[1],
                    submodule=nn.Sequential(
                        ConvBlock(
                            spatial_dims=spatial_dims,
                            in_channels=channels[0],
                            out_channels=channels[1],
                            strides=strides[0],
                            dropout=self.dropout,
                        ),
                        subblock,
                    ),
                    up_kernel_size=up_kernel_size,
                    strides=strides[0],
                    dropout=dropout,
                )
            else:
                return self._get_bottom_layer(channels[0], channels[1], strides[0])

        encdec = _create_block(self.channels, self.strides)
        self.model = nn.Sequential(head, encdec, reduce_channels)

    def _get_bottom_layer(self, in_channels: int, out_channels: int, strides: int) -> nn.Module:
        return AttentionLayer(
            spatial_dims=self.spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            submodule=ConvBlock(
                spatial_dims=self.spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                dropout=self.dropout,
            ),
            up_kernel_size=self.up_kernel_size,
            strides=strides,
            dropout=self.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_m: torch.Tensor = self.model(x)
        return x_m

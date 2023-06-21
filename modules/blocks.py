import torch
import torch.nn as nn

__all__ = ["ResidualBlock", "ResidualLayer", "Upsample", "Downsample"]


class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, activation=None, reduction=4, *args, **kwargs
    ) -> None:
        """Basic Residual block. It is essentially a Bottleneck residual block with GroupNorm.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            activation (torch.nn.Module(), optional): The inner activation functions of the block.
                Defaults to None.
            reduction (int, optional): The reduction factor of the bottleneck.
                Defaults to 4.
        """

        super().__init__(*args, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reduction = reduction
        self.bottleneck_channels = in_channels // reduction

        if activation is not None:
            self.bottleneck_activation = activation
            self.final_activation = activation
        else:
            self.bottleneck_activation = nn.GELU()
            self.final_activation = nn.GELU()

        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=self.bottleneck_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.gn1 = nn.GroupNorm(num_groups=1, num_channels=self.bottleneck_channels)

        self.conv2 = nn.Conv3d(
            in_channels=self.bottleneck_channels,
            out_channels=self.bottleneck_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.gn2 = nn.GroupNorm(num_groups=1, num_channels=self.bottleneck_channels)

        self.conv3 = nn.Conv3d(
            in_channels=self.bottleneck_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.gn3 = nn.GroupNorm(num_groups=1, num_channels=self.out_channels)

        if self.in_channels != self.out_channels:
            self.projection = nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        else:
            self.projection = None

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.gn1(x)

        x = self.conv2(x)
        x = self.gn2(x)
        x = self.bottleneck_activation(x)

        x = self.conv3(x)
        x = self.gn3(x)

        if self.projection is not None:
            residual = self.projection(residual)

        x += residual
        x = self.final_activation(x)

        return x


class ResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels, depth, *args, **kwargs) -> None:
        """Basic Residual layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            depth (int): Number of Residual blocks in the layer.
        """

        super().__init__(*args, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.activation = kwargs.get("activation", None)
        self.reduction = kwargs.get("reduction", 4)

        if self.in_channels != self.out_channels:
            self.projection = nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        else:
            self.projection = None

        self.residual_blocks = nn.ModuleList()

        for i in range(depth):
            self.residual_blocks.append(
                ResidualBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    activation=self.activation,
                    reduction=self.reduction,
                )
            )

    def forward(self, x):
        for residual_block in self.residual_blocks:
            x = residual_block(x)

        if self.projection is not None:
            x = self.projection(x)

        return x


class Upsample(nn.Module):
    def __init__(
        self, in_channels, out_channels, upsample_factor, *args, **kwargs
    ) -> None:
        """Upsampling block. Upsampling followed by an optional 1x1 convolution.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels. If out_channels != in_channels,
                a 1x1 convolution is applied after the upsampling.
            upsample_factor (int): Upsampling factor.
        """
        super().__init__(*args, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample_factor = upsample_factor

        if self.in_channels != self.out_channels:
            self.conv = nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        else:
            self.conv = None
        self.upsample = nn.Upsample(
            scale_factor=(upsample_factor[0], upsample_factor[1], upsample_factor[2]),
            mode="trilinear",
        )

    def forward(self, x):
        x = self.upsample(x)

        if self.conv is not None:
            x = self.conv(x)

        return x


class Downsample(nn.Module):
    def __init__(
        self, in_channels, out_channels, downsample_factor, *args, **kwargs
    ) -> None:
        """Downsampling block. PixelUnshuffle followed by a 1x1 convolution.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            downsample_factor (int): Downsampling factor.
        """
        super().__init__(*args, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample_factor = downsample_factor

        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(
                downsample_factor[0],
                downsample_factor[1],
                downsample_factor[2],
            ),
            stride=(downsample_factor[0], downsample_factor[1], downsample_factor[2]),
            padding=(0, 0, 0),
        )

    def forward(self, x):
        x = self.conv(x)

        return x

import torch
import torch.nn as nn
from .blocks import ResidualLayer, Upsample, Downsample


class Unet(nn.Module):
    def __init__(
        self,
        in_channels,
        depths=[2, 2, 2, 2, 2],
        embed_dim=48,
        channel_multipliers=[1, 2, 4, 8, 16],
        classes=1,
        final_activation=nn.Identity(),
        *args,
        **kwargs
    ) -> None:
        """Instantiates a Unet model with Magneto blocks.

        Args:
            in_channels (int): The number of input channels of the model.
            depths (list): The number of Magneto blocks in each layer of the model.
            embed_dim (int, optional): The initial embedding dimension. Defaults to 48.
            channel_multipliers (list, optional): The channel multiplier for each layer of the model.
                The layer will have embed_dim * channel_multiplier[i] filters. Defaults to [1, 2, 4, 8, 16].
            classes (int, optional): The number of output classes of the model head. Defaults to 1.
            final_activation (torch.nn.Module(), optional): The final activation function of the model.
                Defaults to nn.Identity().
        """

        super().__init__(*args, **kwargs)

        self.in_channels = in_channels
        self.depths = depths
        self.embed_dim = embed_dim
        self.classes = classes
        self.final_activation = final_activation
        self.channel_multipliers = channel_multipliers

        self.stem_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=2,
            stride=(2, 2, 2),
            padding=0,
        )

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        skip_channels = []

        for i, depth in enumerate(self.depths):
            in_channels = self.embed_dim * self.channel_multipliers[i]
            out_channels = (
                self.embed_dim * self.channel_multipliers[i + 1]
                if i < len(self.depths) - 1
                else self.embed_dim * self.channel_multipliers[i]
            )

            self.down_blocks.append(
                nn.ModuleList(
                    [
                        ResidualLayer(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            depth=depth,
                        ),
                        Downsample(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            downsample_factor=(2, 2, 2),
                        ),
                    ]
                )
            )

            skip_channels.append(in_channels)

        skip_channels.reverse()

        self.channel_multipliers.reverse()

        for i in range(len(self.depths) - 1):
            self.up_blocks.append(
                nn.ModuleList(
                    [
                        Upsample(
                            in_channels=skip_channels[i],
                            out_channels=skip_channels[i + 1],
                            upsample_factor=(2, 2, 2),
                        ),
                        nn.Conv3d(
                            in_channels=skip_channels[i + 1] * 2,
                            out_channels=skip_channels[i + 1],
                            kernel_size=1,
                            stride=1,
                            padding=0,
                        ),
                        ResidualLayer(
                            in_channels=skip_channels[i + 1],
                            out_channels=skip_channels[i + 1],
                            depth=2,
                        ),
                    ]
                )
            )

        self.head = nn.Sequential(
            Upsample(
                in_channels=skip_channels[-1],
                out_channels=skip_channels[-1] // 2,
                upsample_factor=(2, 2, 2),
            ),
            nn.Conv3d(
                in_channels=skip_channels[-1] // 2,
                out_channels=skip_channels[-1] // 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Conv3d(
                in_channels=skip_channels[-1] // 2,
                out_channels=self.classes,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

    def forward(self, x):
        x = self.stem_conv(x)

        down_outputs = []

        down_outputs.append(x)

        for i, (block, downsample) in enumerate(self.down_blocks):
            x = block(x)

            if i < len(self.down_blocks) - 1:
                down_outputs.append(x)
                x = downsample(x)

        for i, (upsample, proj, block) in enumerate(self.up_blocks):
            x = upsample(x)
            x = torch.cat([x, down_outputs[-(i + 1)]], dim=1)
            x = proj(x)
            x = block(x)

        x = self.head(x)
        x = self.final_activation(x)

        return x

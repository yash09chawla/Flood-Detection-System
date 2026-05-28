"""
ResNet50 + UNet++ encoder-decoder with Multi-Task Learning heads.

Architecture (from DeepSARFlood paper 2025):
  - Encoder : ResNet50 (timm, pretrained ImageNet), patched for 6-channel input
  - Decoder : UNet++ with dense nested skip connections + SCSE attention
  - Head 1  : Segmentation head  → 3-class softmax (non-water / flood / permanent)
  - Head 2  : Regression head    → MNDWI prediction (sigmoid, auxiliary task)

Input  : (B, 6, 512, 512)  — [S1-VV, S1-VH, DEM, Slope, JRC, HAND]
Output : flood_logits (B, 3, H, W),  mndwi_pred (B, 1, H, W)
         classes: 0 = non-water, 1 = flood, 2 = permanent water
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ---------------------------------------------------------------------------
# SCSE Attention Block
# ---------------------------------------------------------------------------

class SCSEModule(nn.Module):
    """Spatial and Channel Squeeze-Excitation attention."""

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.channel_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, max(1, in_channels // reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, in_channels // reduction), in_channels),
            nn.Sigmoid(),
        )
        self.spatial_se = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        cse = self.channel_se(x).view(b, c, 1, 1)
        sse = self.spatial_se(x)
        return x * cse + x * sse


# ---------------------------------------------------------------------------
# Basic Conv-BN-ReLU block
# ---------------------------------------------------------------------------

class Conv2dBnRelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 padding: int = 1, use_bn: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                            padding=padding, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# UNet++ Decoder Block
# ---------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    """
    One node in the UNet++ dense grid.
    Upsamples the lower-level feature, concatenates all skip connections
    from the same dense block, then applies Conv-BN-ReLU + optional SCSE.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int,
                 use_attention: bool = True):
        super().__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        conv_in = in_channels + skip_channels
        self.conv1 = Conv2dBnRelu(conv_in, out_channels)
        self.conv2 = Conv2dBnRelu(out_channels, out_channels)
        self.attention = SCSEModule(out_channels) if use_attention else nn.Identity()

        # Weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor,
                skip: torch.Tensor | None = None) -> torch.Tensor:
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention(x)
        return x


# ---------------------------------------------------------------------------
# Segmentation / Regression heads
# ---------------------------------------------------------------------------

class SegmentationHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 2,
                 kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes,
                              kernel_size=kernel_size,
                              padding=kernel_size // 2)
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class RegressionHead(nn.Module):
    """Predicts MNDWI values in [0, 1] (auxiliary MTL task)."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        self.act = nn.Sigmoid()
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


# ---------------------------------------------------------------------------
# UNet++ Decoder (full)
# ---------------------------------------------------------------------------

class UNetPlusPlusDecoder(nn.Module):
    """
    Dense nested UNet++ decoder.
    encoder_channels: list of channel counts from encoder, from shallowest to deepest.
    decoder_channels: output channels for each decoder level.
    """

    def __init__(self,
                 encoder_channels: list[int],
                 decoder_channels: list[int] = (256, 128, 64, 32, 16),
                 use_attention: bool = True):
        super().__init__()

        # encoder_channels: [stem, L1, L2, L3, L4, L5]  (6 values, index 0 = smallest)
        # We build the decoder from deepest (L5) up to stem
        # blocks[i] takes the feature from level (depth-i) up to level (depth-i-1)

        enc_ch = encoder_channels  # e.g. [64, 256, 512, 1024, 2048] for ResNet50
        dec_ch = list(decoder_channels)

        self.blocks = nn.ModuleList()
        in_ch = enc_ch[-1]          # start from the bottleneck
        for i, out_ch in enumerate(dec_ch):
            skip_ch = enc_ch[-(i + 2)] if i < len(enc_ch) - 1 else 0
            self.blocks.append(
                DecoderBlock(in_ch, skip_ch, out_ch, use_attention=use_attention)
            )
            in_ch = out_ch

        self.out_channels = dec_ch[-1]

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        features: list of encoder feature maps, index 0 = shallowest (largest spatial),
                  last index = deepest (smallest spatial / bottleneck).
        """
        x = features[-1]   # bottleneck
        skips = features[:-1][::-1]   # reverse: deepest skip first

        for i, block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = block(x, skip)
        return x


# ---------------------------------------------------------------------------
# Main Model: ResNet50 + UNet++ + MTL
# ---------------------------------------------------------------------------

class FloodSegmentationModel(nn.Module):
    """
    ResNet50 encoder (timm, pretrained, 6-channel input)
    + UNet++ decoder with SCSE attention
    + dual heads for MTL:
        - flood segmentation (2-class)
        - MNDWI regression (auxiliary)
    """

    ENCODER_CHANNELS = [64, 256, 512, 1024, 2048]  # ResNet50 feature channels

    def __init__(self,
                 in_channels: int = 6,
                 num_classes: int = 3,
                 decoder_channels: tuple = (256, 128, 64, 32, 16),
                 pretrained: bool = True,
                 use_attention: bool = True):
        super().__init__()

        # --- Encoder (ResNet50 via timm) ---
        self.encoder = timm.create_model(
            "resnet50",
            pretrained=pretrained,
            features_only=True,
            in_chans=in_channels,      # timm handles channel adaptation
            out_indices=(0, 1, 2, 3, 4),
        )

        # --- Decoder ---
        self.decoder = UNetPlusPlusDecoder(
            encoder_channels=self.ENCODER_CHANNELS,
            decoder_channels=decoder_channels,
            use_attention=use_attention,
        )

        final_ch = decoder_channels[-1]

        # --- Heads ---
        self.seg_head = SegmentationHead(final_ch, num_classes=num_classes)
        self.reg_head = RegressionHead(final_ch)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            flood_logits : (B, num_classes, H, W)  — raw logits, apply softmax for probs
                           default num_classes=3: [non-water, flood, permanent]
            mndwi_pred   : (B, 1, H, W)  — MNDWI regression in [0, 1]
        """
        features = self.encoder(x)          # list of 5 feature maps
        decoded = self.decoder(list(features))

        flood_logits = self.seg_head(decoded)
        # Upsample back to input resolution if needed
        if flood_logits.shape[-2:] != x.shape[-2:]:
            flood_logits = F.interpolate(flood_logits, size=x.shape[-2:],
                                         mode="bilinear", align_corners=False)

        mndwi_pred = self.reg_head(decoded)
        if mndwi_pred.shape[-2:] != x.shape[-2:]:
            mndwi_pred = F.interpolate(mndwi_pred, size=x.shape[-2:],
                                       mode="bilinear", align_corners=False)

        return flood_logits, mndwi_pred

    def predict_flood_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: returns flood probability map (B, H, W) in [0,1] for class 1."""
        with torch.no_grad():
            logits, _ = self.forward(x)
            prob = torch.softmax(logits, dim=1)[:, 1]  # class 1 = flood
        return prob

    def predict_class_map(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: returns argmax class map (B, H, W) with values in [0, num_classes)."""
        with torch.no_grad():
            logits, _ = self.forward(x)
            return logits.argmax(dim=1)


def build_model(in_channels: int = 6,
                num_classes: int = 3,
                pretrained: bool = True,
                decoder_channels: tuple = (256, 128, 64, 32, 16)) -> FloodSegmentationModel:
    """Factory function. Returns the ResNet50+UNet++ MTL model."""
    return FloodSegmentationModel(
        in_channels=in_channels,
        num_classes=num_classes,
        decoder_channels=decoder_channels,
        pretrained=pretrained,
        use_attention=True,
    )


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = build_model(pretrained=False)
    model.eval()
    dummy = torch.randn(2, 6, 512, 512)
    with torch.no_grad():
        logits, mndwi = model(dummy)
    print(f"Segmentation logits : {logits.shape}")   # (2, 3, 512, 512)
    print(f"MNDWI prediction    : {mndwi.shape}")    # (2, 1, 512, 512)
    assert logits.shape == (2, 3, 512, 512), \
        f"Expected (2,3,512,512) logits, got {tuple(logits.shape)}"
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total parameters    : {total_params:.1f}M")

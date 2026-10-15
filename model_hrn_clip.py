import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from clip.model import VisionTransformer

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, relu=True):
        super(BasicConv, self).__init__()
        layers = [
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_planes)
        ]
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
        
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride
        
        if stride > 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x):
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]

# ==================== ResNet ====================
class HRNCLIP_Progressive(nn.Module):
    def __init__(self, layers, output_dim, heads, num_macros, num_scripts, num_languages,
                 input_resolution=224, width=64, feature_size=512):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # CLIP Stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # Residual layers
        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

        # Level-specific convolutional branches before CLIP attention pooling.
        num_ftrs = embed_dim
        self.macro_conv = nn.Sequential(
            BasicConv(num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, num_ftrs, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.script_conv = nn.Sequential(
            BasicConv(num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, num_ftrs, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.language_conv = nn.Sequential(
            BasicConv(num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, num_ftrs, kernel_size=3, stride=1, padding=1, relu=True)
        )

        # Projection heads.
        self.macro_proj = nn.Sequential(
            nn.BatchNorm1d(output_dim),
            nn.Linear(output_dim, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, 512)
        )
        self.script_proj = nn.Sequential(
            nn.BatchNorm1d(output_dim),
            nn.Linear(output_dim, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, 512)
        )
        self.language_proj = nn.Sequential(
            nn.BatchNorm1d(output_dim),
            nn.Linear(output_dim, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, 512)
        )

        # Logit heads. Use raw logits for CE, tree loss, and confidence calibration.
        self.classifier_macro = nn.Linear(512, num_macros)
        self.classifier_script = nn.Linear(512, num_scripts)
        self.classifier_language = nn.Linear(512, num_languages)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]
        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))
        return nn.Sequential(*layers)

    def encode_features(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # [B, 2048, 7, 7]

        x_macro = self.macro_conv(x)
        x_script = self.script_conv(x)
        x_language = self.language_conv(x)

        x1 = x + x_macro
        x2 = x + x_script
        x3 = x + x_language

        x_macro_pooled = self.attnpool(x1)
        x_script_pooled = self.attnpool(x2)
        x_language_pooled = self.attnpool(x3)

        x_macro_fc = self.macro_proj(x_macro_pooled)
        x_script_fc = self.script_proj(x_script_pooled)
        x_language_fc = self.language_proj(x_language_pooled)

        macro_feat = self.relu(x_macro_fc)
        script_feat = self.relu(x_script_fc + x_macro_fc)
        language_feat = self.relu(x_language_fc + x_script_fc + x_macro_fc)

        return macro_feat, script_feat, language_feat

    def forward(self, x):
        macro_feat, script_feat, language_feat = self.encode_features(x)

        logits_macro = self.classifier_macro(macro_feat)
        logits_script = self.classifier_script(script_feat)
        logits_language = self.classifier_language(language_feat)

        feats = {
            "macro": macro_feat,
            "script": script_feat,
            "language": language_feat,
        }

        return logits_macro, logits_script, logits_language, feats
    
    
# ==================== ViT ====================
    
class HRNCLIP_ViT(nn.Module):
    def __init__(self, num_macros, num_scripts, num_languages,
                 input_resolution=224, patch_size=16,
                 width=768, layers=12, heads=12, output_dim=512,
                 feature_size=512):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        self.relu = nn.ReLU(inplace=True)

        self.visual = VisionTransformer(
            input_resolution=input_resolution,
            patch_size=patch_size,
            width=width,
            layers=layers,
            heads=heads,
            output_dim=output_dim
        )

        num_ftrs = width

        self.macro_conv = nn.Sequential(
            BasicConv(num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, num_ftrs, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.script_conv = nn.Sequential(
            BasicConv(num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, num_ftrs, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.language_conv = nn.Sequential(
            BasicConv(num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, num_ftrs, kernel_size=3, stride=1, padding=1, relu=True)
        )

        self.attnpool = AttentionPool2d(
            spacial_dim=input_resolution // patch_size,   # 14
            embed_dim=num_ftrs,                            # 768
            num_heads=heads,
            output_dim=output_dim
        )


        self.macro_proj = nn.Sequential(
            nn.BatchNorm1d(output_dim),
            nn.Linear(output_dim, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, 512)
        )
        self.script_proj = nn.Sequential(
            nn.BatchNorm1d(output_dim),
            nn.Linear(output_dim, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, 512)
        )
        self.language_proj = nn.Sequential(
            nn.BatchNorm1d(output_dim),
            nn.Linear(output_dim, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, 512)
        )

        self.classifier_macro = nn.Sequential(nn.Linear(512, num_macros), nn.Sigmoid())
        self.classifier_script = nn.Sequential(nn.Linear(512, num_scripts), nn.Sigmoid())
        self.classifier_language = nn.Sequential(nn.Linear(512, num_languages), nn.Sigmoid())
        self.classifier_language_softmax = nn.Linear(512, num_languages)


    def encode_features(self, x):
        x = self.visual.conv1(x)                          # [B, 768, 14, 14]
        x = x.reshape(x.shape[0], x.shape[1], -1)        # [B, 768, 196]
        x = x.permute(0, 2, 1)                            # [B, 196, 768]

        x = torch.cat([
            self.visual.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
            ), x
        ], dim=1)                                         # [B, 197, 768]

        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(x)

        x = x.permute(1, 0, 2)        # NLD -> LND
        x = self.visual.transformer(x)
        x = x.permute(1, 0, 2)        # LND -> NLD, [B, 197, 768]

        patch_tokens = x[:, 1:, :]                # [B, 196, 768]
        B, N, D = patch_tokens.shape
        H = W = int(N ** 0.5)                     # 14
        x_spatial = patch_tokens.permute(0, 2, 1).reshape(B, D, H, W)  # [B, 768, 14, 14]

        x_macro = self.macro_conv(x_spatial)
        x_script = self.script_conv(x_spatial)
        x_language = self.language_conv(x_spatial)

        x1 = x_spatial + x_macro
        x2 = x_spatial + x_script
        x3 = x_spatial + x_language

        x_macro_pooled = self.attnpool(x1)
        x_script_pooled = self.attnpool(x2)
        x_language_pooled = self.attnpool(x3)

        x_macro_fc = self.macro_proj(x_macro_pooled)
        x_script_fc = self.script_proj(x_script_pooled)
        x_language_fc = self.language_proj(x_language_pooled)
            
            
        macro_feat = self.relu(x_macro_fc)
        script_feat = self.relu(x_script_fc + x_macro_fc)
        language_feat = self.relu(x_language_fc + x_script_fc + x_macro_fc)
        
        return macro_feat, script_feat, language_feat
        
    def forward(self, x):
        macro_feat, script_feat, language_feat = self.encode_features(x)

        logits_macro = self.classifier_macro(macro_feat)
        logits_script = self.classifier_script(script_feat)
        logits_language = self.classifier_language(language_feat)

        feats = {
            "macro": macro_feat,
            "script": script_feat,
            "language": language_feat,
        }

        return logits_macro, logits_script, logits_language, feats
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import models, transforms
from PIL import Image
import os
import io
import numpy as np
from torchvision.transforms import Resize
from pathlib import Path
import tempfile
import cv2
import imageio
import time
import warnings

warnings.filterwarnings("ignore")

# Explanation of NST
nst_explanation = """
Neural Style Transfer (NST) combines the content of one image with the style of another, creating a new image that retains the original structure but adopts the artistic style. This is done using deep neural networks to minimize the differences in content and style between the input and generated images.

Go a head and choose the model from the dropdown menu and click Go button to start, and there are example images in the left sidebar menu
"""

# Define constants for all apps
NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
NORMALIZATION_STD = (0.229, 0.224, 0.225)

image_normalize = T.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD)
image_denormalize = T.Normalize(mean=[-m / s for m, s in zip(NORMALIZATION_MEAN, NORMALIZATION_STD)], 
                                std=[1 / s for s in NORMALIZATION_STD])

# Shared image transforms
def get_image_transforms(image_size=None, crop_size=None, center_crop_flag=False):
    transform_list = []
    if image_size:
        transform_list.append(T.Resize(image_size))
    if crop_size:
        if center_crop_flag:
            transform_list.append(T.CenterCrop(crop_size))
        else:
            transform_list.append(T.RandomCrop(crop_size))
    transform_list.append(T.ToTensor())
    transform_list.append(image_normalize)
    return T.Compose(transform_list)

def load_image_from_path(path, image_size=None, crop_size=None, center_crop_flag=False):
    transforms = get_image_transforms(image_size=image_size, crop_size=crop_size, center_crop_flag=center_crop_flag)
    image = Image.open(path).convert("RGB")
    return transforms(image).unsqueeze(0)

def convert_mp4_to_gif(mp4_path, gif_path, max_frames=150, target_fps=8):
    """Convert MP4 to GIF using opencv and imageio - no moviepy required"""
    try:
        # Open video with opencv
        cap = cv2.VideoCapture(mp4_path)
        
        if not cap.isOpened():
            st.error("Could not open video file")
            return False
        
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame skip to achieve target fps and limit frames
        frame_skip = max(1, int(original_fps / target_fps))
        max_frames_to_read = min(max_frames * frame_skip, total_frames)
        
        frames = []
        frame_count = 0
        frames_read = 0
        
        st.info(f"Converting MP4 to GIF... Reading {max_frames_to_read} frames")
        
        while cap.isOpened() and frames_read < max_frames_to_read:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Only keep every nth frame to reduce size and achieve target fps
            if frame_count % frame_skip == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize frame to reduce file size (max width 400px)
                height, width = frame_rgb.shape[:2]
                if width > 400:
                    new_width = 400
                    new_height = int(height * (new_width / width))
                    frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
                
                frames.append(frame_rgb)
                
                if len(frames) >= max_frames:
                    break
            
            frame_count += 1
            frames_read += 1
        
        cap.release()
        
        if len(frames) == 0:
            st.error("No frames extracted from video")
            return False
        
        # Save as GIF using imageio
        st.info(f"Saving GIF with {len(frames)} frames...")
        imageio.mimsave(gif_path, frames, fps=target_fps, loop=0)
        
        st.success(f"Successfully converted MP4 to GIF ({len(frames)} frames)")
        return True
        
    except Exception as e:
        st.error(f"Error converting MP4 to GIF: {str(e)}")
        return False

# ADAIN classes
class VGGEncoder(nn.Module):
    def __init__(self):
        super(VGGEncoder, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.encoder_layers = nn.Sequential()
        self.encoder_layers.add_module("0", vgg[0])
        self.encoder_layers.add_module("1", vgg[1])
        self.encoder_layers.add_module("2", vgg[2])
        self.encoder_layers.add_module("3", vgg[3])
        self.encoder_layers.add_module("4", vgg[4])
        self.encoder_layers.add_module("5", vgg[5])
        self.encoder_layers.add_module("6", vgg[6])
        self.encoder_layers.add_module("7", vgg[7])
        self.encoder_layers.add_module("8", vgg[8])
        self.encoder_layers.add_module("9", vgg[9])
        self.encoder_layers.add_module("10", vgg[10])
        self.encoder_layers.add_module("11", vgg[11])
        self.encoder_layers.add_module("12", vgg[12])
        self.encoder_layers.add_module("13", vgg[13])
        self.encoder_layers.add_module("14", vgg[14])
        self.encoder_layers.add_module("15", vgg[15])
        self.encoder_layers.add_module("16", vgg[16])
        self.encoder_layers.add_module("17", vgg[17])
        self.encoder_layers.add_module("18", vgg[18])
        self.encoder_layers.add_module("19", vgg[19])
        self.encoder_layers.add_module("20", vgg[20])
        for param in self.encoder_layers.parameters():
            param.requires_grad = False

    def forward(self, x, get_style_features=False):
        if not get_style_features:
            return self.encoder_layers(x)
        features = []
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x)
            if i in [1, 6, 11, 20]:
                features.append(x)
        return features

class AdaIN(nn.Module):
    def __init__(self):
        super(AdaIN, self).__init__()

    def forward(self, content_features, style_features):
        assert content_features.size()[:2] == style_features.size()[:2], "Input features must have same channel and batch size"
        size = content_features.size()
        content_mean, content_std = self.calc_mean_std(content_features)
        style_mean, style_std = self.calc_mean_std(style_features)
        normalized_content = (content_features - content_mean.expand(size)) / content_std.expand(size)
        return normalized_content * style_std.expand(size) + style_mean.expand(size)

    def calc_mean_std(self, x, eps=1e-5):
        size = x.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = x.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = x.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder_layers = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, (3, 3))
        )

    def forward(self, x):
        return self.decoder_layers(x)

# Dumoulin classes
NUM_STYLES = 40

class ConditionalInstanceNormalization(nn.Module):
    def __init__(self, num_styles, num_channels):
        super(ConditionalInstanceNormalization, self).__init__()
        self.normalize = nn.InstanceNorm2d(num_channels, affine=False)
        self.offset = nn.Parameter(0.01 * torch.randn(1, num_styles, num_channels))
        self.scale = nn.Parameter(1 + 0.01 * torch.randn(1, num_styles, num_channels))

    def forward(self, x, style_codes):
        b, c, h, w = x.size()
        x = self.normalize(x)
        gamma = torch.sum(self.scale * style_codes, dim=1).view(b, c, 1, 1)
        beta = torch.sum(self.offset * style_codes, dim=1).view(b, c, 1, 1)
        x = x * gamma + beta
        return x.view(b, c, h, w)

class ConvolutionWithCIN(nn.Module):
    def __init__(self, num_styles, in_channels, out_channels, stride, activation_type, kernel_size):
        super(ConvolutionWithCIN, self).__init__()
        self.padding = nn.ReflectionPad2d(kernel_size // 2)
        self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.cin = ConditionalInstanceNormalization(num_styles, out_channels)
        if activation_type == "relu":
            self.activation = nn.ReLU()
        elif activation_type == "linear":
            self.activation = lambda x: x

    def forward(self, x, style_codes):
        x = self.padding(x)
        x = self.convolution(x)
        x = self.cin(x, style_codes)
        x = self.activation(x)
        return x

class CustomResidualBlock(nn.Module):
    def __init__(self, num_styles, in_channels, out_channels):
        super(CustomResidualBlock, self).__init__()
        self.conv1 = ConvolutionWithCIN(num_styles, in_channels, out_channels, 1, "relu", 3)
        self.conv2 = ConvolutionWithCIN(num_styles, out_channels, out_channels, 1, "linear", 3)

    def forward(self, x, style_codes):
        out = self.conv1(x, style_codes)
        out = self.conv2(out, style_codes)
        return x + out

class CustomUpsampleBlock(nn.Module):
    def __init__(self, num_styles, in_channels, out_channels):
        super(CustomUpsampleBlock, self).__init__()
        self.convolution = ConvolutionWithCIN(num_styles, in_channels, out_channels, 1, "relu", 3)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, style_codes):
        x = self.upsample(x)
        x = self.convolution(x, style_codes)
        return x

class ImageStyleTransferNet(nn.Module):
    def __init__(self, num_styles=NUM_STYLES):
        super(ImageStyleTransferNet, self).__init__()
        self.conv1 = ConvolutionWithCIN(num_styles, 3, 32, 1, 'relu', 9)
        self.conv2 = ConvolutionWithCIN(num_styles, 32, 64, 2, 'relu', 3)
        self.conv3 = ConvolutionWithCIN(num_styles, 64, 128, 2, 'relu', 3)
        self.residual1 = CustomResidualBlock(num_styles, 128, 128)
        self.residual2 = CustomResidualBlock(num_styles, 128, 128)
        self.residual3 = CustomResidualBlock(num_styles, 128, 128)
        self.residual4 = CustomResidualBlock(num_styles, 128, 128)
        self.residual5 = CustomResidualBlock(num_styles, 128, 128)
        self.upsampling1 = CustomUpsampleBlock(num_styles, 128, 64)
        self.upsampling2 = CustomUpsampleBlock(num_styles, 64, 32)
        self.conv4 = ConvolutionWithCIN(num_styles, 32, 3, 1, 'linear', 9)

    def forward(self, x, style_codes):
        x = self.conv1(x, style_codes)
        x = self.conv2(x, style_codes)
        x = self.conv3(x, style_codes)
        x = self.residual1(x, style_codes)
        x = self.residual2(x, style_codes)
        x = self.residual3(x, style_codes)
        x = self.residual4(x, style_codes)
        x = self.residual5(x, style_codes)
        x = self.upsampling1(x, style_codes)
        x = self.upsampling2(x, style_codes)
        x = self.conv4(x, style_codes)
        return x

# Dumoulin V2 classes (shared with multi and video)
class StyleStatisticsExtractor(nn.Module):
    def __init__(self):
        super(StyleStatisticsExtractor, self).__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.relu_1_2 = nn.Sequential(*list(vgg.children())[:4])
        self.relu_2_2 = nn.Sequential(*list(vgg.children())[:9])
        self.relu_3_3 = nn.Sequential(*list(vgg.children())[:16])
        self.relu_4_2 = nn.Sequential(*list(vgg.children())[:23])

        for layer in [self.relu_1_2, self.relu_2_2, self.relu_3_3, self.relu_4_2]:
            for param in layer.parameters():
                param.requires_grad = False
    
    def extract_statistics(self, style_image):
        stats = {}
        feat_1_2 = self.relu_1_2(style_image)
        feat_2_2 = self.relu_2_2(style_image) 
        feat_3_3 = self.relu_3_3(style_image)
        feat_4_2 = self.relu_4_2(style_image)
        
        for layer_name, features in [
            ('relu_1_2', feat_1_2),
            ('relu_2_2', feat_2_2), 
            ('relu_3_3', feat_3_3),
            ('relu_4_2', feat_4_2)
        ]:
            mean = features.mean(dim=[2, 3], keepdim=True)
            std = features.std(dim=[2, 3], keepdim=True)
            stats[layer_name] = {'mean': mean, 'std': std}
            
        return stats

class StyleParameterGenerator(nn.Module):
    def __init__(self):
        super(StyleParameterGenerator, self).__init__()
        input_dim = (64 + 128 + 256 + 512) * 2
        self.param_generators = nn.ModuleDict({
            '32': self._make_param_generator(input_dim, 32),
            '64': self._make_param_generator(input_dim, 64),
            '128': self._make_param_generator(input_dim, 128),
            '3': self._make_param_generator(input_dim, 3),
        })
        
    def _make_param_generator(self, input_dim, output_channels):
        return nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_channels * 2)
        )
    
    def forward(self, style_stats):
        combined_stats = []
        for layer_name in ['relu_1_2', 'relu_2_2', 'relu_3_3', 'relu_4_2']:
            mean = style_stats[layer_name]['mean'].flatten(1)
            std = style_stats[layer_name]['std'].flatten(1)
            combined_stats.extend([mean, std])
        
        style_vector = torch.cat(combined_stats, dim=1)
        cin_params = {}
        for channels, generator in self.param_generators.items():
            params = generator(style_vector)
            scale_offset = params.chunk(2, dim=1)
            cin_params[channels] = {
                'scale': scale_offset[0],
                'offset': scale_offset[1]
            }
        
        return cin_params

class ConditionalInstanceNormalizationV2(nn.Module):
    def __init__(self, num_channels):
        super(ConditionalInstanceNormalizationV2, self).__init__()
        self.normalize = nn.InstanceNorm2d(num_channels, affine=False)
        self.num_channels = num_channels
        
    def forward(self, x, cin_params):
        b, c, h, w = x.size()
        x = self.normalize(x)
        channel_key = str(c)
        if channel_key in cin_params:
            scale = cin_params[channel_key]['scale'].view(b, c, 1, 1)
            offset = cin_params[channel_key]['offset'].view(b, c, 1, 1)
        else:
            scale = torch.zeros(b, c, 1, 1, device=x.device)
            offset = torch.zeros(b, c, 1, 1, device=x.device)
        
        return x * (1 + scale) + offset

class ConvolutionWithCINV2(nn.Module):
    def __init__(self, in_channels, out_channels, stride, activation_type, kernel_size):
        super(ConvolutionWithCINV2, self).__init__()
        self.padding = nn.ReflectionPad2d(kernel_size // 2)
        self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.cin = ConditionalInstanceNormalizationV2(out_channels)

        if activation_type == "relu":
            self.activation = nn.ReLU()
        elif activation_type == "linear":
            self.activation = lambda x: x

    def forward(self, x, cin_params):
        x = self.padding(x)
        x = self.convolution(x)
        x = self.cin(x, cin_params)
        x = self.activation(x)
        return x

class CustomResidualBlockV2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CustomResidualBlockV2, self).__init__()
        self.conv1 = ConvolutionWithCINV2(in_channels, out_channels, 1, "relu", 3)
        self.conv2 = ConvolutionWithCINV2(out_channels, out_channels, 1, "linear", 3)

    def forward(self, x, cin_params):
        out = self.conv1(x, cin_params)
        out = self.conv2(out, cin_params)
        return x + out

class CustomUpsampleBlockV2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CustomUpsampleBlockV2, self).__init__()
        self.convolution = ConvolutionWithCINV2(in_channels, out_channels, 1, "relu", 3)

    def forward(self, x, cin_params):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False, antialias=True)
        x = self.convolution(x, cin_params)
        return x

class ImageStyleTransferNetV2(nn.Module):
    def __init__(self):
        super(ImageStyleTransferNetV2, self).__init__()
        self.style_extractor = StyleStatisticsExtractor()
        self.param_generator = StyleParameterGenerator()
        self.conv1 = ConvolutionWithCINV2(3, 32, 1, 'relu', 9)
        self.conv2 = ConvolutionWithCINV2(32, 64, 2, 'relu', 3)
        self.conv3 = ConvolutionWithCINV2(64, 128, 2, 'relu', 3)

        self.residual1 = CustomResidualBlockV2(128, 128)
        self.residual2 = CustomResidualBlockV2(128, 128)
        self.residual3 = CustomResidualBlockV2(128, 128)
        self.residual4 = CustomResidualBlockV2(128, 128)
        self.residual5 = CustomResidualBlockV2(128, 128)

        self.upsampling1 = CustomUpsampleBlockV2(128, 64)
        self.upsampling2 = CustomUpsampleBlockV2(64, 32)
        self.conv4 = ConvolutionWithCINV2(32, 3, 1, 'linear', 9)

    def forward(self, content_image, style_image):
        style_stats = self.style_extractor.extract_statistics(style_image)
        cin_params = self.param_generator(style_stats)
        x = self.conv1(content_image, cin_params)
        x = self.conv2(x, cin_params)
        x = self.conv3(x, cin_params)
        x = self.residual1(x, cin_params)
        x = self.residual2(x, cin_params)
        x = self.residual3(x, cin_params)
        x = self.residual4(x, cin_params)
        x = self.residual5(x, cin_params)
        x = self.upsampling1(x, cin_params)
        x = self.upsampling2(x, cin_params)
        x = self.conv4(x, cin_params)
        return x

# Dumoulin V2 Multi-Style classes (extends V2)
class ImageStyleTransferNetMulti(nn.Module):
    def __init__(self):
        super(ImageStyleTransferNetMulti, self).__init__()
        self.style_extractor = StyleStatisticsExtractor()
        self.param_generator = StyleParameterGenerator()
        self.conv1 = ConvolutionWithCINV2(3, 32, 1, 'relu', 9)
        self.conv2 = ConvolutionWithCINV2(32, 64, 2, 'relu', 3)
        self.conv3 = ConvolutionWithCINV2(64, 128, 2, 'relu', 3)

        self.residual1 = CustomResidualBlockV2(128, 128)
        self.residual2 = CustomResidualBlockV2(128, 128)
        self.residual3 = CustomResidualBlockV2(128, 128)
        self.residual4 = CustomResidualBlockV2(128, 128)
        self.residual5 = CustomResidualBlockV2(128, 128)

        self.upsampling1 = CustomUpsampleBlockV2(128, 64)
        self.upsampling2 = CustomUpsampleBlockV2(64, 32)
        self.conv4 = ConvolutionWithCINV2(32, 3, 1, 'linear', 9)

    def forward(self, content_image, style_image1, style_image2):
        B, C, H, W = content_image.shape
        assert W % 2 == 0, "Image width must be even for vertical splitting into halves."
        half_W = W // 2
        overlap = 16
        if overlap >= half_W:
            overlap = half_W // 2

        content_left = content_image[:, :, :, :half_W + overlap]
        content_right = content_image[:, :, :, half_W - overlap :]

        stats1 = self.style_extractor.extract_statistics(style_image1)
        cin_params1 = self.param_generator(stats1)
        
        stats2 = self.style_extractor.extract_statistics(style_image2)
        cin_params2 = self.param_generator(stats2)
        
        def forward_pass(content_half, params):
            x = self.conv1(content_half, params)
            x = self.conv2(x, params)
            x = self.conv3(x, params)
            x = self.residual1(x, params)
            x = self.residual2(x, params)
            x = self.residual3(x, params)
            x = self.residual4(x, params)
            x = self.residual5(x, params)
            x = self.upsampling1(x, params)
            x = self.upsampling2(x, params)
            x = self.conv4(x, params)
            return x

        left_out = forward_pass(content_left, cin_params1)
        right_out = forward_pass(content_right, cin_params2)

        blend_start_global = half_W - overlap
        blend_width = 2 * overlap
        
        left_nonblend = left_out[:, :, :, :blend_start_global]
        blend_left = left_out[:, :, :, blend_start_global : blend_start_global + blend_width]
        blend_right = right_out[:, :, :, :blend_width]
        right_nonblend = right_out[:, :, :, blend_width : ]
        
        mask = torch.linspace(1.0, 0.0, blend_width, device=content_image.device)
        mask = mask.view(1, 1, 1, -1).expand(B, -1, -1, -1)
        
        blended = mask * blend_left + (1 - mask) * blend_right

        output = torch.cat([left_nonblend, blended, right_nonblend], dim=3)
        
        assert output.shape[3] == W, f"Output width mismatch: {output.shape[3]} != {W}"
        
        return output

# Johnson classes
class Conv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2, padding_mode='reflect')

    def forward(self, x):
        return self.conv2d(x)

class ResidualUnit(nn.Module):
    def __init__(self, channels):
        super(ResidualUnit, self).__init__()
        kernel_size = 3
        stride_size = 1
        self.conv1 = Conv2dLayer(channels, channels, kernel_size=kernel_size, stride=stride_size)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = Conv2dLayer(channels, channels, kernel_size=kernel_size, stride=stride_size)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        residual = x
        out = F.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        return out + residual

class UpsampleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.upsampling_factor = stride
        self.conv2d = Conv2dLayer(in_channels, out_channels, kernel_size, stride=1)

    def forward(self, x):
        if self.upsampling_factor > 1:
            x = torch.nn.functional.interpolate(x, scale_factor=self.upsampling_factor, mode='nearest')
        return self.conv2d(x)

def conv_layer_block(in_c, out_c, k, s, up=False):
    layers = []
    if up:
        layers.append(UpsampleConvBlock(in_c, out_c, k, s))
    else:
        layers.append(Conv2dLayer(in_c, out_c, k, s))
    layers.append(nn.InstanceNorm2d(out_c, affine=True))
    layers.append(nn.ReLU())
    return nn.Sequential(*layers)

class ImageTransformationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.Sequential(
            conv_layer_block(3, 32, 9, 1),
            conv_layer_block(32, 64, 3, 2),
            conv_layer_block(64, 128, 3, 2),
        )
        self.res = nn.Sequential(
            ResidualUnit(128),
            ResidualUnit(128),
            ResidualUnit(128),
            ResidualUnit(128),
            ResidualUnit(128),
        )
        self.up = nn.Sequential(
            conv_layer_block(128, 64, 3, 2, up=True),
            conv_layer_block(64, 32, 3, 2, up=True),
            Conv2dLayer(32, 3, 9, 1)
        )

    def forward(self, x):
        y = self.down(x)
        y = self.res(y)
        y = self.up(y)
        return torch.clamp(y, 0, 255)

# Video functions (for dumoulin_v2_video-st)
def apply_nst_to_video_safe(video_path, style_image_path, output_video_path, model, device, target_size=256, fps=None, progress_bar=None, time_text=None):
    """Enhanced video processing with better error handling"""
    try:
        style_image = load_image_from_path(style_image_path, image_size=256, crop_size=240, center_crop_flag=True)
        style_image = style_image.to(device)

        # Try to read the video with better error handling
        try:
            reader = imageio.get_reader(video_path)
            frames = []
            frame_count = 0
            max_frames = 300  # Limit frames to prevent memory issues
            
            for frame in reader:
                frames.append(frame)
                frame_count += 1
                if frame_count >= max_frames:
                    st.warning(f"Video truncated to {max_frames} frames to prevent memory issues")
                    break
                    
            meta = reader.get_meta_data()
            detected_fps = meta.get('fps', 10.0)  # Default to lower FPS for stability
            reader.close()
            
        except Exception as e:
            st.error(f"Error reading video file: {str(e)}")
            return False

        if fps is None:
            fps = min(detected_fps, 15.0)  # Cap FPS for stability

        num_frames = len(frames)
        if num_frames == 0:
            st.error("No frames found in video")
            return False
            
        resized_frames = [get_resized_frame(frame, target_size) for frame in frames]
        processed_frames = [preprocess_frame(resized_frame) for resized_frame in resized_frames]

        stylized_frames = []
        prev_stylized = None
        prev_resized = None

        start_time = time.time()
        avg_time_per_frame = 0

        for i, frame in enumerate(processed_frames):
            frame_start = time.time()
            frame = frame.to(device)
            
            with torch.no_grad():
                stylized = model(frame, style_image)
            
            if prev_stylized is not None and prev_resized is not None:
                try:
                    curr_resized = resized_frames[i]
                    prev_gray = cv2.cvtColor(prev_resized, cv2.COLOR_RGB2GRAY)
                    curr_gray = cv2.cvtColor(curr_resized, cv2.COLOR_RGB2GRAY)
                    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    
                    h, w = flow.shape[:2]
                    y, x = np.mgrid[0:h, 0:w]
                    flow_map = np.stack((x, y), axis=-1).astype(np.float32)
                    flow_map += flow
                    flow_map[..., 0] = 2.0 * flow_map[..., 0] / max(w - 1, 1) - 1.0
                    flow_map[..., 1] = 2.0 * flow_map[..., 1] / max(h - 1, 1) - 1.0
                    
                    flow_map = torch.from_numpy(flow_map).unsqueeze(0).to(device).float()
                    
                    prev_stylized_warped = F.grid_sample(prev_stylized, flow_map, mode='bilinear', padding_mode='border', align_corners=False)
                    
                    stylized = 0.8 * stylized + 0.2 * prev_stylized_warped
                except Exception as optical_flow_error:
                    # Skip optical flow if it fails
                    pass
            
            stylized_frames.append(stylized)
            prev_stylized = stylized.clone().detach()
            prev_resized = resized_frames[i]

            # Update progress
            frame_time = time.time() - frame_start
            avg_time_per_frame = ((avg_time_per_frame * i) + frame_time) / (i + 1)
            remaining_frames = num_frames - (i + 1)
            time_left = remaining_frames * avg_time_per_frame
            progress = (i + 1) / num_frames

            if progress_bar is not None:
                progress_bar.progress(progress)
            if time_text is not None:
                time_text.text(f"Processing frame {i+1}/{num_frames} - Estimated time left: {time_left:.1f}s")

        # Process stylized frames
        stylized_frames_processed = [postprocess_frame(frame) for frame in stylized_frames]
        stylized_frames_np = [convert_tensor_to_numpy(frame) for frame in stylized_frames_processed]

        # Create output directory
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

        frame_height, frame_width = stylized_frames_np[0].shape[:2]

        # Use more compatible codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        for frame in stylized_frames_np:
            out.write(frame)

        out.release()
        return True
        
    except Exception as e:
        st.error(f"Error during video processing: {str(e)}")
        return False
      
def get_resized_frame(frame, target_size=256):
    frame = np.array(frame)
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

    h, w = frame.shape[:2]
    if h > w:
        new_h = target_size
        new_w = int(w * (target_size / h))
    else:
        new_w = target_size
        new_h = int(h * (target_size / w))
    return cv2.resize(frame, (new_w, new_h))

def preprocess_frame(resized_frame):
    frame = T.ToTensor()(resized_frame).unsqueeze(0)
    frame = image_normalize(frame)
    return frame

def postprocess_frame(stylized_frame):
    stylized_frame = image_denormalize(stylized_frame).clamp_(0.0, 1.0)
    return stylized_frame

def convert_tensor_to_numpy(stylized_frame):
    stylized_frame = stylized_frame.squeeze(0)
    stylized_frame = stylized_frame.cpu().detach().numpy()
    stylized_frame = np.transpose(stylized_frame, (1, 2, 0))
    stylized_frame = (stylized_frame * 255).astype(np.uint8)
    stylized_frame = cv2.cvtColor(stylized_frame, cv2.COLOR_RGB2BGR)
    return stylized_frame

# ADAIN app function
def adain_app():
    st.title("ADAIN Neural Style Transfer")
    st.write("How to Stylize Using ADAIN:\n\nUpload Images: Upload your content image and style image.\n\nStart Stylization: Click \"Start Stylization\" to apply the style to your content.\n\nDownload: After processing, click the download button to save your stylized image.")
    model_path = "adain_model_final.pth"  # Replace with actual path
    content_file = st.file_uploader("Upload Content Image", type=["jpg", "png", "jpeg"], key="adain_content")
    style_file = st.file_uploader("Upload Style Image", type=["jpg", "png", "jpeg"], key="adain_style")

    if st.button("Start Stylization", key="adain_start"):
        if content_file and style_file:
            content_img = Image.open(content_file).convert("RGB")
            style_img = Image.open(style_file).convert("RGB")
            with st.spinner("Stylizing the image..."):
                stylized = stylize_adain(content_img, style_img, model_path)
            st.image(stylized, caption="Stylized Image", use_container_width=True)
            
            buf = io.BytesIO()
            stylized.save(buf, format="JPEG")
            stylized_bytes = buf.getvalue()
            
            st.download_button("Download Stylized Image", stylized_bytes, file_name="stylized.jpg", mime="image/jpeg", key="adain_download")
        else:
            st.warning("Please upload both content and style images.")

    if st.button("Back to Main Page", key="adain_back"):
        st.session_state.page = 'main'
        st.rerun()

def stylize_adain(content_img, style_img, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder = Decoder().to(device)
    decoder.load_state_dict(torch.load(model_path, map_location=device))
    decoder.eval()
    encoder = VGGEncoder().to(device)
    adain = AdaIN().to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    content_preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    style_preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        normalize
    ])
    content_tensor = content_preprocess(content_img).unsqueeze(0).to(device)
    style_tensor = style_preprocess(style_img).unsqueeze(0).to(device)
    with torch.no_grad():
        content_features = encoder(content_tensor)
        style_features = encoder(style_tensor, get_style_features=True)
        t = adain(content_features, style_features[-1])
        stylized_image = decoder(t)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    stylized_image = stylized_image * std + mean
    stylized_image = torch.clamp(stylized_image, 0, 1)
    original_size = content_img.size
    stylized_pil = transforms.ToPILImage()(stylized_image.squeeze(0).cpu())
    resized_stylized_pil = stylized_pil.resize(original_size, Image.BICUBIC)
    return resized_stylized_pil

# Dumoulin app function
def dumoulin_app():
    st.title("Dumoulin Neural Style Transfer")
    st.write("How to Stylize Using Dumoulin's Method:\n\nUpload Content Image: Upload the image you want to stylize.\n\nSelect Style Index: Choose a style index (from 0 to 39) that best matches the artwork you want from the style refrence photo.\n\nStart Stylization: Click \"Start Stylization\" to apply the selected style to your image.\n\nDownload: After processing, download your stylized image.")
    style_reference_image = "style_index.png"  # Replace with actual path
    st.image(style_reference_image, caption="Style Indexes Reference", use_container_width=True)
    model_path = "model_final.ckpt"  # Replace with actual path
    content_file = st.file_uploader("Upload Content Image", type=["jpg", "png", "jpeg"], key="dumoulin_content")
    style_index = st.number_input("Enter Style Index (0-39)", min_value=0, max_value=39, value=0, key="dumoulin_style_index")

    if st.button("Start Stylization", key="dumoulin_start"):
        if content_file:
            content_img = Image.open(content_file).convert("RGB")
            if not os.path.exists(model_path):
                st.error(f"Model file not found: {model_path}")
            else:
                with st.spinner("Stylizing the image..."):
                    stylized = stylize_dumoulin(content_img, style_index, model_path)
                st.image(stylized, caption="Stylized Image", use_container_width=True)
                
                buf = io.BytesIO()
                stylized.save(buf, format="JPEG")
                stylized_bytes = buf.getvalue()
                
                st.download_button("Download Stylized Image", stylized_bytes, file_name="stylized.jpg", mime="image/jpeg", key="dumoulin_download")
        else:
            st.warning("Please upload a content image.")

    if st.button("Back to Main Page", key="dumoulin_back"):
        st.session_state.page = 'main'
        st.rerun()

def stylize_dumoulin(content_img, style_index, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageStyleTransferNet()
    checkpoint = torch.load(model_path, map_location=device)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    transforms = get_image_transforms(image_size=256)
    content_image = transforms(content_img).unsqueeze(0).to(device)
    style_code = torch.zeros(1, NUM_STYLES, 1).to(device)
    style_code[0, style_index, 0] = 1
    with torch.no_grad():
        stylized_image = model(content_image, style_code)
    stylized_denorm = image_denormalize(stylized_image).clamp_(0.0, 1.0).squeeze(0)
    stylized_pil = T.ToPILImage()(stylized_denorm.cpu())
    return stylized_pil

# Dumoulin V2 app function
def dumoulin_v2_app():
    st.title("Dumoulin V2 Neural Style Transfer")
    st.write("How to Stylize Using Dumoulin V2 (Advanced):\n\nUpload Content and Style Images: Upload both the content image (your base photo) and style image (the artwork).\n\nStart Stylization: Click \"Start Stylization\" to transform your content into the style of the uploaded image.\n\nDownload: Download your final stylized image once it's ready.")
    model_path = "model_cin_adain_12000.ckpt"  # Replace with actual path
    content_file = st.file_uploader("Upload Content Image", type=["jpg", "png", "jpeg"], key="v2_content")
    style_file = st.file_uploader("Upload Style Image", type=["jpg", "png", "jpeg"], key="v2_style")

    if st.button("Start Stylization", key="v2_start"):
        if content_file and style_file:
            content_path = "temp_content.jpg"
            style_path = "temp_style.jpg"
            with open(content_path, "wb") as f:
                f.write(content_file.getbuffer())
            with open(style_path, "wb") as f:
                f.write(style_file.getbuffer())
            with st.spinner("Stylizing the image..."):
                stylized = stylize_v2(content_path, style_path, model_path)
            st.image(stylized, caption="Stylized Image", use_container_width=True)
            
            buf = io.BytesIO()
            stylized.save(buf, format="JPEG")
            stylized_bytes = buf.getvalue()
            
            st.download_button("Download Stylized Image", stylized_bytes, file_name="stylized.jpg", mime="image/jpeg", key="v2_download")
            os.remove(content_path)
            os.remove(style_path)
        else:
            st.warning("Please upload both content and style images.")

    if st.button("Back to Main Page", key="v2_back"):
        st.session_state.page = 'main'
        st.rerun()

def stylize_v2(content_path, style_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    model = ImageStyleTransferNetV2()
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    content_tensor = load_image_from_path(content_path, image_size=256).to(device)
    style_tensor = load_image_from_path(style_path, image_size=256).to(device)
    with torch.no_grad():
        stylized_image = model(content_tensor, style_tensor)
    stylized_denorm = image_denormalize(stylized_image).clamp_(0.0, 1.0).squeeze(0)
    stylized_pil = T.ToPILImage()(stylized_denorm.cpu())
    return stylized_pil

# Dumoulin V2 Multi-Style app function
def dumoulin_v2_multi_app():
    st.title("Dumoulin V2 Multi-Style Neural Style Transfer")
    st.write("How to Stylize Using Dumoulin V2 with Two Styles:\n\nUpload Content and Two Style Images: Upload the content image and two style images (one for the left half, one for the right half).\n\nStart Stylization: Click \"Start Stylization\" to apply the two styles to different parts of your content image.\n\nDownload: After the image is stylized, you can download it.")
    content_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"], key="multi_content")
    style1_file = st.file_uploader("Upload Style Image 1 (Left Half)", type=["jpg", "jpeg", "png"], key="multi_style1")
    style2_file = st.file_uploader("Upload Style Image 2 (Right Half)", type=["jpg", "jpeg", "png"], key="multi_style2")

    if st.button("Start Stylization", key="multi_start"):
        if content_file is not None and style1_file is not None and style2_file is not None:
            try:
                content_img = Image.open(content_file).convert("RGB")
                style1_img = Image.open(style1_file).convert("RGB")
                style2_img = Image.open(style2_file).convert("RGB")

                original_width, original_height = content_img.size

                content_image = process_image_multi(content_img)
                style_image1 = process_image_multi(style1_img)
                style_image2 = process_image_multi(style2_img)

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                checkpoint_path = 'model_cin_adain_final.ckpt'  # Adjust if needed
                if not os.path.exists(checkpoint_path):
                    st.error(f"Model checkpoint not found at: {checkpoint_path}")
                else:
                    model = ImageStyleTransferNetMulti()
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    model.load_state_dict(checkpoint['state_dict'])
                    model = model.to(device)
                    model.eval()

                    with torch.no_grad():
                        stylized_image = model(content_image.to(device), style_image1.to(device), style_image2.to(device))

                    resize_transform = Resize(size=(original_height, original_width))
                    stylized_image = resize_transform(stylized_image)

                    stylized_denorm = image_denormalize(stylized_image.cpu()).clamp_(0.0, 1.0).squeeze(0).permute(1, 2, 0).numpy()

                    st.image(stylized_denorm, caption="Stylized Image", use_container_width=True)

                    buf = io.BytesIO()
                    img = Image.fromarray((stylized_denorm * 255).astype(np.uint8))
                    img.save(buf, format="JPEG")
                    byte_im = buf.getvalue()
                    st.download_button(
                        label="Download Stylized Image",
                        data=byte_im,
                        file_name="stylized_image.jpg",
                        mime="image/jpeg",
                        key="multi_download"
                    )
            except Exception as e:
                st.error(f"Error during stylization: {str(e)}")
        else:
            st.warning("Please upload all images.")

    if st.button("Back to Main Page", key="multi_back"):
        st.session_state.page = 'main'
        st.rerun()

def process_image_multi(img, image_size=(512, 512)):
    transforms = get_image_transforms(image_size=image_size)
    return transforms(img).unsqueeze(0)

# Dumoulin V2 Video app function
def dumoulin_v2_video_app():
    st.title("Dumoulin V2 Video Neural Style Transfer")
    st.write("How to Stylize a Video Using Dumoulin V2:\n\nUpload Content Video: Upload the video file you want to stylize.\n\nUpload Style Image: Upload the style image you want to apply to your video.\n\nStart Stylization: Click \"Start Stylization\" to apply the style to each frame of the video.\n\nDownload: Once the video is processed, you can download the stylized version.")
    
    # Add warning about video processing
    st.warning("⚠️ Video processing is resource-intensive. For best results:\n- Keep videos under 30 seconds\n- MP4 files will be converted to GIF format for processing\n- Processing may take several minutes")
    
    content_video = st.file_uploader("Upload Content Video", type=["mp4", "gif"], key="video_content")
    style_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"], key="video_style")

    if st.button("Start Stylization", key="video_start"):
        if content_video is not None and style_file is not None:
            try:
                video_suffix = os.path.splitext(content_video.name)[1].lower()
                
                # Create temporary files
                with tempfile.NamedTemporaryFile(delete=False, suffix=video_suffix) as tmp_video:
                    tmp_video.write(content_video.read())
                    video_path = tmp_video.name

                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_style:
                    tmp_style.write(style_file.read())
                    style_path = tmp_style.name

                # Handle MP4 conversion to GIF
                processing_video_path = video_path
                conversion_needed = False
                
                if video_suffix == '.mp4':
                    st.info("Converting MP4 to GIF for processing...")
                    gif_path = tempfile.mktemp(suffix=".gif")
                    
                    conversion_success = convert_mp4_to_gif(video_path, gif_path)
                    if conversion_success:
                        processing_video_path = gif_path
                        conversion_needed = True
                    else:
                        st.error("Failed to convert MP4. Please try uploading a GIF instead.")
                        return

                output_path = tempfile.mktemp(suffix=".mp4")

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                st.info(f"Using device: {device}")

                checkpoint_path = 'model_cin_adain_12000.ckpt'
                if not os.path.exists(checkpoint_path):
                    st.error(f"Model checkpoint not found at: {checkpoint_path}")
                else:
                    model = ImageStyleTransferNetV2()
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    model.load_state_dict(checkpoint['state_dict'])
                    model = model.to(device)
                    model.eval()

                    progress_bar = st.progress(0)
                    time_text = st.empty()

                    success = apply_nst_to_video_safe(
                        video_path=processing_video_path,
                        style_image_path=style_path,
                        output_video_path=output_path,
                        model=model,
                        device=device,
                        target_size=1024,  # Reduced size for stability
                        fps=None,
                        progress_bar=progress_bar,
                        time_text=time_text
                    )

                    if success and os.path.exists(output_path):
                        st.success("Video stylization completed!")
                        st.video(output_path)

                        with open(output_path, "rb") as f:
                            st.download_button(
                                label="Download Stylized Video",
                                data=f,
                                file_name="stylized_video.mp4",
                                mime="video/mp4",
                                key="video_download"
                            )
                    else:
                        st.error("Video processing failed. Please try with a smaller video or different format.")

                # Cleanup temporary files
                try:
                    os.unlink(video_path)
                    os.unlink(style_path)
                    if conversion_needed and os.path.exists(gif_path):
                        os.unlink(gif_path)
                    if os.path.exists(output_path):
                        os.unlink(output_path)
                except:
                    pass  # Ignore cleanup errors

            except Exception as e:
                st.error(f"Error during stylization: {str(e)}")
                st.info("Try using a shorter video or GIF format for better compatibility.")
        else:
            st.warning("Please upload both the video and style image.")

    if st.button("Back to Main Page", key="video_back"):
        st.session_state.page = 'main'
        st.rerun()
      
# Johnson app function
def johnson_app():
    st.title("Johnson Neural Style Transfer")
    st.write("How to Stylize Using Johnson's Method (Custom Styles):\n\nUpload Content Image: Upload the content image (the photo you want to stylize).\n\nChoose a Style: Select a style from the list (e.g., Starry Night, Scream, The River Seine).\n\nStart Stylization: Click \"Start Stylization\" to apply the chosen style to your content image.\n\nDownload: After processing, download your stylized image.")
    styles = {
        "Starry Night": "johnson_style1_model_final.pth",
        "Scream": "johnson_style2_model_final.pth",
        "The River Seine at Chatou": "johnson_style3_model_final.pth"
    }
    content_file = st.file_uploader("Upload Content Image", type=["jpg", "png", "jpeg"], key="johnson_content")
    style_choice = st.selectbox("Choose Style", list(styles.keys()), key="johnson_style")

    if st.button("Start Stylization", key="johnson_start"):
        if content_file:
            content_img = Image.open(content_file).convert("RGB")
            model_path = styles[style_choice]
            if not os.path.exists(model_path):
                st.error(f"Model file not found: {model_path}")
            else:
                with st.spinner("Stylizing the image..."):
                    stylized = stylize_johnson(content_img, model_path)
                st.image(stylized, caption="Stylized Image", use_container_width=True)
                
                buf = io.BytesIO()
                stylized.save(buf, format="JPEG")
                stylized_bytes = buf.getvalue()
                
                st.download_button("Download Stylized Image", stylized_bytes, file_name="stylized.jpg", mime="image/jpeg", key="johnson_download")
        else:
            st.warning("Please upload a content image.")

    if st.button("Back to Main Page", key="johnson_back"):
        st.session_state.page = 'main'
        st.rerun()

def stylize_johnson(content_img, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageTransformationNet().to(device)
    model, _ , _ = load_model_johnson(model_path, model, map_location=device)
    model.eval()
    inference_transform = transforms.ToTensor()
    content_tensor = inference_transform(content_img).unsqueeze(0).to(device)
    orig_w, orig_h = content_img.size
    with torch.no_grad():
        input_to_net = content_tensor * 255.0
        stylized_image = model(input_to_net)
        stylized_image = F.interpolate(stylized_image, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        stylized_image = stylized_image / 255.0
    stylized_pil = transforms.ToPILImage()(stylized_image.squeeze(0).cpu())
    return stylized_pil

def load_model_johnson(checkpoint_path, model, optimizer=None, map_location=None):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    iteration = checkpoint['iteration']
    return model, epoch, iteration

# Sidebar for examples
st.sidebar.title("Example Images")
with st.sidebar.expander("View Examples"):
    st.image("content.jpg", caption="Example Image 1")
    st.image("c1.jpg", caption="Example Image 2")
    st.image("000000000008.jpg", caption="Example Image 3")
    st.image("style1.jpg", caption="Example Image 4")
    st.image("style2.jpg", caption="Example Image 5")
    st.image("wave.jpg", caption="Example Image 6")
    st.video("video2.gif")

# Main app logic
if 'page' not in st.session_state:
    st.session_state.page = 'main'

if st.session_state.page == 'main':
    st.title("Neural Style Transfer Apps")
    
    # Add an image under the title
    st.image("nst-example.png", caption="Neural Style Transfer", use_container_width=True) 
    
    st.write(nst_explanation)
    option = st.selectbox("Choose an NST Method", ["ADAIN", "Dumoulin", "Dumoulin V2", "Dumoulin V2 Multi-Style", "Dumoulin V2 Video", "Johnson"])
    if st.button("Go"):
        st.session_state.page = option
        st.rerun()
else:
    if st.session_state.page == "ADAIN":
        adain_app()
    elif st.session_state.page == "Dumoulin":
        dumoulin_app()
    elif st.session_state.page == "Dumoulin V2":
        dumoulin_v2_app()
    elif st.session_state.page == "Dumoulin V2 Multi-Style":
        dumoulin_v2_multi_app()
    elif st.session_state.page == "Dumoulin V2 Video":
        dumoulin_v2_video_app()
    elif st.session_state.page == "Johnson":
        johnson_app()

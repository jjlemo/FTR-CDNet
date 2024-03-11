import math
import warnings
from functools import partial

import numpy as np
import torch
import torch.nn.functional
import torch.nn.functional as F
from scipy.io import savemat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import nn

from models.gap import GlobalAvgPool2D


class GlobalAvgPool2DBaseline(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2DBaseline, self).__init__()

    def forward(self, x):
        x_pool = torch.mean(x.view(x.size(0), x.size(1), x.size(2) * x.size(3)), dim=2)

        x_pool = x_pool.view(x.size(0), x.size(1), 1, 1).contiguous()
        return x_pool


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(UpsampleConvLayer, self).__init__()
        self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=1)

    def forward(self, x):
        out = self.conv2d(x)
        return out


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLayer, self).__init__()
        #         reflection_padding = kernel_size // 2
        #         self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        #         out = self.reflection_pad(x)
        out = self.conv2d(x)
        return out


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out


def init_weights(m):
    """
    Initialize weights of layers using Kaiming Normal (He et al.) as argument of "Apply" function of
    "nn.Module"
    :param m: Layer to initialize
    :return: None
    """
    if isinstance(m, nn.Conv2d):
        '''
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        trunc_normal_(m.weight, std=math.sqrt(1.0/fan_in)/.87962566103423978)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        '''
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class MultiScale_ChangeRelation_V2(nn.Module):
    def __init__(self, in_channels):
        super(MultiScale_ChangeRelation_V2, self).__init__()
        self.decoder_f43 = decoder(in_channels, mid_channel=in_channels, out_channels=128, scal=False)
        self.decoder_f32 = decoder(in_channels, mid_channel=128, out_channels=128)
        self.decoder_f21 = decoder(in_channels, mid_channel=128, out_channels=64)

    def forward(self, features):
        f2 = self.decoder_f43(features[0], features[1])
        f3 = self.decoder_f32(f2, features[2])
        f4 = self.decoder_f21(f3, features[3])
        return f4


class decoder(nn.Module):
    def __init__(self, in_channels, mid_channel, out_channels, scal=True):
        super(decoder, self).__init__()
        self.scal = scal
        self.baseConv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channel, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(mid_channel, mid_channel, 1),
        )
        self.change_encoder = nn.Sequential(
            GlobalAvgPool2D(),
            nn.Conv2d(mid_channel, out_channels, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 1),
        )
        self.content_encoders = nn.Sequential(
            nn.Conv2d(mid_channel, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.feature_reencoders = nn.Sequential(
            nn.Conv2d(mid_channel, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.feature_result = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.normalizer = nn.Sigmoid()

    def forward(self, low_scale, high_scale):
        if self.scal:
            high_scale = self.baseConv(high_scale)
        # 对输入的特征图进行change_encoder
        low = self.change_encoder(low_scale)
        high = self.content_encoders(high_scale)
        re_encoder = self.feature_reencoders(high_scale)
        f1 = self.normalizer(F.relu(low * high).sum(dim=1, keepdim=True))
        result = self.feature_result(f1 * re_encoder)

        return result


class ConvLayers(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLayers, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        out = self.conv2d(x)
        return out


class finalConv(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel):
        super(finalConv, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, mid_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(),
            nn.Conv2d(mid_channel, mid_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(),
            nn.Conv2d(mid_channel, out_channel, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class FusionConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FusionConv, self).__init__()
        self.skip_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )
        self.fusion_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=3, padding=1),
            nn.BatchNorm2d(int(in_channels / 2)),
            nn.ReLU(),
            nn.Conv2d(int(in_channels / 2), out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        res = self.skip_layer(x)
        result = self.fusion_conv(x)
        result += res
        return result


# 存储结果
def save_to_mat(x1, x2, fx1, fx2, p1, p2, cp, file_name):
    # Save to mat files
    x1_np = x1.detach().cpu().numpy()
    x2_np = x2.detach().cpu().numpy()
    p1_np = p1.detach().cpu().numpy()
    p2_np = p2.detach().cpu().numpy()

    fx1_0_np = fx1[0].detach().cpu().numpy()
    fx2_0_np = fx2[0].detach().cpu().numpy()
    fx1_1_np = fx1[1].detach().cpu().numpy()
    fx2_1_np = fx2[1].detach().cpu().numpy()
    fx1_2_np = fx1[2].detach().cpu().numpy()
    fx2_2_np = fx2[2].detach().cpu().numpy()
    fx1_3_np = fx1[3].detach().cpu().numpy()
    fx2_3_np = fx2[3].detach().cpu().numpy()
    # fx1_4_np = fx1[4].detach().cpu().numpy()
    # fx2_4_np = fx2[4].detach().cpu().numpy()

    cp_np = cp[-1].detach().cpu().numpy()

    # weight = {'x1': x1_np, 'x2': x2_np, 'p1': p1_np, 'p2': p2_np,
    #           'fx1_0': fx1_0_np, 'fx1_1': fx1_1_np, 'fx1_2': fx1_2_np, 'fx1_3': fx1_3_np, 'fx1_4': fx1_4_np,
    #           'fx2_0': fx2_0_np, 'fx2_1': fx2_1_np, 'fx2_2': fx2_2_np, 'fx2_3': fx2_3_np, 'fx2_4': fx2_4_np,
    #           "final_pred": cp_np}

    weight = {'x1': x1_np, 'x2': x2_np, 'p1': p1_np, 'p2': p2_np,
              'fx1_0': fx1_0_np, 'fx1_1': fx1_1_np, 'fx1_2': fx1_2_np, 'fx1_3': fx1_3_np,
              'fx2_0': fx2_0_np, 'fx2_1': fx2_1_np, 'fx2_2': fx2_2_np, 'fx2_3': fx2_3_np,
              "final_pred": cp_np}

    savemat("./weight/MIT_CD/vis/mat/" + file_name + ".mat", weight)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# Attention模块 骨干网络
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


# 骨干网络的Block
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class FeedCov(nn.Module):
    def __init__(self, d_model):
        super(FeedCov, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=2, dilation=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, y, H, W):
        B, N, C = x.shape
        q_x = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            y_ = y.permute(0, 2, 1).reshape(B, C, H, W)
            y_ = self.sr(y_).reshape(B, C, -1).permute(0, 2, 1)
            y_ = self.norm(y_)
            kv = self.kv(y_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(y).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q_x @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        y = (attn @ v).transpose(1, 2).reshape(B, N, C)
        y = self.proj(y)
        y = self.proj_drop(y)

        return y


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, img_size=256, patch_size=3, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 norm_layer=nn.LayerNorm,
                 sr_ratio=1):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                       attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.norm2 = norm_layer(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, y, B, H, W):
        x = x.flatten(2).transpose(1, 2)
        y = y.flatten(2).transpose(1, 2)
        x = x + self.attn(self.norm1(x), self.norm1(y), self.H, self.W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, scale):
        super().__init__()

        self.scale = scale

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.scale, k.transpose(-2, -1))  # 1.Matmul

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)  # 3.Mask

        attn = F.softmax(attn, dim=-1)  # 4.Softmax
        output = torch.matmul(attn, v)  # 5.Output

        return attn, output


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, dim_in, num_heads=4, linear_dim=64, num_features=256):
        super(MultiHeadedAttention, self).__init__()
        self.num_heads = num_heads  # No of heads
        self.in_pixels = dim_in  # No of pixels in the input image
        self.linear_dim = linear_dim  # Dim of linear-layer (outputs)

        self.linear_q = nn.Linear(dim_in ** 2, num_heads * linear_dim, bias=False)
        self.linear_k = nn.Linear(dim_in ** 2, num_heads * linear_dim, bias=False)
        self.linear_v = nn.Linear(dim_in ** 2, num_heads * linear_dim, bias=False)
        self.fc = nn.Linear(num_heads * linear_dim, dim_in ** 2, bias=False)  # Final fully connected layer
        self.attention = ScaledDotProductAttention(scale=np.power(linear_dim, 0.5))
        self.OutBN = nn.BatchNorm2d(num_features=num_features)

    def forward(self, x, mask=None):
        b, c, h, w = x.shape
        n_head = self.num_heads
        linear_dim = self.linear_dim
        x = x.view(b, c, h * w)
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)

        q = q.view(b, c, n_head, linear_dim).transpose(1, 2)
        k = k.view(b, c, n_head, linear_dim).transpose(1, 2)
        v = v.view(b, c, n_head, linear_dim).transpose(1, 2)

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)

        attn, output = self.attention(q, k, v, mask=mask)
        # attn = attn.transpose(1, 2).contiguous().view(b, c, n_head * linear_dim)  # batch, n, dim_v
        output = output.transpose(1, 2).contiguous().view(b, c, n_head * linear_dim)
        output = self.fc(output)
        output = output.view(b, c, h, w)
        output = self.OutBN(output)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model):
        super(FeedForward, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=2, dilation=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class TransformerBlockV2(nn.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, dim_in, num_head=4, linear_dim=64, hidden=256):
        super().__init__()
        self.attention = MultiHeadedAttention(dim_in=dim_in, num_heads=num_head, linear_dim=linear_dim,
                                              num_features=hidden)
        self.feed_forward = FeedForward(hidden)

    def forward(self, x):
        x, b, c = x['x'], x['b'], x['c']
        x = x + self.attention(x)
        x = x + self.feed_forward(x)
        return {'x': x, 'b': b, 'c': c}


class MultiHeadedAttention_v2(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, dim_in, num_heads=4, linear_dim=64, num_features=256):
        super(MultiHeadedAttention_v2, self).__init__()
        self.num_heads = num_heads  # No of heads
        self.in_pixels = dim_in  # No of pixels in the input image
        self.linear_dim = linear_dim  # Dim of linear-layer (outputs)

        self.linear_q = nn.Linear(dim_in ** 2, num_heads * linear_dim, bias=False)
        self.linear_k = nn.Linear(dim_in ** 2, num_heads * linear_dim, bias=False)
        self.linear_v = nn.Linear(dim_in ** 2, num_heads * linear_dim, bias=False)
        self.fc = nn.Linear(num_heads * linear_dim, dim_in ** 2, bias=False)  # Final fully connected layer
        self.attention = ScaledDotProductAttention(scale=np.power(linear_dim, 0.5))
        self.OutBN = nn.BatchNorm2d(num_features=num_features)

    def forward(self, x, x1, mask=None):
        b, c, h, w = x.shape
        n_head = self.num_heads
        linear_dim = self.linear_dim
        x = x.view(b, c, h * w)
        x1 = x1.view(b, c, h * w)
        q = self.linear_q(x)
        k = self.linear_k(x1)
        v = self.linear_v(x1)
        q = q.view(b, c, n_head, linear_dim).transpose(1, 2)
        k = k.view(b, c, n_head, linear_dim).transpose(1, 2)
        v = v.view(b, c, n_head, linear_dim).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        attn, output = self.attention(q, k, v, mask=mask)
        # attn = attn.transpose(1, 2).contiguous().view(b, c, self.dim_v)  # batch, n, dim_v
        output = output.transpose(1, 2).contiguous().view(b, c, n_head * linear_dim)  # 3.Concat
        output = self.fc(output)
        output = output.view(b, c, h, w)
        output = self.OutBN(output)
        return output


class TForwardBlockV2(nn.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, dim_in, num_head=4, linear_dim=256, hidden=256):
        super().__init__()
        self.attention = MultiHeadedAttention_v2(dim_in=dim_in, num_heads=num_head, linear_dim=linear_dim,
                                                 num_features=hidden)
        self.feed_forward = FeedForward(hidden)

    def forward(self, x):
        # pdb.set_trace()
        x0, x1 = x['x'][0], x['x'][1]
        x = x0 + self.attention(x0, x1)
        x = x + self.feed_forward(x)
        return {'x': x}


class ChangeTransformerV2(nn.Module):
    def __init__(self, init_weights=True, dim_in=8, in_channels=256, linear_dim=40, stack_num=1):
        super(ChangeTransformerV2, self).__init__()
        blocks = []
        for _ in range(stack_num):
            blocks.append(TransformerBlockV2(dim_in=dim_in, num_head=4, linear_dim=linear_dim, hidden=in_channels))
        self.transformer = nn.Sequential(*blocks)
        self.tfforward = TForwardBlockV2(dim_in=dim_in, num_head=4, linear_dim=linear_dim, hidden=in_channels)

    def forward(self, x1, x2):
        # b = 1
        b, c, h, w = x1.size()
        enc_feat = self.tfforward({'x': [x1, x2], 'b': b, 'c': c})['x']
        # enc_feat = self.transformer({'x': enc_feat, 'b': b, 'c': c})['x']
        return enc_feat

    def infer(self, feat):
        t, c, _, _ = feat.size()
        enc_feat = self.transformer(
            {'x': feat, 'b': 1, 'c': c})['x']
        return enc_feat


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


# DepthWise Convolution的一个卷积核负责一个通道，一个通道只被一个卷积核卷积。
# 同样是对于一张5×5像素、三通道彩色输入图片（shape为5×5×3），DepthWise Convolution首先经过第一次卷积运算，不同于上面的常规卷积，
# DW完全是在二维平面内进行。卷积核的数量与上一层的通道数相同（通道和卷积核一一对应）。所以一个三通道的图像经过运算后生成了3个Feature map
# (如果有same padding则尺寸与输入层相同为5×5)先卷积然后下采样
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


# combine 模块
def conv_combine(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


# combine 模块
def fusion_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
    )


def make_prediction(in_channels, out_channels, norm_layer=nn.BatchNorm2d):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    )


def resize(input, size=None, scale_factor=None, mode='nearest', align_corners=None, warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class EncoderTransformer_MIT_v1(nn.Module):
    """
    Transformer Encoder
    """

    def __init__(self, img_size=256, patch_size=3, in_chans=3, num_classes=2, embed_dims=None, num_heads=[2, 2, 4, 8],
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., depths=None,
                 norm_layer=nn.LayerNorm, sr_ratios=[8, 4, 2, 1], mlp_ratios=[4, 4, 4, 4]):
        super(EncoderTransformer_MIT_v1, self).__init__()
        self.num_classes = num_classes
        self.depths = depths
        if depths is None:
            self.depths = [3, 4, 6, 3]
        self.embed_dims = embed_dims
        if embed_dims is None:
            self.embed_dims = [64, 128, 256, 512]
        self.head = None
        # assert

        # patch embedding definitions
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    # def init_weights(self, pretrained=None):
    #     if isinstance(pretrained, str):
    #         logger = get_root_logger()
    #         load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)

        return x


class EncoderTransformer_MIT_v2(nn.Module):
    """
    Transformer Encoder
    """

    def __init__(self, img_size=256, patch_size=3, in_chans=3, num_classes=2, embed_dims=None, num_heads=[2, 2, 4, 8],
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., depths=None,
                 norm_layer=nn.LayerNorm, sr_ratios=[8, 4, 2, 1], mlp_ratios=[4, 4, 4, 4]):
        super(EncoderTransformer_MIT_v2, self).__init__()
        self.num_classes = num_classes
        self.depths = depths
        if depths is None:
            self.depths = [3, 4, 6, 3]
        self.embed_dims = embed_dims
        if embed_dims is None:
            self.embed_dims = [64, 128, 256, 512]
        self.head = None
        # assert

        # patch embedding definitions
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=patch_size, stride=2,
                                              in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=patch_size, stride=2,
                                              in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=patch_size, stride=2,
                                              in_chans=embed_dims[2], embed_dim=embed_dims[3])

        # Stage-1 (x1/4 scale)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        # Stage-2 (x1/8 scale)
        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        # Stage-3 (x1/16 scale)
        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        # Stage-4 (x1/32 scale)
        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x1, H1, W1 = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x1 = blk(x1, H1, W1)
        x1 = self.norm1(x1)
        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x1)

        # stage 2
        x1, H1, W1 = self.patch_embed2(x1)
        for i, blk in enumerate(self.block2):
            x1 = blk(x1, H1, W1)
        x1 = self.norm2(x1)
        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x1)

        # stage 3
        x1, H1, W1 = self.patch_embed3(x1)
        for i, blk in enumerate(self.block3):
            x1 = blk(x1, H1, W1)
        x1 = self.norm3(x1)
        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x1)

        # stage 4
        x1, H1, W1 = self.patch_embed4(x1)
        for i, blk in enumerate(self.block4):
            x1 = blk(x1, H1, W1)
        x1 = self.norm4(x1)
        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x1)
        return outs

    def forward(self, x):
        x = self.forward_features(x)
        return x


class FTR_Decoder(nn.Module):
    def __init__(self, input_transform='multiple_select', in_index=None, align_corners=True, in_channels=None,
                 embedding_dim=256, output_nc=2, decoder_softmax=False, normal_init=True):
        super(FTR_Decoder, self).__init__()
        # assert
        self.in_index = in_index
        if in_index is None:
            self.in_index = [0, 1, 2, 3]
        self.in_channels = in_channels
        if in_channels is None:
            self.in_channels = [32, 64, 128, 256]
        # self.img_size = 256
        # settings
        self.input_transform = input_transform
        self.align_corners = align_corners
        self.embedding_dim = embedding_dim
        self.output_nc = output_nc
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        # convolutional Difference Modules
        self.combine_c4 = conv_combine(in_channels=c4_in_channels * 2, out_channels=self.embedding_dim)
        self.combine_c3 = conv_combine(in_channels=c3_in_channels * 2, out_channels=self.embedding_dim)
        self.combine_c2 = conv_combine(in_channels=c2_in_channels * 2, out_channels=self.embedding_dim)
        self.combine_c1 = conv_combine(in_channels=c1_in_channels * 2, out_channels=self.embedding_dim)

        self.cov_c4 = ConvLayers(c4_in_channels, self.embedding_dim, kernel_size=3, stride=1, padding=1)
        self.cov_c3 = ConvLayers(c3_in_channels, self.embedding_dim, kernel_size=3, stride=1, padding=1)
        self.cov_c2 = ConvLayers(c2_in_channels, self.embedding_dim, kernel_size=3, stride=1, padding=1)
        self.cov_c1 = ConvLayers(c1_in_channels, self.embedding_dim, kernel_size=3, stride=1, padding=1)

        self.fusion_c4 = FusionConv(in_channels=self.embedding_dim * 3, out_channels=self.embedding_dim)
        self.fusion_c3 = FusionConv(in_channels=self.embedding_dim * 3, out_channels=self.embedding_dim)
        self.fusion_c2 = FusionConv(in_channels=self.embedding_dim * 3, out_channels=self.embedding_dim)
        self.fusion_c1 = FusionConv(in_channels=self.embedding_dim * 3, out_channels=self.embedding_dim)

        self.ch_trans_c4 = ChangeTransformerV2(in_channels=embedding_dim, dim_in=8, stack_num=0, linear_dim=40)
        self.ch_trans_c3 = ChangeTransformerV2(in_channels=embedding_dim, dim_in=16, stack_num=0, linear_dim=40)
        self.ch_trans_c2 = ChangeTransformerV2(in_channels=embedding_dim, dim_in=32, stack_num=0, linear_dim=40)
        self.ch_trans_c1 = ChangeTransformerV2(in_channels=embedding_dim, dim_in=64, stack_num=0, linear_dim=40)

        # Final linear fusion layer
        self.sr = MultiScale_ChangeRelation_V2(in_channels=self.embedding_dim)
        self.sr_1 = MultiScale_ChangeRelation_V2(in_channels=self.embedding_dim)
        self.sr_2 = MultiScale_ChangeRelation_V2(in_channels=self.embedding_dim)

        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.final = finalConv(in_channel=192, mid_channel=64, out_channel=self.output_nc)
        # Final activation
        self.output_softmax = decoder_softmax
        self.active = nn.Sigmoid()

        if normal_init:
            self.init_weights()

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs1, inputs2, change):

        x_1 = self._transform_inputs(inputs1)  # len=4, 1/2, 1/4, 1/8, 1/16
        x_2 = self._transform_inputs(inputs2)  # len=4, 1/2, 1/4, 1/8, 1/16
        x_3 = self._transform_inputs(change)  # len=4, 1/2, 1/4, 1/8, 1/16

        # img1 and img2 features
        c1_1, c2_1, c3_1, c4_1 = x_1
        c1_2, c2_2, c3_2, c4_2 = x_2
        c1_3, c2_3, c3_3, c4_3 = x_3

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4_1.shape

        outputs = []
        outputs.append(c1_1)
        outputs.append(c2_1)
        outputs.append(c3_1)
        outputs.append(c4_1)

        outputs.append(c1_2)
        outputs.append(c2_2)
        outputs.append(c3_2)
        outputs.append(c4_2)

        outputs.append(c1_3)
        outputs.append(c2_3)
        outputs.append(c3_3)
        outputs.append(c4_3)
        # Stage 4: x1/32 scale
        _c4_1 = self.combine_c4(torch.cat((torch.add(torch.matmul(c4_1, c4_3, out=None), c4_1), c4_1), dim=1))
        _c4_2 = self.combine_c4(torch.cat((torch.add(torch.matmul(c4_2, c4_3, out=None), c4_2), c4_2), dim=1))
        _c4_3 = self.cov_c4(c4_3)
        _c4_1 = self.ch_trans_c4(_c4_1, _c4_2)
        _c4_2 = self.ch_trans_c4(_c4_2, _c4_1)
        _c4_d = self.fusion_c4(torch.cat((_c4_1, _c4_2, _c4_3), dim=1))
        outputs.append(_c4_1)
        outputs.append(_c4_2)
        outputs.append(_c4_d)
        # outputs.append(p_c4)

        # Stage 3: x1/16 scale
        _c3_1 = self.combine_c3(torch.cat((torch.add(torch.matmul(c3_1, c3_3, out=None), c3_1), c3_1), dim=1))
        _c3_2 = self.combine_c3(torch.cat((torch.add(torch.matmul(c3_2, c3_3, out=None), c3_2), c3_2), dim=1))
        _c3_3 = self.cov_c3(c3_3)
        _c3_1 = self.ch_trans_c3(_c3_1, _c3_2)
        _c3_2 = self.ch_trans_c3(_c3_2, _c3_1)
        _c3_d = self.fusion_c3(torch.cat((_c3_1, _c3_2, _c3_3), dim=1))
        outputs.append(_c3_1)
        outputs.append(_c3_2)
        outputs.append(_c3_d)
        # outputs.append(p_c3)

        # Stage 2: x1/8 scale
        _c2_1 = self.combine_c2(torch.cat((torch.add(torch.matmul(c2_1, c2_3, out=None), c2_1), c2_1), dim=1))
        _c2_2 = self.combine_c2(torch.cat((torch.add(torch.matmul(c2_2, c2_3, out=None), c2_2), c2_2), dim=1))
        _c2_3 = self.cov_c2(c2_3)
        _c2_1 = self.ch_trans_c2(_c2_1, _c2_2)
        _c2_2 = self.ch_trans_c2(_c2_2, _c2_1)
        _c2_d = self.fusion_c2(torch.cat((_c2_1, _c2_2, _c2_3), dim=1))
        outputs.append(_c2_1)
        outputs.append(_c2_2)
        outputs.append(_c2_d)
        # outputs.append(p_c2)

        # Stage 1: x1/4 scale
        _c1_1 = self.combine_c1(torch.cat((torch.add(torch.matmul(c1_1, c1_3, out=None), c1_1), c1_1), dim=1))
        _c1_2 = self.combine_c1(torch.cat((torch.add(torch.matmul(c1_2, c1_3, out=None), c1_2), c1_2), dim=1))
        _c1_3 = self.cov_c1(c1_3)
        _c1_1 = self.ch_trans_c1(_c1_1, _c1_2)
        _c1_2 = self.ch_trans_c1(_c1_2, _c1_1)
        _c1_d = self.fusion_c1(torch.cat((_c1_1, _c1_2, _c1_3), dim=1))
        outputs.append(_c1_1)
        outputs.append(_c1_2)
        outputs.append(_c1_d)
        # outputs.append(p_c1)

        # Linear Fusion of difference image from all scales
        _c_sr_1 = self.sr_1((_c4_1, _c3_1, _c2_1, _c1_1))

        _c_sr_2 = self.sr_2((_c4_2, _c3_2, _c2_2, _c1_2))

        _c_sr_d = self.sr((_c4_d, _c3_d, _c2_d, _c1_d))

        outputs.append(_c_sr_1)
        outputs.append(_c_sr_2)
        outputs.append(_c_sr_d)

        _c_sr_d = self.upsamplex4(_c_sr_d)
        _c_sr_1 = self.upsamplex4(_c_sr_1)
        _c_sr_2 = self.upsamplex4(_c_sr_2)

        outputs.append(_c_sr_1)
        outputs.append(_c_sr_2)
        outputs.append(_c_sr_d)

        cp = self.final(torch.cat((_c_sr_d, _c_sr_1, _c_sr_2), dim=1))

        outputs.append(cp)

        if self.output_softmax:
            temp = outputs
            outputs = []
            for pred in temp:
                outputs.append(self.active(pred))

        return outputs

    def init_weights(self):
        self.final.apply(init_weights)

        self.upsamplex4.apply(init_weights)

        self.sr_2.apply(init_weights)
        self.sr_1.apply(init_weights)
        self.sr.apply(init_weights)

        self.ch_trans_c4.apply(init_weights)
        self.ch_trans_c3.apply(init_weights)
        self.ch_trans_c2.apply(init_weights)
        self.ch_trans_c1.apply(init_weights)

        self.fusion_c4.apply(init_weights)
        self.fusion_c3.apply(init_weights)
        self.fusion_c2.apply(init_weights)
        self.fusion_c1.apply(init_weights)

        self.cov_c4.apply(init_weights)
        self.cov_c3.apply(init_weights)
        self.cov_c2.apply(init_weights)
        self.cov_c1.apply(init_weights)

        self.combine_c4.apply(init_weights)
        self.combine_c3.apply(init_weights)
        self.combine_c2.apply(init_weights)
        self.combine_c1.apply(init_weights)


class FTR_Decoder_V2(nn.Module):
    def __init__(self, input_transform='multiple_select', in_index=None, align_corners=True, in_channels=None,
                 embedding_dim=256, output_nc=2, decoder_softmax=False, normal_init=True):
        super(FTR_Decoder_V2, self).__init__()
        # assert
        self.in_index = in_index
        if in_index is None:
            self.in_index = [0, 1, 2, 3]
        self.in_channels = in_channels
        if in_channels is None:
            self.in_channels = [32, 64, 128, 256]
        # self.img_size = 256
        # settings
        self.input_transform = input_transform
        self.align_corners = align_corners
        self.embedding_dim = embedding_dim
        self.output_nc = output_nc
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        # convolutional Difference Modules
        self.combine_c4 = conv_combine(in_channels=c4_in_channels * 2, out_channels=self.embedding_dim)
        self.combine_c3 = conv_combine(in_channels=c3_in_channels * 2, out_channels=self.embedding_dim)
        self.combine_c2 = conv_combine(in_channels=c2_in_channels * 2, out_channels=self.embedding_dim)
        self.combine_c1 = conv_combine(in_channels=c1_in_channels * 2, out_channels=self.embedding_dim)

        self.cov_c4 = ConvLayers(c4_in_channels, self.embedding_dim, kernel_size=3, stride=1, padding=1)
        self.cov_c3 = ConvLayers(c3_in_channels, self.embedding_dim, kernel_size=3, stride=1, padding=1)
        self.cov_c2 = ConvLayers(c2_in_channels, self.embedding_dim, kernel_size=3, stride=1, padding=1)
        self.cov_c1 = ConvLayers(c1_in_channels, self.embedding_dim, kernel_size=3, stride=1, padding=1)

        self.fusion_c4 = FusionConv(in_channels=self.embedding_dim * 3, out_channels=self.embedding_dim)
        self.fusion_c3 = FusionConv(in_channels=self.embedding_dim * 3, out_channels=self.embedding_dim)
        self.fusion_c2 = FusionConv(in_channels=self.embedding_dim * 3, out_channels=self.embedding_dim)
        self.fusion_c1 = FusionConv(in_channels=self.embedding_dim * 3, out_channels=self.embedding_dim)

        self.ch_trans_c4 = ChangeTransformerV2(in_channels=embedding_dim, dim_in=8, stack_num=0, linear_dim=40)
        self.ch_trans_c3 = ChangeTransformerV2(in_channels=embedding_dim, dim_in=16, stack_num=0, linear_dim=40)
        self.ch_trans_c2 = ChangeTransformerV2(in_channels=embedding_dim, dim_in=32, stack_num=0, linear_dim=40)
        self.ch_trans_c1 = ChangeTransformerV2(in_channels=embedding_dim, dim_in=64, stack_num=0, linear_dim=40)

        # Final linear fusion layer
        self.sr = MultiScale_ChangeRelation_V2(in_channels=self.embedding_dim)
        self.sr_1 = MultiScale_ChangeRelation_V2(in_channels=self.embedding_dim)
        self.sr_2 = MultiScale_ChangeRelation_V2(in_channels=self.embedding_dim)

        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.final = finalConv(in_channel=64, mid_channel=64, out_channel=self.output_nc)
        # Final activation
        self.output_softmax = decoder_softmax
        self.active = nn.Sigmoid()

        if normal_init:
            self.init_weights()

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs1, inputs2, change):

        x_1 = self._transform_inputs(inputs1)  # len=4, 1/2, 1/4, 1/8, 1/16
        x_2 = self._transform_inputs(inputs2)  # len=4, 1/2, 1/4, 1/8, 1/16
        x_3 = self._transform_inputs(change)  # len=4, 1/2, 1/4, 1/8, 1/16

        # img1 and img2 features
        c1_1, c2_1, c3_1, c4_1 = x_1
        c1_2, c2_2, c3_2, c4_2 = x_2
        c1_3, c2_3, c3_3, c4_3 = x_3

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4_1.shape

        outputs = []
        # Stage 4: x1/32 scale
        _c4_1 = self.combine_c4(torch.cat((torch.add(torch.matmul(c4_1, c4_3, out=None), c4_1), c4_1), dim=1))
        _c4_2 = self.combine_c4(torch.cat((torch.add(torch.matmul(c4_2, c4_3, out=None), c4_2), c4_2), dim=1))
        _c4_3 = self.cov_c4(c4_3)
        _c4_1 = self.ch_trans_c4(_c4_1, _c4_2)
        _c4_2 = self.ch_trans_c4(_c4_2, _c4_1)
        _c4_d = self.fusion_c4(torch.cat((_c4_1, _c4_2, _c4_3), dim=1))
        # outputs.append(p_c4)

        # Stage 3: x1/16 scale
        _c3_1 = self.combine_c3(torch.cat((torch.add(torch.matmul(c3_1, c3_3, out=None), c3_1), c3_1), dim=1))
        _c3_2 = self.combine_c3(torch.cat((torch.add(torch.matmul(c3_2, c3_3, out=None), c3_2), c3_2), dim=1))
        _c3_3 = self.cov_c3(c3_3)
        _c3_1 = self.ch_trans_c3(_c3_1, _c3_2)
        _c3_2 = self.ch_trans_c3(_c3_2, _c3_1)
        _c3_d = self.fusion_c3(torch.cat((_c3_1, _c3_2, _c3_3), dim=1))
        # outputs.append(p_c3)

        # Stage 2: x1/8 scale
        _c2_1 = self.combine_c2(torch.cat((torch.add(torch.matmul(c2_1, c2_3, out=None), c2_1), c2_1), dim=1))
        _c2_2 = self.combine_c2(torch.cat((torch.add(torch.matmul(c2_2, c2_3, out=None), c2_2), c2_2), dim=1))
        _c2_3 = self.cov_c2(c2_3)
        _c2_1 = self.ch_trans_c2(_c2_1, _c2_2)
        _c2_2 = self.ch_trans_c2(_c2_2, _c2_1)
        _c2_d = self.fusion_c2(torch.cat((_c2_1, _c2_2, _c2_3), dim=1))
        # outputs.append(p_c2)

        # Stage 1: x1/4 scale
        _c1_1 = self.combine_c1(torch.cat((torch.add(torch.matmul(c1_1, c1_3, out=None), c1_1), c1_1), dim=1))
        _c1_2 = self.combine_c1(torch.cat((torch.add(torch.matmul(c1_2, c1_3, out=None), c1_2), c1_2), dim=1))
        _c1_3 = self.cov_c1(c1_3)
        _c1_1 = self.ch_trans_c1(_c1_1, _c1_2)
        _c1_2 = self.ch_trans_c1(_c1_2, _c1_1)
        _c1_d = self.fusion_c1(torch.cat((_c1_1, _c1_2, _c1_3), dim=1))
        # outputs.append(p_c1)

        # Linear Fusion of difference image from all scales
        _c_sr_1 = self.sr_1((_c4_1, _c3_1, _c2_1, _c1_1))

        _c_sr_2 = self.sr_2((_c4_2, _c3_2, _c2_2, _c1_2))

        _c_sr_d = self.sr((_c4_d, _c3_d, _c2_d, _c1_d))

        outputs.append(_c_sr_d)
        outputs.append(_c_sr_1)
        outputs.append(_c_sr_2)

        _c_sr_d = self.upsamplex4(_c_sr_d)
        _c_sr_1 = self.upsamplex4(_c_sr_1)
        _c_sr_2 = self.upsamplex4(_c_sr_2)

        cp = self.final(torch.abs(_c_sr_1 - _c_sr_2) + _c_sr_d)

        outputs.append(cp)

        if self.output_softmax:
            temp = outputs
            outputs = []
            for pred in temp:
                outputs.append(self.active(pred))

        return outputs

    def init_weights(self):
        self.final.apply(init_weights)

        self.upsamplex4.apply(init_weights)

        self.sr_2.apply(init_weights)
        self.sr_1.apply(init_weights)
        self.sr.apply(init_weights)

        self.ch_trans_c4.apply(init_weights)
        self.ch_trans_c3.apply(init_weights)
        self.ch_trans_c2.apply(init_weights)
        self.ch_trans_c1.apply(init_weights)

        self.fusion_c4.apply(init_weights)
        self.fusion_c3.apply(init_weights)
        self.fusion_c2.apply(init_weights)
        self.fusion_c1.apply(init_weights)

        self.cov_c4.apply(init_weights)
        self.cov_c3.apply(init_weights)
        self.cov_c2.apply(init_weights)
        self.cov_c1.apply(init_weights)

        self.combine_c4.apply(init_weights)
        self.combine_c3.apply(init_weights)
        self.combine_c2.apply(init_weights)
        self.combine_c1.apply(init_weights)


class FTR_Decoder_V3(nn.Module):
    def __init__(self, input_transform='multiple_select', in_index=None, align_corners=True, in_channels=None,
                 embedding_dim=256, output_nc=2, decoder_softmax=False, normal_init=True):
        super(FTR_Decoder_V3, self).__init__()
        # assert
        self.in_index = in_index
        if in_index is None:
            self.in_index = [0, 1, 2, 3]
        self.in_channels = in_channels
        if in_channels is None:
            self.in_channels = [32, 64, 128, 256]
        # self.img_size = 256
        # settings
        self.input_transform = input_transform
        self.align_corners = align_corners
        self.embedding_dim = embedding_dim
        self.output_nc = output_nc
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        # convolutional Difference Modules
        self.combine_c4 = conv_combine(in_channels=c4_in_channels * 2, out_channels=self.embedding_dim)
        self.combine_c3 = conv_combine(in_channels=c3_in_channels * 2, out_channels=self.embedding_dim)
        self.combine_c2 = conv_combine(in_channels=c2_in_channels * 2, out_channels=self.embedding_dim)
        self.combine_c1 = conv_combine(in_channels=c1_in_channels * 2, out_channels=self.embedding_dim)

        self.cov_c4 = ConvLayers(c4_in_channels, self.embedding_dim, kernel_size=3, stride=1, padding=1)
        self.cov_c3 = ConvLayers(c3_in_channels, self.embedding_dim, kernel_size=3, stride=1, padding=1)
        self.cov_c2 = ConvLayers(c2_in_channels, self.embedding_dim, kernel_size=3, stride=1, padding=1)
        self.cov_c1 = ConvLayers(c1_in_channels, self.embedding_dim, kernel_size=3, stride=1, padding=1)

        self.fusion_c4 = FusionConv(in_channels=self.embedding_dim * 3, out_channels=self.embedding_dim)
        self.fusion_c3 = FusionConv(in_channels=self.embedding_dim * 3, out_channels=self.embedding_dim)
        self.fusion_c2 = FusionConv(in_channels=self.embedding_dim * 3, out_channels=self.embedding_dim)
        self.fusion_c1 = FusionConv(in_channels=self.embedding_dim * 3, out_channels=self.embedding_dim)

        self.ch_trans_c4 = ChangeTransformerV2(in_channels=embedding_dim, dim_in=8, stack_num=0, linear_dim=40)
        self.ch_trans_c3 = ChangeTransformerV2(in_channels=embedding_dim, dim_in=16, stack_num=0, linear_dim=40)
        self.ch_trans_c2 = ChangeTransformerV2(in_channels=embedding_dim, dim_in=32, stack_num=0, linear_dim=40)
        self.ch_trans_c1 = ChangeTransformerV2(in_channels=embedding_dim, dim_in=64, stack_num=0, linear_dim=40)

        # Final linear fusion layer
        self.sr = MultiScale_ChangeRelation_V2(in_channels=self.embedding_dim)
        self.sr_1 = MultiScale_ChangeRelation_V2(in_channels=self.embedding_dim)
        self.sr_2 = MultiScale_ChangeRelation_V2(in_channels=self.embedding_dim)

        # Final linear fusion layer
        # self.linear_fuse = nn.Sequential(
        #     nn.Conv2d(in_channels=192, out_channels=self.embedding_dim, kernel_size=1),
        #     nn.BatchNorm2d(self.embedding_dim)
        # )
        # self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        # Final predction head
        self.convd2x = UpsampleConvLayer(64, 64, kernel_size=4, stride=2)
        self.dense_2x = nn.Sequential(ResidualBlock(64))
        self.convd1x = UpsampleConvLayer(64, 64, kernel_size=4, stride=2)
        self.dense_1x = nn.Sequential(ResidualBlock(64))
        self.final = finalConv(in_channel=256, mid_channel=64, out_channel=self.output_nc)
        # Final activation
        self.output_softmax = decoder_softmax
        self.active = nn.Sigmoid()

        if normal_init:
            self.init_weights()

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs1, inputs2, change):

        x_1 = self._transform_inputs(inputs1)  # len=4, 1/2, 1/4, 1/8, 1/16
        x_2 = self._transform_inputs(inputs2)  # len=4, 1/2, 1/4, 1/8, 1/16
        x_3 = self._transform_inputs(change)  # len=4, 1/2, 1/4, 1/8, 1/16

        # img1 and img2 features
        c1_1, c2_1, c3_1, c4_1 = x_1
        c1_2, c2_2, c3_2, c4_2 = x_2
        c1_3, c2_3, c3_3, c4_3 = x_3

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4_1.shape

        outputs = []
        # Stage 4: x1/32 scale
        _c4_1 = self.combine_c4(torch.cat((torch.add(torch.matmul(c4_1, c4_3, out=None), c4_1), c4_1), dim=1))
        _c4_2 = self.combine_c4(torch.cat((torch.add(torch.matmul(c4_2, c4_3, out=None), c4_2), c4_2), dim=1))
        _c4_3 = self.cov_c4(c4_3)
        _c4_1 = self.ch_trans_c4(_c4_1, _c4_2)
        _c4_2 = self.ch_trans_c4(_c4_2, _c4_1)
        _c4_d = self.fusion_c4(torch.cat((_c4_1, _c4_2, _c4_3), dim=1))
        # outputs.append(p_c4)

        # Stage 3: x1/16 scale
        _c3_1 = self.combine_c3(torch.cat((torch.add(torch.matmul(c3_1, c3_3, out=None), c3_1), c3_1), dim=1))
        _c3_2 = self.combine_c3(torch.cat((torch.add(torch.matmul(c3_2, c3_3, out=None), c3_2), c3_2), dim=1))
        _c3_3 = self.cov_c3(c3_3)
        _c3_1 = self.ch_trans_c3(_c3_1, _c3_2)
        _c3_2 = self.ch_trans_c3(_c3_2, _c3_1)
        _c3_d = self.fusion_c3(torch.cat((_c3_1, _c3_2, _c3_3), dim=1))
        # outputs.append(p_c3)

        # Stage 2: x1/8 scale
        _c2_1 = self.combine_c2(torch.cat((torch.add(torch.matmul(c2_1, c2_3, out=None), c2_1), c2_1), dim=1))
        _c2_2 = self.combine_c2(torch.cat((torch.add(torch.matmul(c2_2, c2_3, out=None), c2_2), c2_2), dim=1))
        _c2_3 = self.cov_c2(c2_3)
        _c2_1 = self.ch_trans_c2(_c2_1, _c2_2)
        _c2_2 = self.ch_trans_c2(_c2_2, _c2_1)
        _c2_d = self.fusion_c2(torch.cat((_c2_1, _c2_2, _c2_3), dim=1))
        # outputs.append(p_c2)

        # Stage 1: x1/4 scale
        _c1_1 = self.combine_c1(torch.cat((torch.add(torch.matmul(c1_1, c1_3, out=None), c1_1), c1_1), dim=1))
        _c1_2 = self.combine_c1(torch.cat((torch.add(torch.matmul(c1_2, c1_3, out=None), c1_2), c1_2), dim=1))
        _c1_3 = self.cov_c1(c1_3)
        _c1_1 = self.ch_trans_c1(_c1_1, _c1_2)
        _c1_2 = self.ch_trans_c1(_c1_2, _c1_1)
        _c1_d = self.fusion_c1(torch.cat((_c1_1, _c1_2, _c1_3), dim=1))
        # outputs.append(p_c1)

        # Linear Fusion of difference image from all scales
        _c_sr_1 = self.sr_1((_c4_1, _c3_1, _c2_1, _c1_1))

        _c_sr_2 = self.sr_2((_c4_2, _c3_2, _c2_2, _c1_2))

        _c_sr_d = self.sr((_c4_d, _c3_d, _c2_d, _c1_d))

        outputs.append(_c_sr_1)
        outputs.append(_c_sr_2)
        outputs.append(_c_sr_d)

        _c_sr_m = torch.abs(torch.sub(_c_sr_1, _c_sr_2))

        _c_sr_1 = self.convd2x(_c_sr_1)
        _c_sr_1 = self.dense_2x(_c_sr_1)
        _c_sr_1 = self.convd1x(_c_sr_1)
        _c_sr_1 = self.dense_1x(_c_sr_1)

        _c_sr_2 = self.convd2x(_c_sr_2)
        _c_sr_2 = self.dense_2x(_c_sr_2)
        _c_sr_2 = self.convd1x(_c_sr_2)
        _c_sr_2 = self.dense_1x(_c_sr_2)

        _c_sr_d = self.convd2x(_c_sr_d)
        _c_sr_d = self.dense_2x(_c_sr_d)
        _c_sr_d = self.convd1x(_c_sr_d)
        _c_sr_d = self.dense_1x(_c_sr_d)

        _c_sr_m = self.convd2x(_c_sr_m)
        _c_sr_m = self.dense_2x(_c_sr_m)
        _c_sr_m = self.convd1x(_c_sr_m)
        _c_sr_m = self.dense_1x(_c_sr_m)

        # Final prediction
        cp = self.final(torch.cat((_c_sr_1, _c_sr_2, _c_sr_d, _c_sr_m), dim=1))

        outputs.append(cp)

        if self.output_softmax:
            temp = outputs
            outputs = []
            for pred in temp:
                outputs.append(self.active(pred))

        return outputs

    def init_weights(self):

        self.sr_2.apply(init_weights)
        self.sr_1.apply(init_weights)
        self.sr.apply(init_weights)

        self.ch_trans_c4.apply(init_weights)
        self.ch_trans_c3.apply(init_weights)
        self.ch_trans_c2.apply(init_weights)
        self.ch_trans_c1.apply(init_weights)

        self.fusion_c4.apply(init_weights)
        self.fusion_c3.apply(init_weights)
        self.fusion_c2.apply(init_weights)
        self.fusion_c1.apply(init_weights)

        self.cov_c4.apply(init_weights)
        self.cov_c3.apply(init_weights)
        self.cov_c2.apply(init_weights)
        self.cov_c1.apply(init_weights)

        self.combine_c4.apply(init_weights)
        self.combine_c3.apply(init_weights)
        self.combine_c2.apply(init_weights)
        self.combine_c1.apply(init_weights)


class FTRNet_V1(nn.Module):

    def __init__(self, input_nc=3, output_nc=2, decoder_softmax=False, embed_dim=256):
        super(FTRNet_V1, self).__init__()
        # Transformer Encoder
        self.embed_dims = [64, 128, 320, 512]
        self.depths = [3, 4, 6, 3]  # [3, 3, 6, 18, 3]
        self.embedding_dim = embed_dim
        self.drop_rate = 0.0
        self.attn_drop = 0.
        self.drop_path_rate = 0.1

        self.Tenc_x2 = EncoderTransformer_MIT_v1(img_size=256, patch_size=4, in_chans=input_nc, num_classes=output_nc,
                                                 embed_dims=self.embed_dims, num_heads=[1, 2, 5, 8],
                                                 mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None,
                                                 drop_rate=self.drop_rate, attn_drop_rate=self.attn_drop,
                                                 drop_path_rate=self.drop_path_rate,
                                                 norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=self.depths,
                                                 sr_ratios=[8, 4, 2, 1])

        # Transformer Decoder
        self.TDec_x2 = FTR_Decoder(input_transform='multiple_select', in_index=[0, 1, 2, 3], align_corners=False,
                                   in_channels=self.embed_dims, embedding_dim=self.embedding_dim, output_nc=output_nc,
                                   decoder_softmax=decoder_softmax)

    def forward(self, x1, x2, c):
        [fx1, fx2, fx3] = [self.Tenc_x2(x1), self.Tenc_x2(x2), self.Tenc_x2(c)]

        cp = self.TDec_x2(fx1, fx2, fx3)

        # exit()
        return cp


class FTRNet_V2(nn.Module):

    def __init__(self, input_nc=3, output_nc=2, decoder_softmax=False, embed_dim=256):
        super(FTRNet_V2, self).__init__()
        # Transformer Encoder
        self.embed_dims = [64, 128, 320, 512]
        self.depths = [3, 3, 4, 3]  # [3, 3, 6, 18, 3]
        self.embedding_dim = embed_dim
        self.drop_rate = 0.1
        self.attn_drop = 0.1
        self.drop_path_rate = 0.1

        self.Tenc_x2 = EncoderTransformer_MIT_v2(img_size=256, patch_size=7, in_chans=input_nc, num_classes=output_nc,
                                                 embed_dims=self.embed_dims, num_heads=[1, 2, 4, 8],
                                                 mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None,
                                                 drop_rate=self.drop_rate, attn_drop_rate=self.attn_drop,
                                                 drop_path_rate=self.drop_path_rate,
                                                 norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=self.depths,
                                                 sr_ratios=[8, 4, 2, 1])

        # Transformer Decoder
        self.TDec_x2 = FTR_Decoder(input_transform='multiple_select', in_index=[0, 1, 2, 3], align_corners=False,
                                   in_channels=self.embed_dims, embedding_dim=self.embedding_dim, output_nc=output_nc,
                                   decoder_softmax=decoder_softmax)

    def forward(self, x1, x2, c):
        [fx1, fx2, fx3] = [self.Tenc_x2(x1), self.Tenc_x2(x2), self.Tenc_x2(c)]

        cp = self.TDec_x2(fx1, fx2, fx3)

        # exit()
        return cp


class FTRNet_V3(nn.Module):

    def __init__(self, input_nc=3, output_nc=2, decoder_softmax=False, embed_dim=256):
        super(FTRNet_V3, self).__init__()
        # Transformer Encoder
        self.embed_dims = [64, 128, 320, 512]
        self.depths = [3, 3, 4, 3]  # [3, 3, 6, 18, 3]
        self.embedding_dim = embed_dim
        self.drop_rate = 0.1
        self.attn_drop = 0.1
        self.drop_path_rate = 0.1

        self.Tenc_x2 = EncoderTransformer_MIT_v2(img_size=256, patch_size=7, in_chans=input_nc, num_classes=output_nc,
                                                 embed_dims=self.embed_dims, num_heads=[1, 2, 4, 8],
                                                 mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None,
                                                 drop_rate=self.drop_rate, attn_drop_rate=self.attn_drop,
                                                 drop_path_rate=self.drop_path_rate,
                                                 norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=self.depths,
                                                 sr_ratios=[8, 4, 2, 1])

        # Transformer Decoder
        self.TDec_x2 = FTR_Decoder_V2(input_transform='multiple_select', in_index=[0, 1, 2, 3], align_corners=False,
                                      in_channels=self.embed_dims, embedding_dim=self.embedding_dim,
                                      output_nc=output_nc,
                                      decoder_softmax=decoder_softmax)

    def forward(self, x1, x2, c):
        [fx1, fx2, fx3] = [self.Tenc_x2(x1), self.Tenc_x2(x2), self.Tenc_x2(c)]

        cp = self.TDec_x2(fx1, fx2, fx3)

        # exit()
        return cp


class FTRNet_V4(nn.Module):

    def __init__(self, input_nc=3, output_nc=2, decoder_softmax=False, embed_dim=256):
        super(FTRNet_V4, self).__init__()
        # Transformer Encoder
        self.embed_dims = [64, 128, 320, 512]
        self.depths = [3, 3, 4, 3]  # [3, 3, 6, 18, 3]
        self.embedding_dim = embed_dim
        self.drop_rate = 0.1
        self.attn_drop = 0.1
        self.drop_path_rate = 0.1

        self.Tenc_x2 = EncoderTransformer_MIT_v2(img_size=256, patch_size=7, in_chans=input_nc, num_classes=output_nc,
                                                 embed_dims=self.embed_dims, num_heads=[1, 2, 4, 8],
                                                 mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None,
                                                 drop_rate=self.drop_rate, attn_drop_rate=self.attn_drop,
                                                 drop_path_rate=self.drop_path_rate,
                                                 norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=self.depths,
                                                 sr_ratios=[8, 4, 2, 1])

        # Transformer Decoder
        self.TDec_x2 = FTR_Decoder_V3(input_transform='multiple_select', in_index=[0, 1, 2, 3], align_corners=False,
                                      in_channels=self.embed_dims, embedding_dim=self.embedding_dim,
                                      output_nc=output_nc,
                                      decoder_softmax=decoder_softmax)

    def forward(self, x1, x2, c):
        [fx1, fx2, fx3] = [self.Tenc_x2(x1), self.Tenc_x2(x2), self.Tenc_x2(c)]

        cp = self.TDec_x2(fx1, fx2, fx3)

        # exit()
        return cp


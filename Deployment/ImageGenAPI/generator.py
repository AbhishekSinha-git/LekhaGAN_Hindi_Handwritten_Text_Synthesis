import torch
import torch.nn as nn
import torch.nn.functional as F # <-- Added import


class SelfAttention(nn.Module):
    # ... (Include the SelfAttention class definition from previous examples) ...
    def __init__(self, in_dim):
        super().__init__()
        self.chanel_in = in_dim
        sn = nn.utils.spectral_norm
        self.query_conv = sn(nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1, bias=False))
        self.key_conv = sn(nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1, bias=False))
        self.value_conv = sn(nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, bias=False))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy / (self.chanel_in // 8)**0.5)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out

class SPADE(nn.Module):
    # ... (Include the SPADE class definition from previous examples) ...
    def __init__(self, norm_nc, label_nc, ks=3):
        super().__init__()
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        pw = ks // 2
        nhidden = 128
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
    def forward(self, x, cond_map):
        normalized = self.param_free_norm(x)
        if cond_map.size()[2:] != x.size()[2:]:
            cond_map_interp = F.interpolate(cond_map, size=x.size()[2:], mode='nearest')
        else:
            cond_map_interp = cond_map
        actv = self.mlp_shared(cond_map_interp)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta
        return out

class BigGANResBlockSPADEUp(nn.Module):
    def __init__(self, in_channels, out_channels, label_nc, upsample=True):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest') if upsample else None
        fmiddle = min(in_channels, out_channels)

        # Define layers
        self.norm1 = SPADE(in_channels, label_nc)
        self.conv1 = nn.Conv2d(in_channels, fmiddle, kernel_size=3, padding=1, bias=False)
        self.norm2 = SPADE(fmiddle, label_nc)
        self.conv2 = nn.Conv2d(fmiddle, out_channels, kernel_size=3, padding=1, bias=False)
        self.activation = nn.ReLU(True)

        # Shortcut connection
        self.learn_shortcut = (in_channels != out_channels) or upsample
        if self.learn_shortcut:
            self.conv_s = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            # No SPADE on shortcut path usually, but could be added

    def forward(self, x, cond_map):
        # Shortcut path
        x_s = x
        if self.upsample:
            x_s = self.upsample(x_s)
        if self.learn_shortcut:
            x_s = self.conv_s(x_s)

        # Main path
        dx = self.norm1(x, cond_map)
        dx = self.activation(dx)
        if self.upsample:
            dx = self.upsample(dx) # Upsample after first norm/act
        dx = self.conv1(dx)
        dx = self.norm2(dx, cond_map)
        dx = self.activation(dx)
        dx = self.conv2(dx)

        out = x_s + dx
        return out

class UNetStyleGenerator(nn.Module): # Renamed for clarity
    def __init__(self, fasttext_dim=300, noise_dim=100, input_channels=1, output_channels=1,
                 ngf=96, label_nc=128, add_attention=True):
        super().__init__()
        self.fasttext_dim = fasttext_dim
        self.noise_dim = noise_dim
        self.combined_cond_dim = fasttext_dim + noise_dim
        self.ngf = ngf
        self.label_nc = label_nc # Channel dim for SPADE condition map
        self.add_attention = add_attention

        # Conditioning projection
        self.init_h, self.init_w = 4, 8 # Smallest feature map size (matches enc4 output)
        self.fc_cond = nn.Linear(self.combined_cond_dim, self.label_nc * self.init_h * self.init_w)

        # --- Encoder ---
        # Input: (B, 1, 64, 128)
        self.enc1 = nn.Conv2d(input_channels, ngf, kernel_size=4, stride=2, padding=1) # Out: (B, ngf, 32, 64)
        self.enc2 = nn.Sequential(nn.LeakyReLU(0.2, True), nn.Conv2d(ngf    , ngf * 2, 4, 2, 1, bias=False), nn.InstanceNorm2d(ngf * 2)) # Out: (B, ngf*2, 16, 32)
        self.enc3 = nn.Sequential(nn.LeakyReLU(0.2, True), nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False), nn.InstanceNorm2d(ngf * 4)) # Out: (B, ngf*4, 8, 16)
        self.enc4 = nn.Sequential(nn.LeakyReLU(0.2, True), nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False), nn.InstanceNorm2d(ngf * 8)) # Out: (B, ngf*8, 4, 8)

        # --- Bottleneck --- (Processes features at the smallest spatial dim)
        self.bottleneck_conv = nn.Sequential(
             nn.LeakyReLU(0.2, True),
             nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, padding=1, bias=False), # Process at 4x8
             nn.InstanceNorm2d(ngf * 8),
             nn.ReLU(True) # Added ReLU often helps here
        )

        # --- Decoder using BigGAN-Style ResBlocks with Skip Connections ---
        # Decoder input channels = channels from previous layer + channels from corresponding encoder layer

        # Block 4: 4x8 -> 8x16
        # Input: bottleneck output (ngf*8). No skip connection here. Output: ngf*4
        self.dec4 = BigGANResBlockSPADEUp(in_channels=ngf * 8,         out_channels=ngf * 4, label_nc=self.label_nc, upsample=True)
        # Optional Attention after Block 4 (operates on ngf*4 channels)
        if self.add_attention: self.attn4 = SelfAttention(ngf * 4)

        # Block 3: 8x16 -> 16x32
        # Input: dec4 output (ngf*4) + enc3 output (ngf*4) = ngf*8. Output: ngf*2
        self.dec3 = BigGANResBlockSPADEUp(in_channels=ngf * 8,         out_channels=ngf * 2, label_nc=self.label_nc, upsample=True)
        # Optional Attention after Block 3 (operates on ngf*2 channels)
        if self.add_attention: self.attn3 = SelfAttention(ngf * 2)

        # Block 2: 16x32 -> 32x64
        # Input: dec3 output (ngf*2) + enc2 output (ngf*2) = ngf*4. Output: ngf
        self.dec2 = BigGANResBlockSPADEUp(in_channels=ngf * 4,         out_channels=ngf,     label_nc=self.label_nc, upsample=True)
        # Optional Attention after Block 2 (operates on ngf channels)
        # if self.add_attention: self.attn2 = SelfAttention(ngf) # Can add if needed

        # Block 1: 32x64 -> 64x128
        # Input: dec2 output (ngf) + enc1 output (ngf) = ngf*2. Output: ngf
        self.dec1 = BigGANResBlockSPADEUp(in_channels=ngf * 2,         out_channels=ngf,     label_nc=self.label_nc, upsample=True)

        # --- Final Output Layers ---
        self.final_norm = nn.InstanceNorm2d(ngf, affine=True) # Final norm before output conv
        self.final_activ = nn.ReLU(True)
        self.final_conv = nn.Conv2d(ngf, output_channels, kernel_size=3, padding=1) # Keep kernel 3x3
        self.final_output_act = nn.Sigmoid() # Use Sigmoid for [0, 1] output

    def forward(self, input_patch, fasttext_vector, noise):
        # --- Condition Processing ---
        cond_combined = torch.cat([fasttext_vector, noise], dim=1)
        cond_projected = self.fc_cond(cond_combined)
        # Reshape condition to smallest spatial size for SPADE blocks
        cond_map_base = cond_projected.view(-1, self.label_nc, self.init_h, self.init_w)

        # --- Encoder Path & Save Skip Connections ---
        e1 = self.enc1(input_patch) # (B, ngf,   32, 64)
        e2 = self.enc2(e1)          # (B, ngf*2, 16, 32)
        e3 = self.enc3(e2)          # (B, ngf*4, 8, 16)
        e4 = self.enc4(e3)          # (B, ngf*8, 4, 8)

        # --- Bottleneck ---
        bottleneck = self.bottleneck_conv(e4) # (B, ngf*8, 4, 8)

        # --- Decoder Path with Skip Connections ---
        # Dec4: No skip connection from encoder needed here
        d4_out = self.dec4(bottleneck, cond_map_base) # (B, ngf*4, 8, 16)
        if self.add_attention and hasattr(self, 'attn4'):
            d4_out = self.attn4(d4_out)

        # Dec3: Concatenate d4_out with e3
        d3_in = torch.cat([d4_out, e3], dim=1) # (B, ngf*4 + ngf*4 = ngf*8, 8, 16)
        d3_out = self.dec3(d3_in, cond_map_base) # (B, ngf*2, 16, 32)
        if self.add_attention and hasattr(self, 'attn3'):
            d3_out = self.attn3(d3_out)

        # Dec2: Concatenate d3_out with e2
        d2_in = torch.cat([d3_out, e2], dim=1) # (B, ngf*2 + ngf*2 = ngf*4, 16, 32)
        d2_out = self.dec2(d2_in, cond_map_base) # (B, ngf, 32, 64)
        # if self.add_attention and hasattr(self, 'attn2'):
        #     d2_out = self.attn2(d2_out)

        # Dec1: Concatenate d2_out with e1
        d1_in = torch.cat([d2_out, e1], dim=1) # (B, ngf + ngf = ngf*2, 32, 64)
        d1_out = self.dec1(d1_in, cond_map_base) # (B, ngf, 64, 128)

        # --- Final Layers ---
        x = self.final_norm(d1_out)
        x = self.final_activ(x)
        x = self.final_conv(x)
        output_patch = self.final_output_act(x) # (B, 1, 64, 128)

        return output_patch
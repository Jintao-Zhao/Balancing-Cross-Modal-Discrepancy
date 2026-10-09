import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.Modules_for_RGB import GCM3_for_RGB, GPM_for_RGB, REM11_for_RGB, BasicConv2d_for_RGB
from lib.Modules_for_T import GCM3_for_T, GPM_for_T, REM11_for_T, BasicConv2d_for_T

'''
backbone: resnet50
'''

class LogitsGatedFusion(nn.Module):
    def __init__(self, mid_channels=32, tau=1.0):
        super().__init__()
        self.tau = tau
        self.score = nn.Sequential(
            nn.Conv2d(2, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 2, kernel_size=3, padding=1)
        )
        # self.out_proj = nn.Conv2d(2, 1, kernel_size=1)

    def forward(self, logit_rgb, logit_t):
        x = torch.cat([logit_rgb, logit_t], dim=1)  # [B,2,H,W]
        s = self.score(x) / self.tau
        w = torch.softmax(s, dim=1)                 # [B,2,H,W]
        fused = (w * x).sum(dim=1, keepdim=True)    # [B,1,H,W]
        # fused = self.out_proj(torch.cat([w[:,0:1]*logit_rgb, w[:,1:2]*logit_t], dim=1))
        return fused, w



class Multi_Network(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channels=96):
        super(Multi_Network, self).__init__()
        self.encoder_RGB = timm.create_model(model_name="resnet50", pretrained=True,in_chans=3, features_only=True)
        self.encoder_T   = timm.create_model(model_name="resnet50", pretrained=True, in_chans=3, features_only=True)
        
        self.GCM3_RGB = GCM3_for_RGB(256, channels)
        self.GPM_RGB = GPM_for_RGB()
        self.REM11_RGB = REM11_for_RGB(channels, channels)
        self.LL_down_RGB = nn.Sequential(
            BasicConv2d_for_RGB(channels, channels, stride=2, kernel_size=3, padding=1),
            BasicConv2d_for_RGB(channels, channels, stride=2, kernel_size=3, padding=1)
        )
        self.dePixelShuffle_RGB = torch.nn.PixelShuffle(2)
        self.one_conv_f4_ll_RGB = nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=1)
        self.one_conv_f1_hh_RGB = nn.Conv2d(in_channels=channels + channels // 4, out_channels=channels, kernel_size=1)
        
        self.GCM3_T = GCM3_for_T(256, channels)
        self.GPM_T = GPM_for_T()
        self.REM11_T = REM11_for_T(channels, channels)
        self.LL_down_T = nn.Sequential(
            BasicConv2d_for_T(channels, channels, stride=2, kernel_size=3, padding=1),
            BasicConv2d_for_T(channels, channels, stride=2, kernel_size=3, padding=1)
        )
        self.dePixelShuffle_T = torch.nn.PixelShuffle(2)
        self.one_conv_f4_ll_T = nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=1)
        self.one_conv_f1_hh_T = nn.Conv2d(in_channels=channels + channels // 4, out_channels=channels, kernel_size=1)
        
        self.fusion = LogitsGatedFusion(mid_channels=32, tau=1.0)

    def set_tau(self, tau: float):
        self.fusion.tau = float(tau)
        
    def forward(self, RGB, thermal, use_rgb: bool = True, use_t: bool = True):
        RGB_feats = self.encoder_RGB(RGB)
        x0_RGB, x1_RGB, x2_RGB, x3_RGB, x4_RGB = RGB_feats
        # ↓ ↓ ↓ #
        # x0: stem（conv1 + bn + relu + maxpool）
        #
        #
        #
        #



        LL_RGB, LH_RGB, HL_RGB, HH_RGB, f1_RGB, f2_RGB, f3_RGB, f4_RGB = self.GCM3_RGB(x1_RGB, x2_RGB, x3_RGB, x4_RGB)
        


        HH_up_RGB = self.dePixelShuffle_RGB(HH_RGB)
        f1_HH_RGB = torch.cat([HH_up_RGB, f1_RGB], dim=1)
        f1_HH_RGB = self.one_conv_f1_hh_RGB(f1_HH_RGB)
        
        LL_down_RGB = self.LL_down_RGB(LL_RGB)
        f4_LL_RGB = torch.cat([LL_down_RGB, f4_RGB], dim=1)
        f4_LL_RGB = self.one_conv_f4_ll_RGB(f4_LL_RGB)
        
        prior_cam_RGB = self.GPM_RGB(x4_RGB)
        pred_0_RGB = F.interpolate(prior_cam_RGB, size=RGB.size()[2:], mode='bilinear', align_corners=False)
        f4_RGB, f3_RGB, f2_RGB, f1_RGB, bound_f4_RGB, bound_f3_RGB, bound_f2_RGB, bound_f1_RGB = \
            self.REM11_RGB([f1_HH_RGB, f2_RGB, f3_RGB, f4_LL_RGB], prior_cam_RGB, RGB)

        
        T_feats = self.encoder_T(thermal)
        x0_T, x1_T, x2_T, x3_T, x4_T = T_feats
        LL_T, LH_T, HL_T, HH_T, f1_T, f2_T, f3_T, f4_T = self.GCM3_T(x1_T, x2_T, x3_T, x4_T)
        
        HH_up_T = self.dePixelShuffle_T(HH_T)
        f1_HH_T = torch.cat([HH_up_T, f1_T], dim=1)
        f1_HH_T = self.one_conv_f1_hh_T(f1_HH_T)
        
        LL_down_T = self.LL_down_T(LL_T)
        f4_LL_T = torch.cat([LL_down_T, f4_T], dim=1)
        f4_LL_T = self.one_conv_f4_ll_T(f4_LL_T)
        
        prior_cam_T = self.GPM_T(x4_T)
        pred_0_T = F.interpolate(prior_cam_T, size=thermal.size()[2:], mode='bilinear', align_corners=False)
        f4_T, f3_T, f2_T, f1_T, bound_f4_T, bound_f3_T, bound_f2_T, bound_f1_T = \
            self.REM11_T([f1_HH_T, f2_T, f3_T, f4_LL_T], prior_cam_T, thermal)

        if not use_rgb:
            f1_RGB = torch.zeros_like(f1_RGB)
        if not use_t:
            f1_T   = torch.zeros_like(f1_T)
        
        fused_feat, fusion_w = self.fusion(f1_RGB, f1_T)  # fused_feat: [B,1,H,W], fusion_w: [B,2,H,W]


        
        return pred_0_RGB, f4_RGB, f3_RGB, f2_RGB, f1_RGB, bound_f4_RGB, bound_f3_RGB, bound_f2_RGB, bound_f1_RGB, \
               pred_0_T, f4_T, f3_T, f2_T, f1_T, bound_f4_T, bound_f3_T, bound_f2_T, bound_f1_T, fused_feat, fusion_w


if __name__ == '__main__':
    RGB = torch.rand(2, 3, 384, 384).cuda()
    thermal = torch.rand(2, 3, 384, 384).cuda()
    
    model = Multi_Network(96).cuda()
    outputs = model(RGB, thermal)
    
    output_names = [
        "pred_0_RGB", "f4_RGB", "f3_RGB", "f2_RGB", "f1_RGB", 
        "bound_f4_RGB", "bound_f3_RGB", "bound_f2_RGB", "bound_f1_RGB", 
        "pred_0_T", "f4_T", "f3_T", "f2_T", "f1_T", 
        "bound_f4_T", "bound_f3_T", "bound_f2_T", "bound_f1_T", "fused_feat"
    ]
    
    for name, output in zip(output_names, outputs):
        print(f"{name}: shape = {output.shape}")
    
    f1_RGB = outputs[4]
    f1_T   = outputs[13]
    print(f"f1_RGB channels = {f1_RGB.shape[1]}")
    print(f"f1_T channels   = {f1_T.shape[1]}")
    
    print("Forward pass test completed successfully!")
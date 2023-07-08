import torch.nn as nn
import torch.nn.functional as F
import torch
from mmcv.cnn import ConvModule, xavier_init, ContextBlock
#
# from mmdet.core import auto_fp16
from mmcv.runner import BaseModule, auto_fp16
from ..builder import NECKS


class channel_attention(nn.Module):
    def __init__(self):
        super(channel_attention,self).__init__()
        M = 5
        d = 128
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(nn.Conv2d(256, d, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(d),
                                nn.ReLU(inplace=False))
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                 nn.Conv2d(d, 256, kernel_size=1, stride=1)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs, num_levels):
        feats_U = torch.sum(inputs, dim=1)
        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)
        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors,dim=1)
        attention_vectors = attention_vectors.view(-1, num_levels, 256, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        return attention_vectors

class spatial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spatial_attention,self).__init__()
        M = 5
        d = 128
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.ModuleList([])
        for i in range(M):
            self.conv.append(
                 nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs, num_levels):
        inputs = torch.sum(inputs, dim=1)
        avg_out = torch.mean(inputs, dim=1, keepdim=True)
        max_out, _ = torch.max(inputs, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)

        attention_vectors = [conv(x) for conv in self.conv]
        attention_vectors = torch.cat(attention_vectors,dim=1)
        attention_vectors = self.softmax(attention_vectors)
        return attention_vectors.unsqueeze(2)


class selective_attention(nn.Module):

    def __init__(self, refine_level):
        super(selective_attention,self).__init__()
        self.refine_level = refine_level
        self.channel_att = channel_attention()
        self.spatial_att = spatial_attention()
        self.refine = ContextBlock(256, 1./16)
        # self.refine = ContextBlock(256, 1. / 2)
        
    def forward(self, inputs):
        feats = []
        gather_size = inputs[self.refine_level].size()[2:]
        num_levels = len(inputs)
        for i in range(num_levels):
            if i < self.refine_level:
                gathered = F.adaptive_max_pool2d(
                    inputs[i], output_size=gather_size)
            else:
                gathered = F.interpolate(
                    inputs[i], size=gather_size, mode='nearest')
            feats.append(gathered)
        feats = torch.cat(feats, dim=1)
        feats = feats.view(feats.shape[0], num_levels, 256, feats.shape[2], feats.shape[3])
        
        channel_attention_vectors = self.channel_att(feats,num_levels)
        feats_C = torch.sum(feats*channel_attention_vectors, dim=1)
        
        spatial_attention_vectors = self.spatial_att(feats,num_levels)
        # return [spatial_attention_vectors[:,0,:,:,:],spatial_attention_vectors[:,1,:,:,:],spatial_attention_vectors[:,2,:,:,:],
        #         spatial_attention_vectors[:,3,:,:,:],spatial_attention_vectors[:,4,:,:,:]]
        feats_S = torch.sum(feats*spatial_attention_vectors, dim=1)

        feats_sum = feats_C + feats_S
        # feats_sum = feats_S
        # bsf = feats_sum
        bsf = self.refine(feats_sum)
        
        residual = F.adaptive_max_pool2d(bsf, output_size=gather_size)
        return residual + inputs[self.refine_level]
        

@NECKS.register_module()
class AttFPN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest')):
        super(AttFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.refine_level = 2
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # For compatibility with previous release
                # TODO: deprecate `extra_convs_on_inputs`
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'
                

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)
        self.selective_attention_0 = selective_attention(0)
        self.selective_attention_1 = selective_attention(1)
        self.selective_attention_2 = selective_attention(2)
        self.selective_attention_3 = selective_attention(3)
        self.selective_attention_4 = selective_attention(4)
        
    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def enhance_feature(self, inputs):
        output = []
        output_0 = self.selective_attention_0(inputs)
        output_1 = self.selective_attention_1(inputs)
        output_2 = self.selective_attention_2(inputs)
        output_3 = self.selective_attention_3(inputs)
        output_4 = self.selective_attention_4(inputs)
        # return output_4
        output.append(output_0)
        output.append(output_1)
        output.append(output_2)
        output.append(output_3)
        output.append(output_4)
        return tuple(output)
        
    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)
        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        
        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        outs = self.enhance_feature(outs)
        return tuple(outs)

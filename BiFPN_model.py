import torch
import torch.nn.functional as F

class DoubleConv2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv2D, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=False),

        )

    # THE FORWARD IS TO BE ABLE TO MOVE THROUGH UNET
    def forward(self, x):
        return self.conv(x)


class DepthwiseConvBlock(torch.nn.Module):
    """
    Depthwise seperable convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, freeze_bn=False):
        super(DepthwiseConvBlock, self).__init__()
        self.depthwise = torch.nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                                         padding, dilation, groups=in_channels, bias=False)
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                         stride=1, padding=0, dilation=1, groups=1, bias=False)

        self.bn = torch.nn.BatchNorm2d(out_channels, momentum=0.9997, eps=4e-5)
        self.act = torch.nn.ReLU(inplace=False)

    def forward(self, inputs):
        x = self.depthwise(inputs)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class BiFPNBlock(torch.nn.Module):
    """
    Bi-directional Feature Pyramid Network
    """

    def __init__(self, feature_size=256, features = [32, 64, 128, 256] , epsilon=0.0001):
        super(BiFPNBlock, self).__init__()
        self.epsilon = epsilon
        self.features = features

        self.p1_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p2_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p3_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p4_td = DepthwiseConvBlock(feature_size, feature_size)


        self.p4_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p3_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p2_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p1_out = DepthwiseConvBlock(feature_size, feature_size)

        # TODO: Init weights
        self.w1 = torch.nn.Parameter(torch.Tensor(5, 3))
        self.w1_relu = torch.nn.ReLU(inplace=False)


        self.p1_ch_in_adj = torch.nn.Conv2d(in_channels=936, out_channels=feature_size, kernel_size=1,
                                            stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.p2_ch_in_adj = torch.nn.Conv2d(in_channels=312, out_channels=feature_size, kernel_size=1,
                                            stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.p3_ch_in_adj = torch.nn.Conv2d(in_channels=104, out_channels=feature_size, kernel_size=1,
                                            stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.p4_ch_in_adj = torch.nn.Conv2d(in_channels=36, out_channels=feature_size, kernel_size=1,
                                            stride=1, padding=0, dilation=1, groups=1, bias=False)

        self.p1_ch_out_adj = torch.nn.Conv2d(feature_size, 936, kernel_size=1,
                                            stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.p2_ch_out_adj = torch.nn.Conv2d(feature_size, 312, kernel_size=1,
                                 stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.p3_ch_out_adj= torch.nn.Conv2d(feature_size, 104, kernel_size=1,
                                 stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.p4_ch_out_adj = torch.nn.Conv2d(feature_size, 36, kernel_size=1,
                                 stride=1, padding=0, dilation=1, groups=1, bias=False)

    #     self.initialize_weights()
    #
    ## ONLY use if weights have to be reinitialized
    # def initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, torch.nn.Conv3d):
    #             torch.nn.init.kaiming_normal_(m.weight)
    #         elif isinstance(m, torch.nn.BatchNorm3d):
    #             torch.nn.init.constant_(m.weight, 1)
    #             torch.nn.init.constant_(m.bias, 0)


    def forward(self, p4, p3, p2, p1):
        p4_x = p4
        p3_x = p3
        p2_x = p2
        p1_x = p1

        # Calculate Top-Down Pathway
        w1 = self.w1
        w1 = w1 / torch.sum(w1, dim=0) + self.epsilon

        #Adjusting the inputs before BiFPN
        p4_x_in = self.p4_ch_in_adj(p4_x)
        p3_x_in = self.p3_ch_in_adj(p3_x)
        p2_x_in = self.p2_ch_in_adj(p2_x)
        p1_x_in = self.p1_ch_in_adj(p1_x)

        p4_td = self.p4_td(p4_x_in)

        p3_td = self.p3_td(w1[0, 0] * p3_x_in + w1[1, 0] * F.interpolate(p4_td, size=(p3_x_in.shape[2],p3_x_in.shape[3])))
        p2_td = self.p2_td(w1[0, 1] * p2_x_in + w1[1, 1] * F.interpolate(p3_td, size=(p2_x_in.shape[2],p2_x_in.shape[3])))
        p1_td = self.p1_td(w1[0, 2] * p1_x_in + w1[1, 2] * F.interpolate(p2_td, size=(p1_x_in.shape[2],p1_x_in.shape[3])))

        # Calculate Bottom-Up Pathway
        p1_out = self.p1_out(p1_td)
        p2_out = self.p2_out(w1[2, 0] * p2_x_in + w1[3, 0] * p2_td + w1[4, 0] * F.interpolate(p1_out, size=(p2_x_in.shape[2],p2_x_in.shape[3])))
        p3_out = self.p3_out(w1[2, 1] * p3_x_in + w1[3, 1] * p3_td + w1[4, 1] * F.interpolate(p2_out, size=(p3_x_in.shape[2],p3_x_in.shape[3])))
        p4_out = self.p4_out(w1[2, 2] * p4_x_in + w1[3, 2] * p4_td + w1[4, 2] * F.interpolate(p3_out, size=(p4_x_in.shape[2],p4_x_in.shape[3])))

        p4_out = self.p4_ch_out_adj(p4_out)
        p3_out = self.p3_ch_out_adj(p3_out)
        p2_out = self.p2_ch_out_adj(p2_out)
        p1_out = self.p1_ch_out_adj(p1_out)

        return p4_out, p3_out, p2_out, p1_out


class BiFPN(torch.nn.Module):
    def __init__(self, feature_size=256, num_layers= 2, epsilon=0.0001):
        super(BiFPN, self).__init__()
        self.bifpns = torch.nn.ModuleList([
            BiFPNBlock(feature_size) for _ in range(num_layers)
        ])

    def forward(self, p4_x, p3_x, p2_x, p1_x):
        p4 = p4_x
        p3 = p3_x
        p2 = p2_x
        p1 = p1_x # Assign initial values

        for bifpn in self.bifpns:
            p4, p3, p2, p1 = bifpn(p4, p3, p2, p1)

        return p4, p3, p2, p1


class Myops_UNET(torch.nn.Module):
    def __init__(self, in_channels =4, out_channels = 4 , features=[32, 64, 128, 256]):
        super(Myops_UNET, self).__init__()
        self.down1 = DoubleConv2D(in_channels, 32)
        self.down2 = DoubleConv2D(36, 68)
        self.down3 = DoubleConv2D(104, 208)
        self.down4 = DoubleConv2D(312, 624)

        self.ups1 = DoubleConv2D(1872, 936)
        self.ups2 = DoubleConv2D(624, 312)
        self.ups3 = DoubleConv2D(208, 104)
        self.ups4 = DoubleConv2D(72, 36)

        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.up_sampling1 = torch.nn.ConvTranspose2d(1872, 936, kernel_size=2, stride=2)
        self.up_sampling2 = torch.nn.ConvTranspose2d(936, 312, kernel_size=2, stride=2)
        self.up_sampling3 = torch.nn.ConvTranspose2d(312, 104, kernel_size=2, stride=2)
        self.up_sampling4 = torch.nn.ConvTranspose2d(104, 36, kernel_size=2, stride=2)
        self.features = features
        self.bi_fpn = BiFPN()
        self.bottleneck = (DoubleConv2D(936, 1872))
        self.pre_final_conv_1 = (torch.nn.Conv2d(36, 12, kernel_size=1))
        self.pre_final_conv_2 = (torch.nn.Conv2d(12, 6, kernel_size=1))
        self.final_conv = (torch.nn.Conv2d(6, out_channels, kernel_size=1))


    # THE FORWARD IS TO BE ABLE TO MOVE THROUGH UNET
    def forward(self, x1):

        x1_out = self.down1(x1)
        x1 = F.interpolate(x1, size=(x1_out.shape[2], x1_out.shape[3]), mode='bilinear',align_corners=False)
        p4_x = torch.cat((x1_out, x1), dim=1)
        x1_out_pooled = self.pool(p4_x)

        x2_out = self.down2(x1_out_pooled)
        x1_out_pooled = F.interpolate(x1_out_pooled, size=(x2_out.shape[2], x2_out.shape[3]), mode='bilinear',align_corners=False)
        p3_x = torch.cat((x2_out, x1_out_pooled), dim=1)
        x2_out_pooled = self.pool(p3_x)

        x3_out = self.down3(x2_out_pooled)
        x2_out_pooled = F.interpolate(x2_out_pooled, size=(x3_out.shape[2], x3_out.shape[3]), mode='bilinear',align_corners=False)
        p2_x = torch.cat((x3_out, x2_out_pooled), dim=1)
        x3_out_pooled = self.pool(p2_x)

        x4_out = self.down4(x3_out_pooled)
        x3_out_pooled = F.interpolate(x3_out_pooled, size=(x4_out.shape[2], x4_out.shape[3]), mode='bilinear',align_corners=False)
        p1_x = torch.cat((x4_out, x3_out_pooled), dim=1)
        x4_out_pooled = self.pool(p1_x)

        x_bottleneck = self.bottleneck(x4_out_pooled)

        p4, p3, p2, p1 = self.bi_fpn(p4_x, p3_x, p2_x, p1_x)
        skip_connection = [p4, p3, p2, p1]



        x1_umspl = self.up_sampling1(x_bottleneck)
        x1_umspl = F.interpolate(x1_umspl, size=(skip_connection[3].shape[2], skip_connection[3].shape[3]),mode='bilinear', align_corners=False)
        x1_up_skip_cat = torch.cat((skip_connection[3], x1_umspl), dim=1)
        x1_up_out = self.ups1(x1_up_skip_cat)


        x2_umspl = self.up_sampling2(x1_up_out)
        x2_umspl = F.interpolate(x2_umspl, size=(skip_connection[2].shape[2], skip_connection[2].shape[3]),mode='bilinear', align_corners=False)
        x2_up_skip_cat = torch.cat((skip_connection[2], x2_umspl), dim=1)
        x2_up_out = self.ups2(x2_up_skip_cat)


        x3_umspl = self.up_sampling3(x2_up_out)
        x3_umspl = F.interpolate(x3_umspl, size=(skip_connection[1].shape[2], skip_connection[1].shape[3]),mode='bilinear', align_corners=False)
        x3_up_skip_cat = torch.cat((skip_connection[1], x3_umspl), dim=1)
        x3_up_out = self.ups3(x3_up_skip_cat)


        x4_umspl = self.up_sampling4(x3_up_out)
        x4_umspl = F.interpolate(x4_umspl, size=(skip_connection[0].shape[2], skip_connection[0].shape[3]),mode='bilinear', align_corners=False)
        x4_up_skip_cat = torch.cat((skip_connection[0], x4_umspl), dim=1)
        x4_up_out = self.ups4(x4_up_skip_cat)

        pre_final_out_1 = self.pre_final_conv_1(x4_up_out)

        pre_final_out_2 = self.pre_final_conv_2(pre_final_out_1)

        return self.final_conv(pre_final_out_2)


#-----------------------------------------------End of Custom built Model-----------------------------------------------------#

model = Myops_UNET()

# Only use if weights need to be RESET
def weight_reset(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear)\
            or isinstance(m, torch.nn.BatchNorm2d)  or isinstance(m, torch.nn.ConvTranspose2d):
        m.reset_parameters()

model.apply(weight_reset)


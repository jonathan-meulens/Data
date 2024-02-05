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

    def forward(self, x):
        return self.conv(x)

class Dense_UNET(torch.nn.Module):
    def __init__(self, in_channels = 1, out_channels = 2 , features=[32, 64, 128, 256]):
        super(Dense_UNET, self).__init__()
        self.down1 = DoubleConv2D(in_channels, 35)
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

        skip_connection = [p4_x, p3_x, p2_x, p1_x]



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
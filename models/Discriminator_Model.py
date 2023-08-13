from torch import nn

class Discriminator(nn.Module):
    def __init__(self, im_chan= 1, hidden_dim = 16):
        super().__init__()
        self.disc = nn.Sequential(
            self.disc_block(im_chan, hidden_dim),
            self.disc_block(hidden_dim, hidden_dim*2),
            nn.Conv2d(hidden_dim*2, 1,kernel_size= 4)
        )
    
    def disc_block(self, input_channels, output_channels, kernel_size=4, stride = 2):
        return nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels= output_channels, kernel_size= kernel_size, stride= stride),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace= True))
    
    def forward(self, image):
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred), -1)
from torch import nn

class Generator(nn.Module):
    def __init__(self, z_dim = 10, im_chan=1, hidden_dim = 64):
        super().__init__()
        self.z_dim = z_dim
        self.gen = nn.Sequential(
                                self.generator_block(z_dim, hidden_dim*4),
                                self.generator_block(hidden_dim*4, hidden_dim*2, kernel_size=4, stride=1),
                                self.generator_block(hidden_dim*2, hidden_dim),
                                nn.ConvTranspose2d(hidden_dim, im_chan, kernel_size= 4, stride = 2),
                                nn.Tanh()
                                )
    def generator_block(self, input_channels, output_channels, kernel_size = 3, stride =2):
        return nn.Sequential(
                            nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                            nn.BatchNorm2d(output_channels),
                            nn.ReLU(inplace=True)
                            )
    def forward(self, noise):
        x = noise.view(len(noise), self.z_dim, 1, 1)
        return self.gen(x)

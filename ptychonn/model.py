import numpy as np
import torch, torchvision
import torch.nn as nn

nconv = 32


class recon_model(nn.Module):

    def __init__(self):
        super(recon_model, self).__init__()

        self.encoder = nn.Sequential( # Appears sequential has similar functionality as TF avoiding need for separate model definition and activ
          nn.Conv2d(in_channels=1, out_channels=nconv, kernel_size=3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Conv2d(nconv, nconv, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.MaxPool2d((2,2)),

          nn.Conv2d(nconv, nconv*2, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=(1,1)),          
          nn.ReLU(),
          nn.MaxPool2d((2,2)),

          nn.Conv2d(nconv*2, nconv*4, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=(1,1)),          
          nn.ReLU(),
          nn.MaxPool2d((2,2)),
          )

        self.decoder1 = nn.Sequential(

          nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Upsample(scale_factor=2, mode='bilinear'),

          nn.Conv2d(nconv*4, nconv*2, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Upsample(scale_factor=2, mode='bilinear'),
            
          nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Upsample(scale_factor=2, mode='bilinear'),

          nn.Conv2d(nconv*2, 1, 3, stride=1, padding=(1,1)),
          nn.Sigmoid() #Amplitude model
          )

        self.decoder2 = nn.Sequential(

          nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Upsample(scale_factor=2, mode='bilinear'),

          nn.Conv2d(nconv*4, nconv*2, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Upsample(scale_factor=2, mode='bilinear'),
            
          nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Upsample(scale_factor=2, mode='bilinear'),

          nn.Conv2d(nconv*2, 1, 3, stride=1, padding=(1,1)),
          nn.Tanh() #Phase model
          )

        # def weight_reset(m):
        #   if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        #     m.reset_parameters()

        # torch.manual_seed(1)
        # torch.cuda.manual_seed(1)
        # torch.cuda.manual_seed_all(1)
        # np.random.seed(1)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

        # self.apply(weight_reset)

        # print(hash(super()))
        # print(hash(self.encoder))
        # print(hash(self.decoder1))
        # print(hash(self.decoder2))

    def forward(self,x):
        x1 = self.encoder(x)
        amp = self.decoder1(x1)
        ph = self.decoder2(x1)

        #Restore -pi to pi range
        ph = ph*np.pi #Using tanh activation (-1 to 1) for phase so multiply by pi

        return amp,ph


# CHAT GPT generated
# Approximately 4 times more parameter(~5M)
class LargeReconModel(nn.Module):
    def __init__(self, nconv=64):  # Increased base filters
        super(LargeReconModel, self).__init__()
        
        # Encoder with more layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, nconv, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(nconv, nconv, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(nconv, nconv, 3, stride=1, padding=1), nn.ReLU(),  # Extra layer
            nn.MaxPool2d(2),
            
            nn.Conv2d(nconv, nconv*2, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=1), nn.ReLU(),  # Extra layer
            nn.MaxPool2d(2),
            
            nn.Conv2d(nconv*2, nconv*4, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=1), nn.ReLU(),  # Extra layer
            nn.MaxPool2d(2),
        )
        
        # Decoders with more layers
        self.decoder1 = self._make_decoder(nconv)
        self.decoder2 = self._make_decoder(nconv, final_activation=nn.Tanh())

    def _make_decoder(self, nconv, final_activation=nn.Sigmoid()):
        return nn.Sequential(
            nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=1), nn.ReLU(),  # Extra layer
            nn.Upsample(scale_factor=2, mode='bilinear'),
            
            nn.Conv2d(nconv*4, nconv*2, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=1), nn.ReLU(),  # Extra layer
            nn.Upsample(scale_factor=2, mode='bilinear'),
            
            nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=1), nn.ReLU(),  # Extra layer
            nn.Upsample(scale_factor=2, mode='bilinear'),
            
            nn.Conv2d(nconv*2, 1, 3, stride=1, padding=1),
            final_activation()
        )
    
    def forward(self, x):
        x1 = self.encoder(x)
        amp = self.decoder1(x1)
        ph = self.decoder2(x1) * torch.pi  # Phase restored to -π to π
        return amp, ph
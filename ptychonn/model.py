import time
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
class ReconModel5M(nn.Module):
    def __init__(self, nconv=64):  # Increased base filters
        super(ReconModel5M, self).__init__()
        
        # Encoder with more layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, nconv, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(nconv, nconv, 3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(nconv, nconv*2, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(nconv*2, nconv*4, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Decoders with more layers
        self.decoder1 = self._make_decoder(nconv)
        self.decoder2 = self._make_decoder(nconv, final_activation=nn.Tanh)

    def _make_decoder(self, nconv, final_activation=nn.Sigmoid):
        return nn.Sequential(
            nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            
            nn.Conv2d(nconv*4, nconv*2, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            
            nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            
            nn.Conv2d(nconv*2, 1, 3, stride=1, padding=1),
            final_activation()
        )
    
    def forward(self, x):
        x1 = self.encoder(x)
        amp = self.decoder1(x1)
        ph = self.decoder2(x1) * torch.pi  # Phase restored to -π to π
        return amp, ph


# CHAT GPT assisted
# Approximately 8 times more parameter(~10M)
class ReconModel10M(nn.Module):
    def __init__(self, nconv=64):  # Increased base filters
        super(ReconModel10M, self).__init__()
        
        # Encoder with more layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, nconv, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(nconv, nconv, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(nconv, nconv, 3, stride=1, padding=1), nn.ReLU(),  # Extra layer
            nn.MaxPool2d(2),
            
            nn.Conv2d(nconv, nconv*2, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=1), nn.ReLU(),  # Extra layer
            nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=1), nn.ReLU(),  # Extra layer
            nn.MaxPool2d(2),
            
            nn.Conv2d(nconv*2, nconv*4, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=1), nn.ReLU(),  # Extra layer
            nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=1), nn.ReLU(),  # Extra layer
            nn.MaxPool2d(2),
        )
        
        # Decoders with more layers
        self.decoder1 = self._make_decoder(nconv)
        self.decoder2 = self._make_decoder(nconv, final_activation=nn.Tanh)

    def _make_decoder(self, nconv, final_activation=nn.Sigmoid):
        return nn.Sequential(
            nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=1), nn.ReLU(),  # Extra layer
            nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=1), nn.ReLU(),  # Extra layer
            nn.Upsample(scale_factor=2, mode='bilinear'),
            
            nn.Conv2d(nconv*4, nconv*2, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=1), nn.ReLU(),  # Extra layer
            nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=1), nn.ReLU(),  # Extra layer
            nn.Upsample(scale_factor=2, mode='bilinear'),
            
            nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=1), nn.ReLU(),  # Extra layer
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



# CHAT GPT assisted
# Approximately 16 times more parameter(~20M)
class ReconModel20M(nn.Module):
    def __init__(self, nconv=96):  # Further increased base filters
        super(ReconModel20M, self).__init__()
        
        # Encoder with more layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, nconv, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(nconv, nconv, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(nconv, nconv, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(nconv, nconv, 3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(nconv, nconv*2, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(nconv*2, nconv*4, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Decoders with more layers
        self.decoder1 = self._make_decoder(nconv)
        self.decoder2 = self._make_decoder(nconv, final_activation=nn.Tanh)

    def _make_decoder(self, nconv, final_activation=nn.Sigmoid):
        return nn.Sequential(
            nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),

            nn.Conv2d(nconv*4, nconv*2, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),

            nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),

            nn.Conv2d(nconv*2, 1, 3, stride=1, padding=1),
            final_activation()
        )
    
    def forward(self, x):
        x1 = self.encoder(x)
        amp = self.decoder1(x1)
        ph = self.decoder2(x1) * torch.pi  # Phase restored to -π to π
        return amp, ph


# CHAT GPT assisted
# Approximately  100M  parameter(~100M)
class ReconModel100M(nn.Module):
    def __init__(self, nconv=128):  # Further increased base filters
        super(ReconModel100M, self).__init__()
        
        # Encoder with many layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, nconv, 3, stride=1, padding=1), nn.ReLU(),
            *[nn.Sequential(nn.Conv2d(nconv, nconv, 3, stride=1, padding=1), nn.ReLU()) for _ in range(10)],
            nn.MaxPool2d(2),
            
            nn.Conv2d(nconv, nconv*2, 3, stride=1, padding=1), nn.ReLU(),
            *[nn.Sequential(nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=1), nn.ReLU()) for _ in range(10)],
            nn.MaxPool2d(2),
            
            nn.Conv2d(nconv*2, nconv*4, 3, stride=1, padding=1), nn.ReLU(),
            *[nn.Sequential(nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=1), nn.ReLU()) for _ in range(10)],
            nn.MaxPool2d(2),
        )
        
        # Decoders with many layers
        self.decoder1 = self._make_decoder(nconv)
        self.decoder2 = self._make_decoder(nconv, final_activation=nn.Tanh)

    def _make_decoder(self, nconv, final_activation=nn.Sigmoid):
        return nn.Sequential(
            *[nn.Sequential(nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=1), nn.ReLU()) for _ in range(10)],
            nn.Upsample(scale_factor=2, mode='bilinear'),

            nn.Conv2d(nconv*4, nconv*2, 3, stride=1, padding=1),
            *[nn.Sequential(nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=1), nn.ReLU()) for _ in range(10)],
            nn.Upsample(scale_factor=2, mode='bilinear'),

            *[nn.Sequential(nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=1), nn.ReLU()) for _ in range(6)],
            nn.Upsample(scale_factor=2, mode='bilinear'),

            nn.Conv2d(nconv*2, 1, 3, stride=1, padding=1),
            final_activation()
        )
    
    def forward(self, x):
        x1 = self.encoder(x)
        amp = self.decoder1(x1)
        ph = self.decoder2(x1) * torch.pi

        return amp, ph


# CHAT GPT assisted
# Approximately  200M  parameter(~200M)
class ReconModel200M(nn.Module):
    def __init__(self, nconv=256):  # Further increased base filters
        super(ReconModel200M, self).__init__()
        
        # Encoder with many layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, nconv, 3, stride=1, padding=1), nn.ReLU(),
            *[nn.Sequential(nn.Conv2d(nconv, nconv, 3, stride=1, padding=1), nn.ReLU()) for _ in range(6)],
            nn.MaxPool2d(2),
            
            nn.Conv2d(nconv, nconv*2, 3, stride=1, padding=1), nn.ReLU(),
            *[nn.Sequential(nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=1), nn.ReLU()) for _ in range(6)],
            nn.MaxPool2d(2),
            
            nn.Conv2d(nconv*2, nconv*4, 3, stride=1, padding=1), nn.ReLU(),
            *[nn.Sequential(nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=1), nn.ReLU()) for _ in range(6)],
            nn.MaxPool2d(2),
        )
        
        # Decoders with many layers
        self.decoder1 = self._make_decoder(nconv)
        self.decoder2 = self._make_decoder(nconv, final_activation=nn.Tanh)

    def _make_decoder(self, nconv, final_activation=nn.Sigmoid):
        return nn.Sequential(
            *[nn.Sequential(nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=1), nn.ReLU()) for _ in range(4)],
            nn.Upsample(scale_factor=2, mode='bilinear'),

            nn.Conv2d(nconv*4, nconv*2, 3, stride=1, padding=1),
            *[nn.Sequential(nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=1), nn.ReLU()) for _ in range(4)],
            nn.Upsample(scale_factor=2, mode='bilinear'),

            *[nn.Sequential(nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=1), nn.ReLU()) for _ in range(4)],
            nn.Upsample(scale_factor=2, mode='bilinear'),

            nn.Conv2d(nconv*2, 1, 3, stride=1, padding=1),
            final_activation()
        )
    
    def forward(self, x):
        x1 = self.encoder(x)
        amp = self.decoder1(x1)
        ph = self.decoder2(x1) * torch.pi

        return amp, ph


def benchmark_model(model, device='cuda', bs=64, warmup=5, iters=10):
    model = model.to(device)
    model.train()  # Ensure model is in training mode
    
    # Create dummy input and target
    input_tensor = torch.randn(bs, 1, 64, 64, device=device, requires_grad=True)
    target = torch.randn(bs, 1, 64, 64, device=device)
    
    # for memory statistics
    mem_cuda = []

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Warm-up iterations (not timed)
    for _ in range(warmup):
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = criterion(output[0], target)
        loss.backward()
        optimizer.step()
    
    # Benchmark iterations
    torch.cuda.synchronize()

    f_passtime = 0
    b_passtime = 0
    for _ in range(iters):
        start = time.time()
        optimizer.zero_grad()
        output = model(input_tensor)
        torch.cuda.synchronize()
        f_passtime += time.time() - start
        loss = criterion(output[0], target) + criterion(output[1], target)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        b_passtime += time.time() - start
        # store in MiB
        mem_cuda.append(torch.cuda.memory_reserved()>>20)

    # get the max as that will be the needed memory
    # inference throughput, training throughput, training memory, forward pass per unit data, backward pass per unit data
    return (bs*iters)/f_passtime, (bs*iters)/(f_passtime + b_passtime), max(mem_cuda), f_passtime/(bs*iters), b_passtime/(bs*iters)


def get_model(type_name:str) -> nn.Module:
    if type_name == "1.25M":
        return recon_model()
    elif type_name == "5M":
        return ReconModel5M()
    elif type_name == "10M":
        return ReconModel10M()
    elif type_name == "20M":
        return ReconModel20M()
    elif type_name == "100M":
        return ReconModel100M()
    elif type_name == "200M":
        return ReconModel200M()

    return None


if __name__=="__main__":
    model = recon_model()
    total_params = sum(param.numel() for param in model.parameters())
    print(f"Total parameters: {total_params}")
    print(benchmark_model(model))
    del model
    torch.cuda.empty_cache()

    model = ReconModel5M()
    total_params = sum(param.numel() for param in model.parameters())
    print(f"Total parameters: {total_params}")
    print(benchmark_model(model))
    del model
    torch.cuda.empty_cache()

    model = ReconModel10M()
    total_params = sum(param.numel() for param in model.parameters())
    print(f"Total parameters: {total_params}")
    print(benchmark_model(model))
    del model
    torch.cuda.empty_cache()

    model = ReconModel20M()
    total_params = sum(param.numel() for param in model.parameters())
    print(f"Total parameters: {total_params}")
    print(benchmark_model(model))
    del model
    torch.cuda.empty_cache()

    model = ReconModel100M()
    total_params = sum(param.numel() for param in model.parameters())
    print(f"Total parameters: {total_params}")
    print(benchmark_model(model))
    del model
    torch.cuda.empty_cache()

    model = ReconModel200M()
    total_params = sum(param.numel() for param in model.parameters())
    print(f"Total parameters: {total_params}")
    print(benchmark_model(model))
    del model
    torch.cuda.empty_cache()


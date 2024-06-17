import torch.nn as nn

# Generador de red
class Generador(nn.Module):
  def __init__(self, args):
    super(Generador, self).__init__()
    self.ngpu = args.ngpu
    self.main = nn.Sequential(
        # Entrada Z; Primera capa de transposición
        nn.ConvTranspose2d(args.nz, args.ngf * 8, 4, 1, 0, bias=False),
        nn.BatchNorm2d(args.ngf * 8),
        nn.ReLU(True),
        
        # Segunda capa de transposición
        nn.ConvTranspose2d(args.ngf * 8, args.ngf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(args.ngf*4),
        nn.ReLU(True),
        
        # Tercera capa de transposición
        nn.ConvTranspose2d(args.ngf*4, args.ngf*2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(args.ngf*2),
        nn.ReLU(True),
        
        # Cuarta capa de transposición
        nn.ConvTranspose2d(args.ngf * 2, args.ngf, 4, 2, 1, bias=False),
        nn.BatchNorm2d(args.ngf),
        nn.ReLU(True),
        
        # Salida nc * 32 * 32
        nn.ConvTranspose2d(args.ngf, args.nc, 4, 2, 3, bias=False),
        nn.Tanh()
    )
    
  def forward(self, input):
    return self.main(input)
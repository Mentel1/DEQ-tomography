import torch
import torch.nn as nn
import torch.optim as optim

from loader.DataLoaders import Tomography
from solver.anderson import anderson
from models.ResNet import ResNetLayer
from models.DEQ import DEQFixedPoint
from trainer.kolter import epoch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

train_loader = Tomography('Abdomen','Training')
test_loader = Tomography('Abdomen','Test')

# f = ResNetLayer(64,128)
# deq = DEQFixedPoint(f, anderson, tol=1e-4, max_iter=100, beta=2.0)
# X = torch.randn(10,64,32,32)
# out = deq(X)
# (out*torch.randn_like(out)).sum().backward()

# chan = 1
# f = ResNetLayer(chan, 64, kernel_size=3)
# model = nn.Sequential(nn.Conv2d(1,chan, kernel_size=3, bias=True, padding=1),
#                       nn.BatchNorm2d(chan),
#                       DEQFixedPoint(f, anderson, tol=1e-2, max_iter=25, m=5),
#                       nn.BatchNorm2d(chan),
#                       nn.AvgPool2d(8,8),
#                       nn.Flatten(),
#                       nn.Linear(chan*4*4,10)).to(device)

# opt = optim.Adam(model.parameters(), lr=1e-3)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, max_epochs*len(train_loader), eta_min=1e-6)

# max_epochs = 50

# for i in range(50):
#     print(epoch(train_loader, model, device, opt, scheduler))
#     print(epoch(test_loader, model, device))
import torch
import torch.nn as nn
import torch.optim as optim

from loader.DataLoaders import Tomography
from printer.image_printer import print_grayscale
from printer.progress import print_progress
from solver.anderson import anderson
from models.ResNet import ResNetLayer
from models.DEQ import DEQFixedPoint
from trainer.kolter import epoch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
batch_size = 5

train_dataset = Tomography('./data/')
test_dataset = Tomography('./data/','test')

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

_, y = next(iter(train_loader))

# print_grayscale(train_loader, 3)
# print_grayscale(train_loader, 3, input_image=False)

target_height, target_width = y.shape[-2:]
chan = 1 * 8 # Change first int only

f = ResNetLayer(chan, 64, kernel_size=3)
model = nn.Sequential(nn.Conv2d(1, chan, kernel_size=3, bias=True, padding=1),
                      nn.BatchNorm2d(chan),
                      DEQFixedPoint(f, anderson, tol=1e-2, max_iter=25, m=5),
                      nn.BatchNorm2d(chan),
                      nn.Upsample(size=[target_height + 4, target_width + 4]),
                      nn.Conv2d(chan, 1, kernel_size=5)
                      ).to(device)

opt = optim.Adam(model.parameters(), lr=1e-3)

MAX_EPOCH = 50
scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, MAX_EPOCH * len(train_loader), eta_min=1e-6)
num_epochs = 2

for i in range(num_epochs):
    epoch_train_loss = epoch(train_loader, model, device, opt, scheduler)
    epoch_test_loss = epoch(test_loader, model, device)
    print_progress(i, num_epochs, epoch_test_loss, training=False)
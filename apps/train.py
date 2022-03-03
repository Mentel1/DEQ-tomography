import torch
import torch.optim as optim

from loader.DataLoaders import Tomography
from printer.progress import print_progress
from solver.anderson import anderson
from models.ForwardBackward import ForwardBackwardLayer
from models.DEQ import ReconstructionDEQ
from trainer.image_reconstruction import reconstruction_epoch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
batch_size = 5

# Data loading
train_dataset = Tomography('./data/')
test_dataset = Tomography('./data/','test')
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

false_operator = torch.randn((110, 512))
f = ForwardBackwardLayer(false_operator, .01)
model = ReconstructionDEQ(f, anderson, tol=1e-2, max_iter=25, m=5).to(device)
opt = optim.Adam(model.parameters(), lr=1e-3)

MAX_EPOCH = 50
scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, MAX_EPOCH * len(train_loader), eta_min=1e-6)
num_epochs = 2

for i in range(num_epochs):
    epoch_train_loss = reconstruction_epoch(train_loader, model, false_operator, device, opt, scheduler)
    epoch_test_loss = reconstruction_epoch(test_loader, model, false_operator, device)
    print_progress(i, num_epochs, epoch_test_loss, training=False)
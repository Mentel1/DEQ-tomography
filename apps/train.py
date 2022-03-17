import torch
import torch.optim as optim

import sys
sys.path.insert(0, 'C:/Users/Lila/Documents/Centrale/3A/Projet Deep Equilibrium/Git projet/DEQ-tomography')

from loader.DataLoaders import Tomography
from printer.progress import print_progress
from solver.anderson import anderson
from models.ForwardBackward import ForwardBackwardLayer
from models.DEQ import ReconstructionDEQ
from trainer.image_reconstruction import reconstruction_epoch

from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

batch_size = 5
channels = 1
tomosipo = False # Change this to False if no tomosipo environment available

# Data loading
train_dataset = Tomography('./data/')
test_dataset = Tomography('./data/','test')
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

if tomosipo:
    from operators.radon import RadonTransform
    operator = RadonTransform(channels=channels)
else:
    operator = torch.randn((110, 512))

f = ForwardBackwardLayer(operator, .01, tomosipo=tomosipo)
# Play with anderson's hyperparameter in case solver raises singular matrix errors
model = ReconstructionDEQ(f, anderson, tol=1e-2, max_iter=25, beta=1., lam=1e-2).to(device)
opt = optim.Adam(model.parameters(), lr=1e-3)
MAX_EPOCH = 50
scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, MAX_EPOCH * len(train_loader), eta_min=1e-6)
num_epochs = 15

writer = SummaryWriter()

for i in range(num_epochs):

    epoch_train_loss = reconstruction_epoch(train_loader, model, operator, device, opt, scheduler)
    epoch_test_loss = reconstruction_epoch(test_loader, model, operator, device)
    writer.add_scalar("Loss/test", epoch_test_loss, i)
    writer.add_scalar("Loss/train", epoch_train_loss, i)
    writer.add_scalars(f'Loss/both', {'test': epoch_test_loss,'train': epoch_train_loss,}, i)
    print_progress(i, num_epochs, epoch_test_loss, training=False)

writer.flush()
# To see results, run 'tensorboard --logdir=runs' and go to the provided url (or to http://localhost:6006/)
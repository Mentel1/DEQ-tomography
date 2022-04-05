import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniUNet(nn.Module):
    '''
    Some sort of Unet but not really
    '''
    def __init__(self, input_size=(512, 512)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.upsample = nn.Upsample(size=list(input_size))
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)

    def forward(self, input_batch):
        """
        Forward pass
        """
        output = F.relu(self.conv1(input_batch))
        output = F.relu(self.conv2(output))
        output = self.max_pool1(output)
        output = F.relu(self.conv3(output))
        output = F.relu(self.conv4(output))
        output = self.upsample(output)
        output = F.relu(self.conv5(output))
        return output

class ForwardBackwardLayer(nn.Module):
    """
    The res net layer
    """
    def __init__(self, operator, learning_rate, input_size=(512, 512), regul_block=MiniUNet, tomosipo=False):
        super().__init__()
        self.tomosipo_ = tomosipo
        self.operator_ = operator
        self.learning_rate_ = learning_rate
        self.prox = regul_block(input_size=input_size)

    def forward(self, input_batch, targets):
        """
        Forward pass
        """
        if self.tomosipo_:
            output = input_batch + self.learning_rate_ * self.operator_.T(targets - self.operator_.torch_operator(input_batch))
        else:
            output = input_batch + self.learning_rate_ * self.operator_.T @ (targets - self.operator_ @ input_batch)

        output = self.prox(output)
        return output

if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size = 3
    channels=1
    tomosipo = True # Change this to False if tomosipo environment not activated

    x, y = torch.randn((batch_size, channels, 512, 512)), torch.randn((batch_size, channels, 110, 512))

    if tomosipo:
        from operators.radon import RadonTransform
        operator = RadonTransform(channels=channels)
    else:
        operator = torch.randn((110, 512))
        
    f = ForwardBackwardLayer(operator, .01, tomosipo=tomosipo)
    x_inf = f(x, y)

    if tomosipo:
        loss = nn.MSELoss()(operator(x_inf), y)
    else:
        loss = nn.MSELoss()(operator @ x_inf, y)

    loss.backward()
    print(loss.item())

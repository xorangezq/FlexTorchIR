
import torch

original = {
    'just.relu': torch.relu,
    'functional.relu': torch.nn.functional.relu,
}
def hooked_just_relu(*args, **kwargs):
    output = original['just.relu'](*args, **kwargs)
    print('hooked just.relu')
    return output
def hooked_functional_relu(*args, **kwargs):
    output = original['functional.relu'](*args, **kwargs)
    print('hooked functional.relu')
    return output
torch.relu = hooked_just_relu
torch.nn.functional.relu = hooked_functional_relu

class Demo(torch.nn.Module):
    def __init__(self):
        super(Demo, self).__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, 3)

    def forward(self, x):
        y = self.conv(x)
        y = torch.nn.functional.relu(y)
        return y

demoModel = Demo()
y = demoModel(torch.randn(1, 3, 224, 224))

'''
Output:
hooked just.relu
hooked functional.relu
'''

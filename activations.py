from namedtensor import ntorch

class ReLU(ntorch.nn.Module):
    def __init__(self):
        super(ReLU, self).__init__()
        
    def forward(self, x):        
        return x.relu()
    
class Identity(ntorch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):        
        return x
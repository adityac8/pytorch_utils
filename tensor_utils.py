import torch

# Channel Shuffle
def channel_shuffle(x):
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % 4 == 0)
    return torch.cat((x[:,::2,:,:],x[:,1::2,:,:]),1)

import torch

# Channel Shuffle
def channel_shuffle(x):
    batchsize, num_channels, height, width = x.size()
    assert (num_channels % 4 == 0)
    return torch.cat((x[:,::2,:,:],x[:,1::2,:,:]),1)

# Channel Unshuffle
def channel_unshuffle(x):
    batchsize, num_channels, height, width = x.size()
    assert (num_channels % 4 == 0)
    return x.view(batchsize, 2, num_channels//2, height, width).permute(0,2,1,3,4).reshape(batchsize, num_channels, height, width)

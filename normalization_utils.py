import torch
import torch.nn as nn
import numbers
import numpy as np

class InstanceNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(InstanceNorm, self).__init__()

        self.weight = nn.Parameter(torch.ones((1,normalized_shape,1,1)))
        self.bias = nn.Parameter(torch.zeros((1,normalized_shape,1,1)))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = torch.mean(x, axis=(2,3), keepdim=True)
        sigma = torch.var(x, axis=(2,3), keepdim=True, unbiased=False)
        return (((x - mu) / torch.sqrt(sigma+1e-5)) * self.weight) + self.bias

class InstanceNormBiasFree(nn.Module):
    def __init__(self, normalized_shape):
        super(InstanceNormBiasFree, self).__init__()

        self.weight = nn.Parameter(torch.ones((1,normalized_shape,1,1)))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = torch.var(x, axis=(2,3), keepdim=True, unbiased=False)
        return ((x / torch.sqrt(sigma+1e-5)) * self.weight)

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNormBiasFree(nn.Module):
    def __init__(self, normalized_shape):
        super(LayerNormBiasFree, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight
    
    
class BatchNormBiasFree(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, use_bias = False, affine=True):
        super(BatchNormBiasFree, self).__init__(num_features, eps, momentum)

        self.use_bias = use_bias;

    def forward(self, x):
        self._check_input_dim(x)
        y = x.transpose(0,1)
        return_shape = y.shape
        y = y.contiguous().view(x.size(1), -1)
        if self.use_bias:
            mu = y.mean(dim=1)
        sigma2 = y.var(dim=1)

        if self.training is not True:
            if self.use_bias:        
                y = y - self.running_mean.view(-1, 1)
            y = y / ( self.running_var.view(-1, 1)**0.5 + self.eps)
        else:
            if self.track_running_stats is True:
                with torch.no_grad():
                    if self.use_bias:
                        self.running_mean = (1-self.momentum)*self.running_mean + self.momentum * mu
                    self.running_var = (1-self.momentum)*self.running_var + self.momentum * sigma2
            if self.use_bias:
                y = y - mu.view(-1,1)
            y = y / (sigma2.view(-1,1)**.5 + self.eps)

        if self.affine:
            y = self.weight.view(-1, 1) * y;
            if self.use_bias:
                y += self.bias.view(-1, 1)

        return y.view(return_shape).transpose(0,1)

if __name__ == "__main__":
    bias = InstanceNorm(5)
    nobias = InstanceNormBiasFree(5)
    for _ in range(5):
        temp_inp = torch.randn(100, 5, 128, 128)*10 + 10;
        bias_out = bias(temp_inp);
        print('bias')
        print('variance %f, mean %f'%(torch.var(bias_out), torch.mean(bias_out)));
    
        bf_out = nobias(temp_inp);
        print('nobias')
        print('variance %f, mean %f'%(torch.var(bf_out), torch.mean(bf_out)))
        print(np.allclose(bf_out.detach().numpy(), bias_out.detach().numpy(), atol=1e-4))
        print('\n\n\n\n')

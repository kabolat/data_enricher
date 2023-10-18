import torch

def to_sigma(sigmatilde):
    return torch.log2(torch.pow(2,sigmatilde)+1)

def from_sigma(sigma):
    return torch.log2(torch.pow(2,sigma)-1)

def to_pi(pitilde):
    return torch.softmax(pitilde,dim=1)

def from_pi(pi):
    return torch.log(pi)-torch.log(pi).mean(dim=1)

def gaussian_kernel_exponent(x, mu, sigma):
    D = mu.shape[-1]
    x_mu = x.unsqueeze(1)-mu
    dist = torch.norm(x_mu,dim=-1)/sigma.squeeze()
    log_det = 2*D*torch.log(sigma).squeeze()
    return -0.5*(dist**2 + log_det + D*2*torch.tensor(torch.pi).log())

class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super(IndexedDataset,self).__init__()
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx], idx

    def __len__(self):
        return len(self.data)
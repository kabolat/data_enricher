import torch
from scipy.stats import ks_2samp as KS
from scipy.stats import cramervonmises_2samp as CVM

def MMDTest(base_samples, test_samples, alphas=[0.5, 1.0, 2.0, 5.0, 10.0]):
    
    n_base, n_test = base_samples.size(0), test_samples.size(0)
    
    a00 = 1.0/(n_base*(n_base - 1))
    a11 = 1.0/(n_test*(n_test - 1))
    a01 = -1.0/(n_base*n_test)

    sample_12 = torch.cat((base_samples, test_samples), 0)
    distances = torch.norm((sample_12.unsqueeze(1)-sample_12),dim=-1)

    kernels = None
    for alpha in alphas:
        kernels_a = torch.exp(-alpha*distances**2)
        if kernels is None: kernels = kernels_a
        else: kernels = kernels + kernels_a

    k_1 = kernels[:n_base, :n_base]
    k_2 = kernels[n_base:, n_base:]
    k_12 = kernels[:n_base, n_base:]

    score = (2*a01*k_12.sum() + 
            a00*(k_1.sum()-torch.trace(k_1)) + 
            a11*(k_2.sum()-torch.trace(k_2)))
    
    return score

def EnergyTest(base_samples, test_samples):

    n_base, n_test = base_samples.size(0), test_samples.size(0)

    a00 = -1.0/(n_base**2)
    a11 = -1.0/(n_test*2)
    a01 = 1.0/(n_base*n_test)

    sample_12 = torch.cat((base_samples, test_samples), 0)
    distances = torch.norm((sample_12.unsqueeze(1)-sample_12),dim=-1)

    d_1 = distances[:n_base, :n_base].sum()
    d_2 = distances[-n_test:, -n_test:].sum()
    d_12 = distances[:n_base, -n_test:].sum()

    score = 2*a01*d_12 + a00*d_1 + a11*d_2

    return score

def sample_comparison(model_samples, train_samples, test_samples, test="mmd", subsample_ratio=0.4, mc_runs=1000):
    assert model_samples.shape[0]==train_samples.shape[0]

    if test=="mmd": test_func = MMDTest
    elif test=="energy": test_func = EnergyTest
    else: raise NotImplementedError("Test not implemented.")

    num_samples = test_samples.shape[0]
    n = int(num_samples*subsample_ratio)

    base_scores, model_scores = torch.zeros(mc_runs), torch.zeros(mc_runs)

    for i in range(mc_runs):
        test_subsamples = test_samples[torch.randperm(num_samples)[:n]]
        base_scores[i] = test_func(test_subsamples,train_samples[torch.randperm(train_samples.shape[0])[:n]])
        model_scores[i] = test_func(test_subsamples,model_samples[torch.randperm(model_samples.shape[0])[:n]])
    return model_scores, base_scores

def model_comparison(model_scores, base_scores, test="ks"):
    if test=="ks": test_func = lambda x,y: KS(x,y).statistic
    elif test=="cvm": test_func = lambda x,y: CVM(x,y).statistic
    elif test=="mean": test_func = lambda x,y: y.mean()-x.mean()
    else: raise NotImplementedError("Test not implemented.")
    return test_func(base_scores, model_scores)

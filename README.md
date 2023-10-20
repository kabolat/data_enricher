
# Data Enricher

Data Enricher is a software tool that aims to enrich data for machine learning applications. The main motivations for data enrichment in this project are:

- **Data Scarcity**: The amount of data available for machine learning applications can be limited.
- **Data Missingness**: The data available for machine learning applications can have missing values.

The fundamental methodology of the Data Enricher is to model the data probabilistically and use this model to cope with these problems. The module contains various probabilistic models equipped with sampling and imputation capabilities and performance testing tools for these models.

- [Models](#models)
  - [KDE-based Models](#kde-based-models)
    - [Vanilla KDE](#vanilla-kde)
    - [Adaptive KDE](#adaptive-kde)
    - [$\pi$-KDE](#pi-kde)
  - [VAE-based Models](#vae-based-models)
    - [Vanilla VAE](#vanilla-vae)
    - [Conditional VAE (CVAE)](#conditional-vae-cvae)
    - [$\beta$-VAE](#beta-vae)
    - [Conditional $\beta$-VAE (C-$\beta$-VAE)](#conditional-beta-vae-c-beta-vae)
- [Testers](#testers)
  - [Sample Comparison](#sample-comparison)
  - [Model Comparison](#model-comparison)
- [Installation](#installation)
- [References](#references)
- [Acknowledgements](#acknowledgements)

## Models

Data Enricher contains two main types of probabilistic models:

- **Kernel Density Estimation (KDE)**, and
- **Variational Autoencoder (VAE)** based models.

### KDE-based Models

All the KDE-based models incorporate isotropic Gaussian kernels. All the model classes share the following methods:

- `sample`: Samples from the model.
- `log_likelihood`: Computes the log-likelihood of the given data.
- `impute`: Imputes the missing values of the given data.
- `get_params`: Returns the parameters of the model.

*The detailed mathematical description of the KDE models can be found in (Bölat, 2023).*

#### Vanilla KDE

Vanilla KDE is a simple KDE model that uses the same kernel bandwidth for each data point. If not given, the bandwidth of the kernel (`sigma`) is estimated using the sum of standard deviations of the features divided by the number of data points.

#### Adaptive KDE

Adaptive KDE (A-KDE) is a KDE model that uses different kernel bandwidths (`sigma`) for each data point. These bandwidths are trained by maximizing the **leave-one-out maximum log-likelihood (LOO-MLL)** objective. Using LOO-MLL guarantees that the training of the model does not result in singular solutions, unlike the regular MLL objective. This can be observed by setting the argument `leave_one_out=False` in the `train` method of the model. 

The following methods are available for the optimization:

- **Modified Expectation Maximization (EM)**: Default method. The bandwidths are optimized using the EM algorithm adapted to LOO-MLL (Bölat, 2023). Requires no additional arguments.
- **Adam**: The bandwidths are optimized using the Adam optimizer (Kingma, 2014). Requires the following additional arguments:
  - `learning_rate`: Learning rate of the Adam optimizer.
  - `batch_size`: Batch size for the training.

#### $\pi$-KDE

$\pi$-KDE (Bölat, 2023) is the generalized version of the A-KDE where each kernel has a learnable weight parameter (`pi`). The training of the model is the same as A-KDE.

### VAE-based Models

VAEs (Kingma, 2013) are neural network-based (deep) latent variable models that learn the distribution of the data by maximizing the evidence lower bound (ELBO) objective. 

All the VAE-based models share the following initialization arguments:

- `input_dim`: Dimension of the input data.
- `latent_dim`: Dimension of the latent space.
- `num_neurons`: Number of neurons in the hidden layers of the encoder and decoder networks.
- `num_layers`: Number of hidden layers in the encoder and decoder networks.
- `learn_decoder_sigma`: If `True`, the standard deviation of the decoder network is learned. Otherwise, it is fixed to 1.

Additionally, all the VAE-based models share the following training arguments:

- `x`: Training data.
- `epochs`: Number of epochs for the training.
- `batch_size`: Batch size for the training.
- `learning_rate`: Learning rate of the Adam optimizer.
- `mc_samples`: Number of Monte Carlo samples taken from the approximate posterior on the forward pass of the training.

Lastly, all the VAE-based models share the following methods:

- `sample`: Samples from the model.
- `impute`: Imputes the missing values of the given data.

In this tool, there is one VAE model class (`VAE`), which incorporates diagonal Gaussian distributions for the inference, the posterior and the prior distributions. The following model variations can be initiated via the given argumentation under them.

#### Vanilla VAE

The default VAE model. The following arguments can be given to the model:

- `VAE(cond_dim=0)` (Default)
- `.train(cond=None, beta=1.0)` (Default)

#### Conditional VAE (CVAE)

CVAE is a VAE model that incorporates conditional information to the model. The following arguments can be given to the model:

- `VAE(cond_dim=cond_dim)`
- `.train(cond=cond, beta=1.0)`

#### $\beta$-VAE

$\beta$-VAE (Higgins, 2016) is a VAE model that incorporates a hyperparameter $\beta>0$ to the ELBO objective. The following arguments can be given to the model:

- `VAE(cond_dim=0)`
- `.train(cond=None, beta=beta)`

#### Conditional $\beta$-VAE (C-$\beta$-VAE)

C-$\beta$-VAE is a VAE model that incorporates conditional information and a hyperparameter $\beta>0$ to the ELBO objective. The following arguments can be given to the model:

- `VAE(cond_dim=cond_dim)`
- `.train(cond=cond, beta=beta)`


## Testers

Data Enricher contains performance testing tools for the models under the `comparison.py` module. These tools mainly consist of two-sample tests which serve two purposes:

- **Sample Comparison Tests**: Compares two sets of multi-dimensional samples.
- **Model Comparison Tests**: Compares two sets of one-dimensional random statistics.

*The detailed description of the testing procedure can be found in (Bölat, 2023).*

### Sample Comparison

Sample comparison tests are used to compare generated data with the original data. The method `sample_comparison` in the `comparison.py` module can be used to perform these tests, and it requires `model_samples`, `train_samples`, and `test_samples`. The `model_samples` are the samples generated by the model, and the `train_samples` and `test_samples` are the training and test samples of the original data, respectively.

The method performs the selected test (see below) between the `test_samples` and the `train_samples` and `model_samples`, separately. The results of the tests are returned as a dictionary containing `model_scores` and `base_scores`. These scores are calculated by random subsampling controlled by `subsample_ratio` and `mc_runs`. The `model_scores` are the scores of the model subsamples, and the `base_scores` are the scores of the training samples.

- **Maximum Mean Discrepancy (MMD)** (Gretton, 2012)
  - `sample_comparison(test='mmd')` (Default)

- **Energy** (Székely, 2013)
  - `sample_comparison(test='energy')`

### Model Comparison

Model comparison tests are used to compare the model statistics with the statistics of the original data. The method `model_comparison` in the `comparison.py` module can be used to perform these tests, and it requires `model_scores` and `base_scores` calculated by the `sample_comparison` method. The method performs the selected test (see below) between the `model_scores` and the `base_scores`, which are one-dimensional random statistics of the model and the original data, respectively.

- **Kolmogorov-Smirnov (KS)** (Kolmogorov, 1933)
  - `model_comparison(test='ks')` (Default)

- **Cramer-von Mises (CvM)** (Anderson, 1962) 
  - `model_comparison(test='cvm')`

- **Mean Difference**
  - `model_comparison(test='mean')`


## Installation

Data Enricher can be installed via `pip`:

```bash
pip install data_enricher
```



## References
- Anderson, T.W., 1962. On the distribution of the two-sample Cramer-von Mises criterion. The Annals of Mathematical Statistics, pp.1148-1159.
- AN, K., 1933. Sulla determinazione empirica di una legge didistribuzione. Giorn Dell'inst Ital Degli Att, 4, pp.89-91.
- Bölat, K., Tindemans, S.H. and Palensky, P., 2023. Stable Training of Probabilistic Models Using the Leave-One-Out Maximum Log-Likelihood Objective. arXiv preprint arXiv:2310.03556.
- Gretton, A., Borgwardt, K.M., Rasch, M.J., Schölkopf, B. and Smola, A., 2012. A kernel two-sample test. The Journal of Machine Learning Research, 13(1), pp.723-773.
- Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick, M., Mohamed, S. and Lerchner, A., 2016, November. beta-vae: Learning basic visual concepts with a constrained variational framework. In International conference on learning representations.
- Kingma, D.P. and Welling, M., 2013. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114.
- Kingma, D.P. and Ba, J., 2014. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.
- Székely, G.J. and Rizzo, M.L., 2013. Energy statistics: A class of statistics based on distances. Journal of statistical planning and inference, 143(8), pp.1249-1272.

## Acknowledgements
This software is developed under the H2020-MSCA-ITN [Innovative Tools for Cyber-Physical Systems (InnoCyPES)](https://innocypes.eu/) project. The project is funded by the European Union's Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No 956433.

<img src="https://upload.wikimedia.org/wikipedia/commons/b/b7/Flag_of_Europe.svg" alt="drawing" width="150"/> 

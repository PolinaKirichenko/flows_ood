import torch
from torch import distributions, nn
from torch.distributions import constraints
from torch.distributions.utils import _standard_normal
import torch.nn.functional as F
import numpy as np
import math

class SSLGaussMixture(torch.distributions.Distribution):

    def __init__(self, means, inv_cov_stds=None, device=None):
        self.n_components, self.d = means.shape
        self.means = means

        if inv_cov_stds is None:
            self.inv_cov_stds = math.log(math.exp(1.0) - 1.0) * torch.ones((len(means)), device=device)
        else:
            self.inv_cov_stds = inv_cov_stds

        self.weights = torch.ones((len(means)), device=device)
        self.device = device

    @property
    def gaussians(self):
        gaussians = [distributions.MultivariateNormal(mean, F.softplus(inv_std)**2 * torch.eye(self.d).to(self.device))
                          for mean, inv_std in zip(self.means, self.inv_cov_stds)]
        return gaussians


    def parameters(self):
       return [self.means, self.inv_cov_std, self.weights]
        
    def sample(self, sample_shape, gaussian_id=None):
        if gaussian_id is not None:
            g = self.gaussians[gaussian_id]
            samples = g.sample(sample_shape)
        else:
            n_samples = sample_shape[0]
            idx = np.random.choice(self.n_components, size=(n_samples, 1), p=F.softmax(self.weights))
            all_samples = [g.sample(sample_shape) for g in self.gaussians]
            samples = all_samples[0]
            for i in range(self.n_components):
                mask = np.where(idx == i)
                samples[mask] = all_samples[i][mask]
        return samples
        
    def log_prob(self, x, y=None, label_weight=1.):
        all_log_probs = torch.cat([g.log_prob(x)[:, None] for g in self.gaussians], dim=1)
        mixture_log_probs = torch.logsumexp(all_log_probs + torch.log(F.softmax(self.weights)), dim=1)
        if y is not None:
            log_probs = torch.zeros_like(mixture_log_probs)
            mask = (y == -1)
            log_probs[mask] += mixture_log_probs[mask]
            for i in range(self.n_components):
                mask = (y == i)
                log_probs[mask] += all_log_probs[:, i][mask] * label_weight
            return log_probs
        else:
            return mixture_log_probs

    def class_logits(self, x):
        log_probs = torch.cat([g.log_prob(x)[:, None] for g in self.gaussians], dim=1)
        log_probs_weighted = log_probs + torch.log(F.softmax(self.weights))
        return log_probs_weighted

    def classify(self, x):
        log_probs = self.class_logits(x)
        return torch.argmax(log_probs, dim=1)

    def class_probs(self, x):
        log_probs = self.class_logits(x)
        return F.softmax(log_probs, dim=1)


class SSLGaussMixtureClassifier(SSLGaussMixture):
    
    def __init__(self, means, cov_std=1., device=None):
        super().__init__(means, cov_std, device)
        self.classifier = nn.Sequential(nn.Linear(self.d, self.n_components))

    def parameters(self):
       return self.classifier.parameters() 

    def forward(self, x):
        return self.classifier.forward(x)

    def log_prob(self, x, y, label_weight=1.):
        all_probs = [torch.exp(g.log_prob(x)) for g in self.gaussians]
        probs = sum(all_probs) / self.n_components
        x_logprobs = torch.log(probs)

        mask = (y != -1)
        labeled_x, labeled_y = x[mask], y[mask].long()
        preds = self.forward(labeled_x)
        y_logprobs = F.cross_entropy(preds, labeled_y)

        return x_logprobs - y_logprobs



class MultivariateExpDistance(torch.distributions.Distribution):
    arg_constraints = {'loc': constraints.real_vector,}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, scale=1., k=None, validate_args=None):
        if loc.dim() < 1:
            raise ValueError("loc must be at least one-dimensional.")
        if k is None:
            self.k = int(np.sqrt(loc.dim()))
        else:
            self.k = k

        self.loc = loc
        self.scale = scale

        # event_shape is dimensionality of rv
        batch_shape, event_shape = self.loc.shape[:-1], self.loc.shape[-1:]
        super(MultivariateExpDistance, self).__init__(batch_shape, event_shape, validate_args=validate_args)


    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(MultivariateDistance, _instance)
        batch_shape = torch.Size(batch_shape)
        loc_shape = batch_shape + self.event_shape
        cov_shape = batch_shape + self.event_shape + self.event_shape
        new.loc = self.loc.expand(loc_shape)
        new._unbroadcasted_scale_tril = self._unbroadcasted_scale_tril
        super(MultivariateExpDistance, new).__init__(batch_shape,
                                                  self.event_shape,
                                                  validate_args=False)
        new._validate_args = self._validate_args
        return new


    @property
    def mean(self):
        return self.loc


    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        dir_var = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        dir_var /= dir_var.norm(p=2, dim=-1, keepdim=True)
        norms = torch._standard_gamma(
            self.k * torch.ones(size=shape[:-1], dtype=self.loc.dtype, device=self.loc.device)
        ).reshape((-1, 1)) * self.scale
        return self.loc + norms * dir_var


    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        D = self.loc.shape[-1]
        diff = value - self.loc
        locdist = torch.norm(diff, dim=-1)
        return - locdist / self.scale + (self.k - D) * torch.log(locdist) \
            - self.k * torch.log(torch.tensor([self.scale])) - math.log(2.) - D/2 * math.log(math.pi) \
            + torch.lgamma(torch.tensor([D/2], dtype=torch.float32, device=self.loc.device)) \
            - torch.lgamma(torch.tensor([self.k], dtype=torch.float32, device=self.loc.device))



# TODO(Polina): make a general class SSLMixture to handle any base distriubtion in the mixture
class SSLExpDistanceMixture(torch.distributions.Distribution):

    def __init__(self, means, device=None):
        self.n_components, self.d = means.shape
        self.means = means

        self.weights = torch.ones((len(means)), device=device)
        self.device = device

    @property
    def components(self):
        components = [MultivariateExpDistance(mean) for mean in self.means]
        return components


    def parameters(self):
       return [self.means, self.weights]
        
    def sample(self, sample_shape, gaussian_id=None):
        if gaussian_id is not None:
            c = self.components[gaussian_id]
            samples = c.sample(sample_shape)
        else:
            n_samples = sample_shape[0]
            idx = np.random.choice(self.n_components, size=(n_samples, 1), p=F.softmax(self.weights))
            all_samples = [c.sample(sample_shape) for c in self.components]
            samples = all_samples[0]
            for i in range(self.n_components):
                mask = np.where(idx == i)
                samples[mask] = all_samples[i][mask]
        return samples

    def log_prob(self, x, y=None, label_weight=1.):
        all_log_probs = torch.cat([c.log_prob(x)[:, None] for c in self.components], dim=1)
        mixture_log_probs = torch.logsumexp(all_log_probs + torch.log(F.softmax(self.weights)), dim=1)
        if y is not None:
            log_probs = torch.zeros_like(mixture_log_probs)
            mask = (y == -1)
            log_probs[mask] += mixture_log_probs[mask]
            for i in range(self.n_components):
                mask = (y == i)
                log_probs[mask] += all_log_probs[:, i][mask] * label_weight
            return log_probs
        else:
            return mixture_log_probs

    def class_logits(self, x):
        log_probs = torch.cat([c.log_prob(x)[:, None] for c in self.components], dim=1)
        log_probs_weighted = log_probs + torch.log(F.softmax(self.weights))
        return log_probs_weighted

    def classify(self, x):
        log_probs = self.class_logits(x)
        return torch.argmax(log_probs, dim=1)

    def class_probs(self, x):
        log_probs = self.class_logits(x)
        return F.softmax(log_probs, dim=1)


class PPCA(torch.distributions.Distribution):

    def __init__(self, data_dim, latent_dim, mean=None, P=None, inv_sigma=None, device=None):
        self.D = data_dim
        self.d = latent_dim

        if mean is None:
            self.data_mean = torch.zeros((data_dim,), device=device)
        else:
            self.data_mean = mean.to(device)

        if P is None:
            self.P = torch.zeros((data_dim, latent_dim), device=device)
            self.P[np.arange(self.d), np.arange(self.d)] = 1.
            # self.P[1][0] = 1.
        else:
            self.P = P.to(device)

        if inv_sigma is None:
            self.inv_sigma = math.log(math.exp(0.1) - 1.0) * torch.ones((self.D),
                device=device)
        else:
            self.inv_sigma = math.log(math.exp(inv_sigma) - 1.0) * torch.ones((self.D), device=device)

        self.device = device

    @property
    def mean(self):
        return self.data_mean

    def parameters(self):
       return [self.data_mean, self.P, self.inv_sigma]

    def sample(self, sample_shape):
        n_samples = sample_shape[0]
        latent_noise = distributions.Normal(0, 1).sample((n_samples, self.d)).cuda()
        data_noise = distributions.Normal(0, F.softplus(self.inv_sigma)).sample((n_samples,))

        samples = latent_noise @ self.P.t() + self.data_mean + data_noise
        return samples

    def log_prob(self, x):
        # TODO: pass precision matrix instead
        normal = distributions.MultivariateNormal(
            self.data_mean,
            self.P @ self.P.t() + F.softplus(self.inv_sigma)**2 * torch.eye(self.D).to(self.device))
        return normal.log_prob(x)


class SSLPCAMixture(torch.distributions.Distribution):

    def __init__(self, means, P, cov, device=None):
        self.n_components, self.d = means.shape
        self.D = P.shape[0]
        self.means = means
        self.P = P
        self.cov = cov

        # if inv_sigma is None:
        #     self.inv_sigma = math.log(math.exp(0.1) - 1.0) * torch.ones((len(means)), device=device)
        # else:
        #    self.inv_sigma = math.log(math.exp(inv_sigma) - 1.0) * torch.ones((self.D), device=device)

        self.weights = torch.ones((len(means)), device=device)
        self.device = device

    @property
    def gaussians(self):
        gaussians = [distributions.MultivariateNormal(self.P @ mean, self.cov) for mean in self.means]
        return gaussians

    def parameters(self):
       return [self.means, self.P, self.cov, self.weights]

    def sample(self, sample_shape, gaussian_id=None):
        if gaussian_id is not None:
            g = self.gaussians[gaussian_id]
            samples = g.sample(sample_shape)
        else:
            n_samples = sample_shape[0]
            idx = np.random.choice(self.n_components, size=(n_samples, 1), p=F.softmax(self.weights))
            all_samples = [g.sample(sample_shape) for g in self.gaussians]
            samples = all_samples[0]
            for i in range(self.n_components):
                mask = np.where(idx == i)
                samples[mask] = all_samples[i][mask]
        return samples

    def log_prob(self, x, y=None, label_weight=1.):
        all_log_probs = torch.cat([g.log_prob(x)[:, None] for g in self.gaussians], dim=1)
        mixture_log_probs = torch.logsumexp(all_log_probs + torch.log(F.softmax(self.weights)), dim=1)
        if y is not None:
            log_probs = torch.zeros_like(mixture_log_probs)
            mask = (y == -1)
            log_probs[mask] += mixture_log_probs[mask]
            for i in range(self.n_components):
                mask = (y == i)
                log_probs[mask] += all_log_probs[:, i][mask] * label_weight
            return log_probs
        else:
            return mixture_log_probs

    def class_logits(self, x):
        log_probs = torch.cat([g.log_prob(x)[:, None] for g in self.gaussians], dim=1)
        log_probs_weighted = log_probs + torch.log(F.softmax(self.weights))
        return log_probs_weighted

    def classify(self, x):
        log_probs = self.class_logits(x)
        return torch.argmax(log_probs, dim=1)

    def class_probs(self, x):
        log_probs = self.class_logits(x)
        return F.softmax(log_probs, dim=1)

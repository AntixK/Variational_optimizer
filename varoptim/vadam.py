import math
import torch
from torch.optim.optimizer import Optimizer
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from numpy import asarray

#=================#
# VADAM OPTIMIZER #
#=================#
class VAdam(Optimizer):
    '''
    Implements the Variational ADAM (VAdam) optimizer algorithm

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)

    Reference(s):
        [1] "Fast and Scalable Bayesian Deep Learning by Weight-Perturbation in Adam"
            https://arxiv.org/abs/1806.04854

    '''

    def __init__(self, params, train_batch_size, prior_precision=1.0,
                 init_precision=1.0, lr = 1e-3, betas=(0.9,0.999),
                 eps=1e-9, num_samples=1):
        if not 0.0 <=lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid espilon value: {}".format(eps))
        if not 0.0 <= betas[0] <= 1.0:
            raise ValueError("Invalid  beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] <= 1.0:
            raise ValueError("Invalid  beta parameter at index 0: {}".format(betas[1]))
        if not 0.0 <= prior_precision:
            raise ValueError("Invalid prior precision value: {}".format(prior_precision))
        if not 0.0 <= init_precision:
            raise ValueError("Invalid initial s value: {}".format(init_precision))
        if num_samples < 1:
            raise ValueError("Invalid num_samples parameter: {}".format(num_samples))
        if train_batch_size < 1:
            raise ValueError("Invalid number of training data points: {}".format(train_set_size))

        self.num_samples = num_samples
        self.train_batch_size = train_batch_size
        defaults = dict(lr=lr, betas=betas,eps=eps, prior_precision = prior_precision,
                        init_precision = init_precision)
        super(VAdam,self).__init__(params,defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
	        1) Perturb data with Gaussian noise
	        2) Do backprop
	        3) Get gradients of this perturbed data
        4) Do steps 1-3 for N number of samples
        5) Perform Adam as usual with the added precision factor (Store unperturbed data)
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        if closure is not None:
            loss = closure()
        t = 0
        for group in self.param_groups:
            for p in group['params']:

                original_value = p.detach().clone()
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    #state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.ones_like(p.data) * (group['init_precision'] -
                                                                     group['prior_precision']) / self.train_batch_size


                for s in range(self.num_samples):
                    # A noisy sample
                    raw_noise = torch.normal(mean=torch.zeros_like(p.data), std=1.0)
                    p.data.addcdiv_(1., raw_noise,
                                    torch.sqrt(self.train_batch_size * state['exp_avg_sq'] + group['prior_precision']))
                    loss = None


                    if p.grad is None:
                        continue
                    if s ==0:
                        grad = p.grad.data
                    else:
                        grad += p.grad.data

                grad.div(self.num_samples)

                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                tlambda = group['prior_precision'] / self.train_batch_size

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad + tlambda * original_value)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                denom = exp_avg_sq.sqrt().add(tlambda * math.sqrt(bias_correction2))
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)
                t += 1

        return loss

import math
import torch
from torch.optim.optimizer import Optimizer
from torch.nn.utils import parameters_to_vector, vector_to_parameters

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

    References:
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

    def step(self, closure):
        '''
        Perform a single optimization step
        Arguments:
            closure(callable):A closure that reevaluates the model and
                               returns the loss
        '''

        loss = None
        # Create a place holder for gradients
        grads = []
        grads_sq = []
        for group in self.param_groups:
            for p in group['params']:
                grads.append([])
                grads_sq.append([])

        for s in range(self.num_samples):

            # Initialization
            t = 0
            original_values = {}
            for group in self.param_groups:
                for p in group['params']:

                    original_values.setdefault(t, p.detach().clone())
                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p.data)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.ones_like(p.data) * (
                                    group['init_precision'] - group['prior_precision']) / self.train_batch_size

                    # Sample noise for each parameter
                    raw_noise = torch.normal(mean=torch.zeros_like(p.data), std=1.0)
                    p.data.addcdiv_(1., raw_noise,
                                    torch.sqrt(self.train_batch_size * state['exp_avg_sq'] + group['prior_precision']))

                    t += 1

            # Call the loss function and do BP to compute gradient
            loss = closure()

            # Accumulate Gradients with respect to each parameter for all samples
            for group in self.param_groups:
                for p in group['params']:

                    p.data = original_values[t]

                    if p.grad is None:
                        continue

                    if p.grad.is_sparse:
                        raise RuntimeError('VAdam currently does not support sparse gradients')

                    # Collect all the gradients and their squares
                    if s == 0:
                        grads[t] = p.grad.detach().clone()
                        grads_sq[2] = grads[t]**2
                    else:
                        grads[t] += p.grad.detach().clone()
                        grads_sq[2] += p.grad.detach().clone()**2

                    t += 1

            # The usual Adam optimizer procedure
            t = 0
            for group in self.param_groups:
                for p in group['params']:

                    grad = grads[t].div(self.num_samples)
                    grad_sq = grads_sq[t].dic(self.num_samples)

                    state = self.state[p]

                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    beta1, beta2 = group['betas']

                    lambda_val = group['prior_precision']/self.train_batch_size

                    # Decay the first and second moment running average coefficient
                    exp_avg.mul_(beta1).add_(1 - beta1, grad + lambda_val*original_values[t])
                    exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad_sq)

                    # Bias correction
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']

                    denominator = exp_avg_sq.div(bias_correction2).sqrt()
                    denominator = denominator.add(lambda_val).add_(group['eps'])
                    numerator = exp_avg.div(bias_correction1)

                    # Update parameters
                    p.data.addcdiv_(-group['lr'], numerator, denominator)

                    t += 1
                    state['step'] += 1
        return loss







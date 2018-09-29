import math
import torch
from torch.optim.optimizer import Optimizer
from torch.nn.utils import parameters_to_vector, vector_to_parameters


#################################
## PyTorch Optimizer for Vadam ##
#################################

class VAdam(Optimizer):
    """Implements Vadam algorithm.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        train_set_size (int): number of data points in the full training set
            (objective assumed to be on the form (1/M)*sum(-log p))
        lr (float, optional): learning rate (default: 1e-3)
        beta (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        prior_prec (float, optional): prior precision on parameters
            (default: 1.0)
        init_precision (float, optional): initial precision for variational dist. q
            (default: 1.0)
        num_samples (float, optional): number of MC samples
            (default: 1)
    """

    def __init__(self, params, train_batch_size, lr=1e-3, betas=(0.9, 0.999), prior_precision=1.0, init_precision=1.0,
                 num_samples=1):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= prior_precision:
            raise ValueError("Invalid prior precision value: {}".format(prior_precision))
        if not 0.0 <= init_precision:
            raise ValueError("Invalid initial s value: {}".format(init_precision))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if num_samples < 1:
            raise ValueError("Invalid num_samples parameter: {}".format(num_samples))
        if train_batch_size < 1:
            raise ValueError("Invalid number of training data points: {}".format(train_batch_size))

        self.num_samples = num_samples
        self.train_batch_size = train_batch_size

        defaults = dict(lr=lr, betas=betas, prior_precision=prior_precision, init_precision=init_precision)
        super(VAdam, self).__init__(params, defaults)

    def step(self, closure):
        """Performs a single optimization step.
        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        if closure is None:
            raise RuntimeError(
                'For now, Vadam only supports that the model/loss can be reevaluated inside the step function')

        grads = []
        grads2 = []
        for group in self.param_groups:
            for p in group['params']:
                grads.append([])
                grads2.append([])

        # Compute grads and grads2 using num_samples MC samples
        for s in range(self.num_samples):

            # Sample noise for each parameter
            pid = 0
            original_values = {}
            for group in self.param_groups:
                for p in group['params']:

                    original_values.setdefault(pid, p.detach().clone())
                    state = self.state[p]
                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p.data)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.ones_like(p.data) * (
                                    group['init_precision'] - group['prior_precision']) / self.train_batch_size

                    # A noisy sample
                    raw_noise = torch.normal(mean=torch.zeros_like(p.data), std=1.0)
                    p.data.addcdiv_(1., raw_noise,
                                    torch.sqrt(self.train_batch_size * state['exp_avg_sq'] + group['prior_precision']))

                    pid = pid + 1

            # Call the loss function and do BP to compute gradient
            loss = closure()

            # Replace original values and store gradients
            pid = 0
            for group in self.param_groups:
                for p in group['params']:

                    # Restore original parameters
                    p.data = original_values[pid]

                    if p.grad is None:
                        continue

                    if p.grad.is_sparse:
                        raise RuntimeError('Vadam does not support sparse gradients')

                    # Aggregate gradients
                    g = p.grad.detach().clone()
                    if s == 0:
                        grads[pid] = g
                        grads2[pid] = g ** 2
                    else:
                        grads[pid] += g
                        grads2[pid] += g ** 2

                    pid = pid + 1

        # Update parameters and states
        pid = 0
        for group in self.param_groups:
            for p in group['params']:

                if grads[pid] is None:
                    continue

                # Compute MC estimate of g and g2
                grad = grads[pid].div(self.num_samples)
                grad2 = grads2[pid].div(self.num_samples)

                tlambda = group['prior_precision'] / self.train_batch_size

                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad + tlambda * original_values[pid])
                exp_avg_sq.mul_(beta2).add_(1 - beta2, grad2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                numerator = exp_avg.div(bias_correction1)
                denominator = exp_avg_sq.div(bias_correction2).sqrt().add(tlambda)

                # Update parameters
                p.data.addcdiv_(-group['lr'], numerator, denominator)

                pid = pid + 1

        return loss

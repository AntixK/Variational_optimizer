import torch
from torch.autograd import Variable
from torch.autograd import grad

def dragan_gradient_penalty(x, f):
    '''
    DRAGAN Gradient Penalty is very similar to WGAN-GP
    except that the perturbation is simply along the manifold
    of the real data rather than the manifold connecting the
    real and fake data.
    ------------------------
    :param f: Discriminator Network
    :param x: Real data - Variable
    :return: Gradient Penalty
    '''
    # Interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape).cuda()
    beta  = torch.rand(x.size()).cuda()
    y = x + 0.5 * x.std() * beta
    z = x + alpha * (y - x)

    # gradient penalty
    z = Variable(z, requires_grad=True).cuda()
    o = f(z)
    g = grad(o, z, grad_outputs=torch.ones(o.size()).cuda(), create_graph=True)[0].view(z.size(0), -1)
    M = 0.05 * torch.ones(z.size(0))
    M = Variable(M).cuda()
    zer =  Variable(torch.zeros(z.size(0))).cuda()
    gp = ((g.norm(p=2, dim=1) - 1)**2).mean() + torch.max(zer,(g.norm(p=2,dim=1) - M)).mean()

    return gp
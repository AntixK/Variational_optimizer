import torch
from torch.autograd import Variable
from torch.autograd import grad

def gradient_penalty(netD, real_data, fake_data, lambda_val=10):
    '''
    Gradient Penalty in WGAN computer as follows -.
    1) Taking a point along the manifold connecting
    the real and fake data points and computing the gradient at that point.
    2) Computing the MSE of the gradient from the value 1.
    ------------------------
    :param netD: Discriminator Network
    :param real_data: Real data - Variable
    :param fake_data: Generated data - Variable
    :param lambda_val: coefficient for the gradient Penalty
    :return: Gradient Penalty
    '''
    #Interpolate Between Real and Fake data
    shape = [real_data.size(0)] + [1] * (real_data.dim() - 1)
    alpha = torch.rand(shape).cuda()
    z = real_data + alpha * (fake_data - real_data)

    # Compute Gradient Penalty
    z = Variable(z, requires_grad=True).cuda()
    disc_z = netD(z)

    gradients = grad(outputs=disc_interpolates, inputs=z,
                              grad_outputs=torch.ones(disc_z.size()).cuda(),
                              create_graph=True)[0].view(z.size(0), -1)

    gradient_penalty = ((gradients.norm(p=2, dim=1) - 1) ** 2).mean() * lambda_val
    return gradient_penalty
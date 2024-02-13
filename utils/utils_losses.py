#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:10:11 2024

@author: rouge
Copied from La Barbera et al. https://github.com/Giammarco07/DeePRAC_project/blob/main/myU-Net/utils/vesselness_torch.py
"""

import numpy as np

import torch
import torchvision.transforms.functional as F

from scipy.ndimage.filters import gaussian_filter

L2 = torch.nn.MSELoss().to(device='cuda')

L1 = torch.nn.L1Loss().to(device='cuda')


# Utils

class eig_real_symmetric(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        '''
        A is the hessian matrix but for all points of all patches, so shape is [batch_dim*x_dim*y_dim*z_dim,3,3]
        so I do evecs[:,0,:] to select for all the voxels, the line that correspond to the x direction (1 for y and 2 for z),
        then I extract the position where this direction is the biggest: xx = torch.argmax(torch.abs(evecs[:,0,:]),dim=1,keepdim=True)

        With torch.linalg.eig the normalized eigenvectors are given per column with respect to the eigenvalue and per row
        with respect to the directions, e.g. for a voxel:
        eval = [-256 -41 0]
        evec = [0.9 0.3 0; 0 0 1; -0.3 0.9 0]

        so eval[0]=-256 and its evec is [0.9; 0; -0.3] = [x; y; z]

        The xx and xxx step is done because the use of permutation matrix (as described in the artcile/thesis) was slower
        and so I just rearranged the eigenvalues and eigenvectors to be [most x-oriented, most y-oriented, most z-oriented].

        So for the example, we will have xx=0, yy=2, zz = 1 and eval will be rearranged as eval = [-256 0 -41]
        and xxx, yyy and zzz simply consist in repeating the values and reorganizing the eigenvector matrix which is used
        for the backpropagation, which will therefore be

        evec = [0.9 0 0.3; 0 1 0; -0.3 0 0.9]
        '''

        evalue, evec = torch.linalg.eig(A)

        evecs = torch.view_as_real(evec)[..., 0]
        evals = torch.view_as_real(evalue)[..., 0]
        evalsnew = evals.clone()
        evecsnew = evecs.clone()

        if torch.isnan(evals).any():
            print(evals[evals != evals])

        xx = torch.argmax(torch.abs(evecs[:, 0, :]), dim=1, keepdim=True)
        xxx = xx.repeat(1, evecs.size()[-1]).view(evecs.size()[0], evecs.size()[-1], 1)
        yy = torch.argmax(torch.abs(evecs[:, 1, :]), dim=1, keepdim=True)
        yyy = yy.repeat(1, evecs.size()[-1]).view(evecs.size()[0], evecs.size()[-1], 1)
        zz = torch.argmax(torch.abs(evecs[:, 2, :]), dim=1, keepdim=True)
        zzz = zz.repeat(1, evecs.size()[-1]).view(evecs.size()[0], evecs.size()[-1], 1)

        evalsnew[:, 0:1] = torch.gather(evals, 1, xx)
        evalsnew[:, 1:2] = torch.gather(evals, 1, yy)
        evalsnew[:, 2:3] = torch.gather(evals, 1, zz)
        evecsnew[:, :, 0:1] = torch.gather(evecs, 2, xxx)
        evecsnew[:, :, 1:2] = torch.gather(evecs, 2, yyy)
        evecsnew[:, :, 2:3] = torch.gather(evecs, 2, zzz)
        ctx.save_for_backward(evecsnew)

        return evalsnew

    @staticmethod
    def backward(ctx, grad_evals):

        '''
        See 'Differentiability of Morphological similarity loss function' for further details on these part
        '''

        # grad_evals: (...,na)
        na = grad_evals.shape[-1]
        nbatch = grad_evals.shape[0]
        dLde = grad_evals.view(-1, na, 1)  # (nbatch, na)
        evecs, = ctx.saved_tensors
        U = evecs.view(-1, na, na)  # (nbatch,na,na)
        UT = U.transpose(-2, -1)  # (nbatch,na,na)
        econtrib = None

        # Control
        U1 = torch.bmm(UT[:, 0:1, :], U[:, :, 0:1])
        U2 = torch.bmm(UT[:, 1:2, :], U[:, :, 1:2])
        U3 = torch.bmm(UT[:, 2:3, :], U[:, :, 2:3])

        if not torch.allclose((torch.mean(U1) + torch.mean(U2) + torch.mean(U3)),
                              torch.tensor([3], dtype=torch.float).to("cuda"), atol=1e-5, rtol=1e-3):
            print(torch.mean(U1), torch.mean(U2), torch.mean(U3))
            # raise ValueError("W not normalized - Gradient forced at 0")
            print("W not normalized - Gradient forced at 0")
            econtrib = torch.zeros((nbatch, na, na))
        else:
            UU1 = (torch.bmm(U[:, :, 0:1], UT[:, 0:1, :])).view(-1, 1, na * na)
            UU2 = (torch.bmm(U[:, :, 1:2], UT[:, 1:2, :])).view(-1, 1, na * na)
            UU3 = (torch.bmm(U[:, :, 2:3], UT[:, 2:3, :])).view(-1, 1, na * na)
            UU = torch.cat((UU1, UU2, UU3), 1)

            '''
            UU = torch.empty(nbatch, na, na*na).to(grad_evals.dtype).to(grad_evals.device)
            # calculate the contribution from grad_evals
            for i in range(nbatch):
                for j in range(3):
                    UU[i,j] = torch.kron(UT[i,j],UT[i,j])
            '''
            UUT = UU.transpose(-2, -1)  # (nbatch,na,na)
            econtrib = torch.bmm(UUT, dLde)
            econtrib = econtrib.view(nbatch, na, na)

        return econtrib

def gaussian_map_hessian(patch_size):
    # Creatian of the gaussian filter
    center_coords = [i // 2 for i in patch_size]
    sigma_scale = 1.
    sigmas = [i * sigma_scale for i in patch_size]
    tmp = np.zeros(patch_size)
    for p,i in enumerate(np.arange(0,patch_size[0]/8,0.25)):
        tmp[p, :, :] = 2**i
        tmp[-p-1, :, :] = 2**i
    for p,j in enumerate(np.arange(0, patch_size[1]/8,0.25)):
        tmp[:, p, :] += 2**j
        tmp[:, -p-1, :] += 2**j
    for p,k in enumerate(np.arange(0, patch_size[2]/8,0.25)):
        tmp[:, :, p] += 2**k
        tmp[:, :, -p-1] += 2**k
    tmp = tmp/tmp.max() * 1

    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant',cval=0)
    gaussian_importance_map = gaussian_importance_map/ np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(np.float32)


    return gaussian_importance_map

def hessian_matrix_sigmas(img, sigmas, gt):
    H = torch.zeros(img.size() + (3, 3), dtype=torch.float32, requires_grad=True).to(torch.device("cuda"))
    '''
    PyTorch or more precisely autograd is not very good in handling in-place operations, 
    especially on those tensors with the requires_grad flag set to True.
    Generally you should avoid in-place operations where it is possible, in some cases it can work, 
    but you should always avoid in-place operations on tensors where you set requires_grad to True.
    Unfortunately there are not many pytorch functions to help out on this problem. 
    So you would have to use a helper tensor to avoid the in-place operation.
    '''
    H = H + 0  # new tensor, out-of-place operation (get around the problem, it is like "magic!")
    sigma = 1
    if gt:
        image = img * 255.0
    else:
        #in case the entire patch is used, in order to deal with the all zero-region a gaussian importance map is
        # applied to have continuity and differentiability
        gaussian_importance_map = torch.as_tensor(
            gaussian_map_hessian((img.size()[1], img.size()[2], img.size()[3])),
            dtype=torch.float16).to("cuda").detach()
        image = (img * 255.0) * gaussian_importance_map


    gaussian_filtered = F.gaussian_blur(image, 2 * int(4 * sigma + 0.5) + 1, sigma)
    for i in range(1 + sigmas // 5, sigmas + 1, sigmas // 5):
        sigma = i
        gaussian_filtered = gaussian_filtered + F.gaussian_blur(image, 2 * int(4 * sigma + 0.5) + 1, sigma)

    gaussian_filtered = gaussian_filtered / gaussian_filtered.max() * 255.0
    H[..., 0, 0] = H[..., 0, 0] + torch.gradient(torch.gradient(gaussian_filtered, dim=1)[0], dim=1)[0]
    H[..., 0, 1] = H[..., 0, 1] + torch.gradient(torch.gradient(gaussian_filtered, dim=1)[0], dim=2)[0]
    H[..., 0, 2] = H[..., 0, 2] + torch.gradient(torch.gradient(gaussian_filtered, dim=1)[0], dim=3)[0]
    H[..., 1, 0] = H[..., 1, 0] + torch.gradient(torch.gradient(gaussian_filtered, dim=2)[0], dim=1)[0]
    H[..., 1, 1] = H[..., 1, 1] + torch.gradient(torch.gradient(gaussian_filtered, dim=2)[0], dim=2)[0]
    H[..., 1, 2] = H[..., 1, 2] + torch.gradient(torch.gradient(gaussian_filtered, dim=2)[0], dim=3)[0]
    H[..., 2, 0] = H[..., 2, 0] + torch.gradient(torch.gradient(gaussian_filtered, dim=3)[0], dim=1)[0]
    H[..., 2, 1] = H[..., 2, 1] + torch.gradient(torch.gradient(gaussian_filtered, dim=3)[0], dim=2)[0]
    H[..., 2, 2] = H[..., 2, 2] + torch.gradient(torch.gradient(gaussian_filtered, dim=3)[0], dim=3)[0]

    return H, gaussian_filtered

 
def eigen_hessian_matrix(image, ref, sigma, gt=True):
    H, gaussian = hessian_matrix_sigmas(image, sigma, gt)
    if gt:
        eigenvalues = eig_real_symmetric.apply(H[ref == 1])
    else:
        eigenvalues = eig_real_symmetric.apply(H[ref == 1].view(-1, H.shape[-2], H.shape[-1]))

    return eigenvalues

    
def vesselness_calc(pred, target, ref, sigma, gt, l = 'L2', deep = None):
    if deep is None:
        eigenv_tr = eigen_hessian_matrix(target, ref, sigma, gt)
        eigenvtrue = eigenv_tr.detach()
    else:
        eigenvtrue = deep
    eigenv = eigen_hessian_matrix(pred, ref, sigma, gt)
    if l=='L2':
        loss = L2(eigenv, eigenvtrue)
    elif l=='L1':
        loss = L1(eigenv, eigenvtrue)
    else:
        print('Warning! Loss neither L2 or L1, forced to use L2')
        loss = L2(eigenv, eigenvtrue)

    if torch.isnan(loss):
        print(eigenv)
        print(eigenvtrue)

    return loss, eigenvtrue

def frangi(eigenvalues, alpha=0.1, beta=0.1, gamma=2):
    Ra = torch.abs(eigenvalues[..., 1]) / (torch.abs(eigenvalues[..., 2]) + 1e-6)
    Rb = torch.abs(eigenvalues[..., 0]) / (
                torch.sqrt(torch.abs(eigenvalues[..., 1]) * (torch.abs(eigenvalues[..., 2])) + 1e-6) + 1e-6)
    S = torch.sqrt((eigenvalues[..., 0] ** 2) + (eigenvalues[..., 1] ** 2) + (eigenvalues[..., 2] ** 2) + 1e-6)
    F = (1 - torch.exp(-(Ra ** 2) / (2 * (alpha ** 2)))) * torch.exp(-(Rb ** 2) / (2 * (beta ** 2))) * (
                1 - torch.exp(-(S ** 2) / (2 * (gamma ** 2))))
    F = torch.where((eigenvalues[..., 1] < 0) & (eigenvalues[..., 2] < 0), F, F * 0.)
    F = F / (torch.max(F) + 1e-6)

    return F


# Losses

def msloss(output,target, sigma=15, gt=True):
    '''
    Loss function to check the morphology of the structures, named morphological similarity loss function and denoted by MsLoss,
    by comparing the eigenvalues ordered by the eigenvectors matrix as presented in the article/thesis

    output: predictions at different level of resolution (list of prediction BxCxHxWxD), if you do not use deep supervision
            you need to pass this argument, i.e. your output prediction BxCxHxWxD, as msloss([output],target,sigma)
    target: reference segmentation (BxCxHxWxD)

    sigma = maximum standard deviations for Hessian matrix, we apply 5 different Gaussian kernel from 1 to sigma. Default sigma=25
    gt = the eigenvalues are calculated just on the dilation with a square structuring element of size 3 × 3 × 3
        of the reference target (calculating eigenvalues over the entire image is expensive in terms of computational time,
        and the use of dilation revealed to be sufficient for our purpose thanks also to the combined use of voxel-wise loss functions).
        Default gt=True
    '''
    flag = 0
    channel_dim = target.size()[1]
    p_range = len(output)
    w = np.array([1 / (2 ** i) for i in range(p_range)])
    w = w / w.sum()

    if gt:
        #creating the dilation of the reference segmentation in which we will calculate the eigenvalues
        kernel = torch.ones((channel_dim, 1, 3, 3, 3), dtype=torch.float16).to("cuda")
        gt_dil = torch.clamp(torch.nn.functional.conv3d(target.type(torch.float16), kernel, padding=1, groups=channel_dim),
                             0, 1)
        for v in range(0, channel_dim):
            gt_ = gt_dil[:, v, :, :, :]
            gt_dilate = gt_.detach()
            #verify is the batch is not all empty patches for that structure
            wt = torch.sum(target[:, v]).detach()
            if wt != 0:
                for p in range(p_range):  #deep supervision
                    out = output[p].clone() + 1e-20
                    if p==0:
                        loss_v, eigentrue = vesselness_calc(out[:, v], target[:, v], gt_dilate, sigma, gt)
                        if flag == 0:
                            loss_vessel = w[p] * loss_v
                            flag = 1
                        else:
                            loss_vessel += w[p] * loss_v
                    else: #we avoid to recalculate the eigenvals of the target (they do not change)
                        loss_v, _ = vesselness_calc(out[:, v], target[:, v], gt_dilate, sigma-(sigma//5*p), gt, deep=eigentrue)
                        loss_vessel += w[p] * loss_v
            else:
                if flag == 0:
                    loss_vessel = torch.sum(output[0][:, v]*0.0) #loss to 0 for that structure without breaking the graph
                    flag = 1
                else:
                    loss_vessel += torch.sum(output[0][:, v] * 0.0)  # loss to 0 for that structure without breaking the graph
    else:
        for v in range(0, channel_dim):
            #selecting just the not empty patches
            ww = torch.sum(target[:, v], dim=(1, 2, 3)) > 0
            wt = ww.detach()
            #verify is the batch is not all empty patches for that structure
            wwt = torch.sum(target[:, v]).detach()
            if wwt != 0:
                for p in range(p_range):  # deep supervision
                    out = output[p].clone() + 1e-20
                    if p == 0:
                        loss_v, eigentrue = vesselness_calc(out[:, v], target[:, v], wt, sigma, gt)
                        if flag == 0:
                            loss_vessel = w[p] * loss_v
                            flag = 1
                        else:
                            loss_vessel += w[p] * loss_v
                    else:  # we avoid to recalculate the eigenvals of the target (they do not change)
                        loss_v, _ = vesselness_calc(out[:, v], target[:, v], wt, sigma - (sigma // 5 * p), gt,
                                                    deep=eigentrue)
                        loss_vessel += w[p] * loss_v
            else:
                if flag == 0:
                    loss_vessel = torch.sum(output[0][:, v] * 0.0)  # loss to 0 for that structure without breaking the graph
                    flag = 1
                else:
                    loss_vessel += torch.sum(output[0][:, v] * 0.0)  # loss to 0 for that structure without breaking the graph
    # remove the multiplicative factor (1/(channel_dim -1))
    return loss_vessel

def fvloss(output,target,sigma=15, alpha=0.1,beta=0.1,gamma=2):
    '''
    Loss function to force prediction of elongated structures as in Frangi’s vesselness function,
    and thus named Frangi’s vesselness loss function FvLoss.
    Default for Frangi are alpha=0.1,beta=0.5,gamma=5.

    output: predictions at different level of resolution (list of prediction BxCxHxWxD), if you do not use deep supervision
            you need to pass this argument, i.e. your output prediction BxCxHxWxD, as msloss([output],target,sigma)
    target: reference segmentation (BxCxHxWxD)

    sigma = maximum standard deviations for Hessian matrix, we apply 5 different Gaussian kernel from 1 to sigma. Default sigma=25
    '''

    flag = 0
    channel_dim = target.size()[1]
    p_range = len(output)
    w = np.array([1 / (2 ** i) for i in range(p_range)])
    w = w / w.sum()

    for v in range(0, channel_dim):
        # verify is the batch is not all empty patches for that structure
        wwt = torch.sum(target[:, v]).detach()
        if wwt != 0:
            for p in range(p_range):  # deep supervision
                out = output[p].clone() + 1e-20
                if p == 0:
                    eigenv = eigen_hessian_matrix(out[:, v], target[:, v], sigma)
                    _, indices = torch.abs(eigenv).sort(dim=1, stable=True)
                    eigenvalues = torch.take_along_dim(eigenv, indices, dim=1)
                    loss_v = torch.mean(1 - frangi(eigenvalues,alpha,beta,gamma))
                    if flag == 0:
                        loss_vessel = w[p] * loss_v
                        flag = 1
                    else:
                        loss_vessel += w[p] * loss_v
                else:
                    eigenv = eigen_hessian_matrix(out[:, v], target[:, v],  sigma - (sigma // 5 * p))
                    _, indices = torch.abs(eigenv).sort(dim=1, stable=True)
                    eigenvalues = torch.take_along_dim(eigenv, indices, dim=1)
                    loss_v = torch.mean(1 - frangi(eigenvalues,alpha,beta,gamma))
                    loss_vessel += w[p] * loss_v
        else:
            if flag == 0:
                loss_vessel = torch.sum(out[:, v] * 0.0)  # loss to 0 for that structure without breaking the graph
                flag = 1
            else:
                loss_vessel += torch.sum(out[:, v] * 0.0)  # loss to 0 for that structure without breaking the graph

    # remove the multiplicative factor (1/(channel_dim -1))
    return loss_vessel
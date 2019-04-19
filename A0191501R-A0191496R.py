'''
Description: CS5340 - Hopfield Network
Name: Anshul Aggarwal, Amit Sukhpal
Matric No.: A0191501R, A0191496R
'''


import matplotlib
matplotlib.use('Agg')
import numpy as np
import glob
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from copy import deepcopy
import math
import time



def load_image(fname):
    img = Image.open(fname).resize((32, 32))
    img_gray = img.convert('L')
    img_eq = ImageOps.autocontrast(img_gray)
    img_eq = np.array(img_eq.getdata()).reshape((img_eq.size[1], -1))
    return img_eq


def binarize_image(img_eq):
    img_bin = np.copy(img_eq)
    img_bin[img_bin < 128] = -1
    img_bin[img_bin >= 128] = 1
    return img_bin


def add_corruption(img):
    img = img.reshape((32, 32))
    t = np.random.choice(3)
    if t == 0:
        i = np.random.randint(32)
        img[i:(i + 8)] = -1
    elif t == 1:
        i = np.random.randint(32)
        img[:, i:(i + 8)] = -1
    else:
        mask = np.sum([np.diag(-np.ones(32 - np.abs(i)), i)
                       for i in np.arange(-4, 5)], 0).astype(np.int)
        img[mask == -1] = -1
    return img.ravel()


def learn_hebbian(imgs):
    img_size = np.prod(imgs[0].shape)
    num_images = imgs.shape[0]

    bias = np.zeros(img_size)
    weights = np.zeros((img_size, img_size))

    nodes = np.array([np.ndarray.flatten(imgs[i]) for i in range(num_images)])

    for i in range(img_size):
        for j in range(i+1,img_size):
            weights[i,j] = np.sum(np.multiply(nodes[:,i],nodes[:,j]))/num_images
            weights[j,i] = weights[i,j]    

    return weights, bias


def calc_val(x):
    # Derivative of the log maximum pseudo-likelihood, reduced.
    # d(-ln(sgm(x))/dx = -1/sgm(x) * sgm(x) * (1 - sgm(x)) 
    #                  = -1*(1 + exp(-x) - 1)/(1 + exp(-x))
    #                  = -1*(exp(-x))/((exp(x) + 1)/exp(x))
    #                  = -1*(exp(-x+x))/(1 + exp(x))
    #                  = -1/(1 + exp(x))
    # The related image vector is multiplied when differentiating by weight, and 1 if differentiating by bias, in learn_maxpl.
    return (-1 / (1 + math.exp(x)))


def get_gradient(weights, bias, nodes, i):
    num_images = nodes.shape[0]
    img_size = nodes.shape[1]
    gradients = np.zeros(num_images)
    for j in range(num_images):
        y = (np.sum(np.multiply(weights[i,:], nodes[j,:])))/(img_size**2) + bias[i]
        gradients[j] = calc_val(y)
        if nodes[j,i] != 1:
            gradients[j] = 1 - gradients[j]

    return gradients


def learn_maxpl(imgs):
    img_size = np.prod(imgs[0].shape)
    num_images = imgs.shape[0]

    weights = np.random.random((img_size, img_size))
    bias = np.random.random(img_size)

    nodes = np.array([np.ndarray.flatten(imgs[i]) for i in range(num_images)])
    epochs = 300
    lr = 0.05    #learning rate
    start = time.time()

    for e_ in range(epochs):
        for i in range(img_size):
            gradients = get_gradient(weights, bias, nodes, i)
            wt_update = np.zeros((num_images, img_size))
            bias_update = np.zeros(num_images)
            for j in range(num_images):
                wt_update[j,:] = (gradients[j]*nodes[j,:])/num_images
                bias_update[j] = gradients[j]/num_images

            weights[i,:] = weights[i,:] - lr*np.sum(wt_update, axis=0)
            bias[i] = bias[i] - lr * np.sum(bias_update)

    for i in range(img_size):
        weights[i,i] = 0

    return weights, bias


def plot_results(imgs, cimgs, rimgs, fname='result.png'):
    '''
    This helper function can be used to visualize results.
    '''
    img_dim = 32
    assert imgs.shape[0] == cimgs.shape[0] == rimgs.shape[0]
    n_imgs = imgs.shape[0]
    fig, axn = plt.subplots(n_imgs, 3, figsize=[8, 8])
    for j in range(n_imgs):
        axn[j][0].axis('off')
        axn[j][0].imshow(imgs[j].reshape(img_dim, img_dim), cmap='Greys_r')
    axn[0, 0].set_title('True')
    for j in range(n_imgs):
        axn[j][1].axis('off')
        axn[j][1].imshow(cimgs[j].reshape(img_dim, img_dim), cmap='Greys_r')
    axn[0, 1].set_title('Corrupted')
    for j in range(n_imgs):
        axn[j][2].axis('off')
        axn[j][2].imshow(rimgs[j].reshape((img_dim, img_dim)), cmap='Greys_r')
    axn[0, 2].set_title('Recovered')
    fig.tight_layout()
    plt.savefig(fname)


def recover(cimgs, W, b):
    img_size = np.prod(cimgs[0].shape)
    num_images = cimgs.shape[0]

    corrupted_images = np.array([np.ndarray.flatten(cimgs[i]) for i in range(num_images)])

    rimgs = []
    for img in corrupted_images:
        epoch_count = 0
        recovd_img = deepcopy(img)
        while(True):
            i = np.random.randint(0,img_size) 
            # Random pixel is chosen for recovery, rather than sequential recovery. 
            # This is because corruption is sequential (block corruption), sequential recovery may cause corrupted pixels to further corrupt non-corrupted pixels.
            update = np.sum(np.multiply(W[i,:], img))
            if update < b[i]:
                img[i] = -1
            else:
                img[i] = 1
            epoch_count += 1

            if epoch_count%int(img_size*2.0) == 0:
                if np.array_equal(recovd_img, img):
                    break
                else:
                    recovd_img = deepcopy(img)

    rimgs = corrupted_images

    return rimgs


def main():
    # Load Images and Binarize
    ifiles = sorted(glob.glob('images/*'))
    timgs = [load_image(ifile) for ifile in ifiles]
    imgs = np.asarray([binarize_image(img) for img in timgs])

    # Add corruption
    cimgs = []
    for i, img in enumerate(imgs):
        cimgs.append(add_corruption(np.copy(imgs[i])))
    cimgs = np.asarray(cimgs)

    # Recover 1 -- Hebbian
    Wh, bh = learn_hebbian(imgs)
    rimgs_h = recover(cimgs, Wh, bh)
    np.save('hebbian.npy', rimgs_h)

    # Recover 2 -- Max Pseudo Likelihood
    Wmpl, bmpl = learn_maxpl(imgs)
    rimgs_mpl = recover(cimgs, Wmpl, bmpl)
    np.save('mpl.npy', rimgs_mpl)


if __name__ == '__main__':
    main()

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

    #Initialize Weights

    for i in range(img_size):
        for j in range(i+1,img_size):
            weights[i,j] = np.sum(np.multiply(nodes[:,i],nodes[:,j]))/num_images
            weights[j,i] = weights[i,j]    

    return weights, bias


def learn_maxpl(imgs):
    img_size = np.prod(imgs[0].shape)
    ######################################################################
    ######################################################################
    weights = np.zeros((img_size, img_size))
    bias = np.zeros(img_size)
    # Complete this function
    # You are allowed to modify anything between these lines
    # Helper functions are allowed
    #######################################################################
    #######################################################################
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
            update = np.sum(np.multiply(W[i,:], img))   # division with (img_size*img_size) to calculate actual dot product is not required, as we only need the sign.
            if update < b[i]:
                img[i] = -1
            else:
                img[i] = 1
            epoch_count += 1
            
            if epoch_count%int(img_size*1.5) == 0:
                if np.array_equal(recovd_img, img):
                    break
                else:
                    recovd_img = deepcopy(img)
        
        print("Converged after", epoch_count, "epochs")
    
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
    plot_results(imgs, cimgs, rimgs_h, "hebbian-results.png")       #not in original code for main

    # Recover 2 -- Max Pseudo Likelihood
    #Wmpl, bmpl = learn_maxpl(imgs)
    #rimgs_mpl = recover(cimgs, Wmpl, bmpl)
    #np.save('mpl.npy', rimgs_mpl)


if __name__ == '__main__':
    main()

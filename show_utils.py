import os
import torch
from torch.autograd import Variable
import itertools
import matplotlib.pyplot as plt


# to make 25 generetive images show in a figure
def show_result(num_epoch,  G, fixed_z, show = False, save = False, path = 'result.png', isFix=False):
    z = torch.randn((5*5, 100))
    z = Variable(z.cuda(), volatile=True)

    G.eval()
    if isFix:
        test_images = G(fixed_z)
    else:
        test_images = G(z)
    G.train()

    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(5*5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        ax[i, j].imshow(test_images[k, :].cpu().data.view(28, 28).numpy(), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    if save:
        plt.savefig(path)
    if show:
        plt.show()
    else:
        plt.close()


# to make 25 mnist images show in a figure
def show_row_mnist(mnist_data, show=False, path='MNIST.png'):
    if not os.path.exists(path):
        raw_mnist = []
        size_figure_grid = 5
        fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
        for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)
        count = 0
        for x_data, _ in mnist_data:
            if count < size_figure_grid ** 2:
                raw_mnist.append(x_data)
                count += 1
        for k in range(size_figure_grid ** 2):
            i = k // 5
            j = k % 5
            ax[i, j].cla()
            ax[i, j].imshow(raw_mnist[k]， cmap='gray')

        label = 'MNIST samples'
        fig.text(0.5, 0.04, label, ha='center')
        plt.savefig(path)

        if show:
            plt.show()
        else:
            plt.close()

# to show the history train loss of G and D
def show_train_hist(hist, show = False, save = False, path ='Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()



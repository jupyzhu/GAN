import os
import pickle
import imageio
import torch
import torch.optim as optim
from torch.autograd import Variable
from model import generator, discriminator, BCE_loss
from show_utils import show_result, show_row_mnist, show_train_hist
from mnist_loader import mnist_loader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# fixed noise
with torch.no_grad():
    fixed_z = torch.randn((5 * 5, 100)).cuda()

def train():
    # training parameters
    batch_size = 128
    lr = 0.0002
    train_epoch = 100

    train_loader = mnist_loader(batch_size)

    # network
    G = generator(input_size=100, image_size=28*28)
    D = discriminator(input_size=28*28, lable_size=1)
    G.cuda()
    D.cuda()

    # Adam optimizer
    G_optimizer = optim.Adam(G.parameters(), lr=lr)
    D_optimizer = optim.Adam(D.parameters(), lr=lr)

    # creat folders
    if not os.path.isdir('GAN_results'):
        os.mkdir('GAN_results')
    if not os.path.isdir('GAN_results/Random_results'):
        os.mkdir('GAN_results/Random_results')
    if not os.path.isdir('GAN_results/Fixed_results'):
        os.mkdir('GAN_results/Fixed_results')

    train_hist = {}
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []

    # ************************************训练开始***********************************
    for epoch in range(train_epoch):
        D_losses = []
        G_losses = []
        for x_data, _ in train_loader:  # the first is image, second is label
            # ************** train discriminator D **********************
            D.zero_grad()
            # make x_(128,1,28,28) to be (128,784)
            x_data = x_data.view(-1, 28 * 28)

            mini_batch = x_data.size()[0]

            y_real = torch.ones(mini_batch).reshape(-1,1)
            y_fake = torch.zeros(mini_batch).reshape(-1,1)

            x_data, y_real, y_fake = Variable(x_data.cuda()), Variable(y_real.cuda()), Variable(y_fake.cuda())
            D_result = D(x_data)
            D_real_loss = BCE_loss(D_result, y_real)
            # D_real_score = D_result

            z_data = torch.randn((mini_batch, 100))
            z_data = Variable(z_data.cuda())
            G_result = G(z_data)

            D_result = D(G_result)
            D_fake_loss = BCE_loss(D_result, y_fake)
            # D_fake_score = D_result

            D_train_loss = D_real_loss + D_fake_loss

            D_train_loss.backward()
            D_optimizer.step()

            D_losses.append(D_train_loss.data[0])

            # ****************************** train generator G *********************
            G.zero_grad()

            z = torch.randn((mini_batch, 100))
            y = torch.ones(mini_batch).view(-1, 1)

            z, y = Variable(z.cuda()), Variable(y.cuda())
            G_result = G(z)
            D_result = D(G_result)
            G_train_loss = BCE_loss(D_result, y)
            G_train_loss.backward()
            G_optimizer.step()

            G_losses.append(G_train_loss.data[0])

        # print loss message every epoch
        print(f'[{(epoch + 1)}/{train_epoch}]: loss_d: {torch.mean(torch.FloatTensor(D_losses)):.3f}, loss_g:\
        {torch.mean(torch.FloatTensor(G_losses)):.3f}')

        # to generate 25 images in a figure with fixed or not fixed noisy under every epoch G model
        p = 'GAN_results/Random_results/' + str(epoch + 1) + '.png'
        fixed_p = 'GAN_results/Fixed_results/' + str(epoch + 1) + '.png'
        show_result((epoch+1), fixed_z, save=True, path=p, isFix=False, G=G)
        show_result((epoch+1), fixed_z, save=True, path=fixed_p, isFix=True, G=G)

        # to record mean D_loss and mean G_loss of each epoch
        train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
        train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))


    print("Training finish!... save training results")

    # save model and loss record
    torch.save(G.state_dict(), "GAN_results/generator_param.pkl")
    torch.save(D.state_dict(), "GAN_results/discriminator_param.pkl")
    with open('GAN_results/train_hist.pkl', 'wb') as f:
        pickle.dump(train_hist, f)

    # draw loss picture
    show_train_hist(train_hist, save=True, path='GAN_results/GAN_train_hist.png')

    # draw animation
    images1 = []
    images2 = []
    for e in range(train_epoch):
        img_name1 = 'GAN_results/Fixed_results/' + str(e + 1) + '.png'
        img_name2 = 'GAN_results/Random_results/' + str(e + 1) + '.png'
        images1.append(imageio.imread(img_name1))
        images2.append(imageio.imread(img_name2))
    imageio.mimsave('GAN_results/generation_animation_Fixed.gif', images1, fps=5)
    imageio.mimsave('GAN_results/generation_animation_Random.gif', images2, fps=5)

    show_row_mnist(train_loader, show=False, path='GAN_results/MNIST.png')


if __name__ == '__main__':
    train()


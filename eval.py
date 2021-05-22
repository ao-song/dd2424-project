import torch
from model import Colorizer
from preprocess import lab_test_loader
from torch import split
from torch.autograd import Variable
from skimage import color
from PIL import Image
import matplotlib.pyplot as plt
from ultils import get_space
import numpy as np

# get trained model
mse_model = Colorizer()
mse_model.load_state_dict((torch.load('models/cifar10_colorizer19')))
mse_model.eval()

ce_model = Colorizer(is_soft_encoding=True)
ce_model.load_state_dict((torch.load('models/cifar10_colorizer_soft_encoding19')))
ce_model.eval()


for i, data in enumerate(lab_test_loader):

    img_name = 'test7.png'
    test_img = data[7, :, :, :]
    # get the first picture in batch
    orig = test_img.data.cpu().numpy().T
    test_img = torch.reshape(test_img, (1, 3, 32, 32))
    print(test_img.shape)



    # save orig image
    # im = Image.fromarray(color.lab2rgb(orig), mode='RGB')
    # im.show()
    # im.save('orig.png', 'PNG')


    # get l dimension of orig img
    lab = split(test_img, [1, 2], dim=1)
    l = lab[0]
    # l = split(l, [1, 99], dim=0)
    # l = l[0] # get the first image
    # print(l.shape)
    l = Variable(l)

    # predict ab
    ab = mse_model(l)
    print(ab.shape)
    out = torch.cat((l, ab), dim=1)
    print(ab)

    # transform to rgb and save the img
    rgb = color.lab2rgb(out.data.cpu().numpy()[0,...].T)

    soft_ab = ce_model(l)
    # Restore predicted color by getting mode
    space = get_space()
    soft_ab = soft_ab.data.cpu().numpy()
    a = space[np.argmax(soft_ab, axis=1)][:, :, :, 0]
    b = space[np.argmax(soft_ab, axis=1)][:, :, :, 1]

    a = torch.from_numpy(a.reshape((1, 1, 32, 32)))
    b = torch.from_numpy(b.reshape((1, 1, 32, 32)))
    ce_out = torch.cat((l, a, b), dim=1)
    # transform to rgb and save the img
    ce_rgb = color.lab2rgb(ce_out.data.cpu().numpy()[0,...].T)
    # print(space.shape)

    # print(a.shape)
    # print(a)
    # print(b.shape)
    # print(b)

    # print(np.argmax(soft_ab, axis=1))
    # print(space[np.argmax(soft_ab, axis=1)].shape)
    # print(space[253])
    # print(l.shape)

    # print(rgb.shape)
    # im = Image.fromarray(rgb, mode='RGB')
    # im.save('rgb.png', 'PNG')
    fig = plt.figure(figsize=(10, 10))
    fig.add_subplot(131)
    plt.imshow(color.lab2rgb(orig))
    plt.title('Orig image')

    fig.add_subplot(132)
    plt.imshow(rgb)
    plt.title('Predicted image 1')

    fig.add_subplot(133)
    plt.imshow(ce_rgb)
    plt.title('Predicted image 2')

    plt.savefig(img_name)

    break

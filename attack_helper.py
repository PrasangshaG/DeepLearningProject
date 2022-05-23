import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

'''
Input:

data: 1 image tensor of size (3, 48, 48)
target: 1 tensor representing the class label

Output: 
Predicted class label, True/False if the prediction was correct
'''


def predict_output(data, target, model, device):
    model.eval()
    with torch.no_grad():
        data, target = Variable(data).to(device), Variable(target).to(device)
        data = data.unsqueeze(0)
        output = model(data)
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        c = pred.eq(target.data.view_as(pred)).item()
        return pred.item(), c


def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return tensor


def gr2bin(img, thresh):
    ht = img.shape[0]
    wt = img.shape[1]
    bina = np.zeros((ht, wt), np.uint8)
    for i in range(ht):
        for j in range(wt):
            if(img[i, j] > thresh):
                bina[i, j] = 1
            else:
                bina[i, j] = 0
    return bina


def distinctVals(grayimage):
    distinctval = {}
    ht = grayimage.shape[0]
    wt = grayimage.shape[1]
    for i in range(ht):
        for j in range(wt):
            pixel = grayimage[i, j]
            if(pixel not in distinctval):
                distinctval[pixel] = 0
            else:
                distinctval[pixel] = distinctval[pixel] + 1
    return distinctval


def plot_results(attack_vals, name):
    # images_perturb_vals
    marklist1 = sorted((value, key) for (key, value) in attack_vals.items())

    sortdict1 = dict([(k, v) for v, k in marklist1])

    res_dict1 = {}
    iterato1 = {}

    i = 0

    for elem in sortdict1:
        i = i+1
        res_dict1[sortdict1[elem]] = i/50*100

    for el in sortdict1:
        val = sortdict1[el]
        if val not in iterato1:
            iterato1[val] = 1
        else:
            iterato1[val] += 1
    # print(sortdict)
    # print(res_dict1)
    x1, y1 = zip(*res_dict1.items())

    plt.figure(figsize=(10, 8))
    plt.plot(x1, y1, 'b-', lw=2.5)
    plt.xlabel('Number of pixels', fontsize=16)
    plt.ylabel('Successful attack(%)', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(range(0, 101, 10), fontsize=16)
    # plt.legend()
    plt.grid(alpha=0.15)
    plt.title(name, fontsize=22)
    plt.show()

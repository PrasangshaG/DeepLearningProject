from attack_helper import distinctVals, predict_output, plot_results
import random
import numpy as np
import torch
import torchvision.transforms as transforms
import cv2
import math
import matplotlib.pyplot as plt

# A neighbor is one which has every pixel changed by k amount ### (all-pixel-gaussian: Attack 3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def RandomNeighbor_allp(bin_ten, k):
    newt = torch.tensor(bin_ten)
    newt = torch.add(newt, k)
    min2 = torch.min(newt)
    max2 = torch.max(newt)
    #newt -= min2
    newt /= max2
    return newt


def SimulatedAnnealing_allp(bin21, initialT, endT, bestf, bestobjf, model):
    initial = np.copy(bin21)
    T = initialT
    current = np.copy(initial)
    print("Simulated Annealing Optimization isn progress...")
    perturbations = []
    pertubedimages = []
    while(T > endT):
        for ite in range(int(len(current)/2)):
            transform = transforms.ToTensor()
            initial_ten = transform(current)
            cls = [0]
            cls_ten = torch.from_numpy(np.asarray(cls))
            currentobj, temp1 = predict_output(
                initial_ten, cls_ten, model, device)
            current_ten = initial_ten
            ####
            k = np.random.normal(0.5, 0.2, 1)[0]
            # print(k)
            if(k >= 0.01 and k <= 0.99):
                new_ten = RandomNeighbor_allp(initial_ten, k)
                new_img = new_ten.permute(1, 2, 0).numpy()
                pertubedimages.append(new_img)
                newobj, temp2 = predict_output(new_ten, cls_ten, model, device)
                #print(newobj, currentobj)
                delE = (newobj - currentobj)
                Pcn = 1/(1 + math.exp((-1*delE)/T))
                ranThresh = random.uniform(0, 1)
                if(ranThresh < Pcn):
                    currentobj = newobj
                    current = new_img
                    if(newobj < bestobjf):
                        bestf = new_img
                        bestobjf = newobj
        T = T*0.9
    return bestf, bestobjf, pertubedimages


def SingleImage_Vulnerability_Allp(test_item, test_class, model):
    #test_item, test_class = iter(test_loader).next()
    # print(test_class[0].item())
    actual_label, cld = predict_output(
        test_item[0], test_class[0], model, device)

    tensor_im = test_item[0].permute(1, 2, 0).numpy()

    # Convert the image to PyTorch tensor
    transform = transforms.ToTensor()
    bv_tensor = transform(tensor_im)
    ac_label, cld = predict_output(bv_tensor, test_class[0], model, device)
    # print(ac_label)
    initialT = 1000
    endT = 0.1

    bin2 = tensor_im
    initial = np.copy(bin2)
    transform = transforms.ToTensor()
    initial_ten = transform(tensor_im)

    cls = [0]
    cls_ten = torch.from_numpy(np.asarray(cls))
    obj, cls = predict_output(initial_ten, cls_ten)
    bestfix = initial_ten
    bestobjfix = obj

    b1, b2, prti = SimulatedAnnealing_allp(
        initial, initialT, endT, bestfix, bestobjfix, model)

    # plt.imshow(initial)
    for i in range(len(prti)):
        kl, nm = predict_output(
            transform(prti[i]), test_class[0], model, device)
        if(kl != actual_label):
            break

    return tensor_im, prti[i], actual_label, kl, i


def all_pixel_gaussian_sample(test_loader, model, sample_size=20):
    imlst = []
    imdict = {}
    idlist = []
    idi = 0
    for img, img_class in test_loader:
        imlst.append(img)
        imdict[img] = img_class
        idlist.append(idi)
        idi += 1

    images_perturb_vals3 = {}
    images_labels = {}

    for i in range(sample_size):
        idx = random.sample(idlist, 1)[0]
        # print(idx)
        idlist.remove(idx)
        im = imlst[idx]
        im_cl = imdict[im]
        k = 1
        actual_image, perturbed_image, actual_label, perturbed_label, mnb = SingleImage_Vulnerability_Allp(
            im, im_cl, model)
        images_perturb_vals3[idx] = k*mnb

    plot_results(images_perturb_vals3, "Attack 2: Contiguous $k$-pixel attack")

    return images_perturb_vals3

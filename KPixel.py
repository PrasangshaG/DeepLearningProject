from attack_helper import distinctVals, predict_output, plot_results
import random
import numpy as np
import torch
import torchvision.transforms as transforms
import cv2
import math
import matplotlib.pyplot as plt
# A neighbor is one which has exactly k pixels different

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def RandomNeighbor(bin211, k):
    v = distinctVals(bin211)
    values = v.values()
    max_value = max(v)
    min_value = min(v)
    neigh_image = np.copy(bin211)
    s1_list = []
    s2_list = []
    for i in range(bin211.shape[0]):
        s1_list.append(i)
    for i in range(bin211.shape[1]):
        s2_list.append(i)
    perturbations = []
    for i in range(k):
        s1_n = random.sample(s1_list, 1)[0]
        s1_list.remove(s1_n)
        s2_n = random.sample(s2_list, 1)[0]
        s2_list.remove(s2_n)
        if(bin211[s1_n][s2_n] != min_value or bin211[s1_n][s2_n] != min_value+1 or bin211[s1_n][s2_n] != min_value+2):
            neigh_image[s1_n][s2_n] = min_value
        else:
            neigh_image[s1_n][s2_n] = max_value
        perturbations.append((s1_n, s2_n))

    return neigh_image, perturbations


def SimulatedAnnealing(bin21, initialT, endT, k, bestf, bestobjf, model):
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
            ####
            new_im, perturb = RandomNeighbor(
                cv2.cvtColor(current, cv2.COLOR_RGB2GRAY), k)
            new_img = cv2.cvtColor(new_im, cv2.COLOR_GRAY2RGB)
            perturbations.append(perturb)
            pertubedimages.append(new_img)
            new_ten = transform(new_img)
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
        T = T*0.95
    return bestf, bestobjf, perturbations, pertubedimages


def SingleImage_Vulnerability_Noconstr(test_item, test_class, k, model):
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
    obj, cls = predict_output(initial_ten, cls_ten, model, device)
    bestfix = initial_ten
    bestobjfix = obj

    b1, b2, prt, prti = SimulatedAnnealing(
        initial, initialT, endT, k, bestfix, bestobjfix, model)

    plt.imshow(initial)
    for i in range(len(prti)):
        kl, nm = predict_output(
            transform(prti[i]), test_class[0], model, device)
        if(kl != actual_label):
            break

    return tensor_im, prti[i], actual_label, kl, i


def k_pixel_attack1(test_loader, model):
    # Executes attack 1 on one image
    test_item, test_class = iter(test_loader).next()
    k = 1
    actual_image, perturbed_image, actual_label, perturbed_label, i = SingleImage_Vulnerability_Noconstr(
        test_item, test_class, k, model)
    print(str("The label of actual image is ")+str(actual_label))
    print(str("The label of perturbed image is ")+str(perturbed_label))
    print(str("The number of changes are  ")+str(k*i))
    print("The actual image is: ")
    plt.imshow(actual_image)
    print("The perturbed image is: ")
    plt.imshow(perturbed_image)


def k_pixel_attack_sample(test_loader, model, sample_size=30):
    # images_perturb_vals
    imlst = []
    imdict = {}
    idlist = []
    idi = 0
    for img, img_class in test_loader:
        imlst.append(img)
        imdict[img] = img_class
        idlist.append(idi)
        idi += 1

    images_perturb_vals1 = {}

    for i in range(sample_size):
        idx = random.sample(idlist, 1)[0]
        # print(idx)
        idlist.remove(idx)
        im = imlst[idx]
        im_cl = imdict[im]
        k = 1
        actual_image, perturbed_image, actual_label, perturbed_label, mnb = SingleImage_Vulnerability_Noconstr(
            im, im_cl, k, model)
        images_perturb_vals1[idx] = k*mnb

    plot_results(images_perturb_vals1, "Attack 1: $k$-pixel attack")

    return images_perturb_vals1

import torch
import copy
import numpy as np

from utils_quantize import *
from scipy.integrate import quad

def top_k_sparsificate_model_weights(weights, fraction):
    tmp_list = []
    # for el in weights:
    #     lay_list = el.reshape((-1)).tolist()
    #     tmp_list = tmp_list + [abs(el) for el in lay_list]
    tmp_list = torch.abs(weights).reshape((-1)).tolist()
    tmp_list.sort(reverse=True)
    #print("total number of parameters:",len(tmp_list))
    #TODO
    # same as weight.reshape.size[0] ? better make it more general
    # write as in 183
    k_th_element = tmp_list[int(fraction*len(tmp_list))-1] # 552874 is the number of parameters of the CNNs!       23608202:Res50   0.0004682019352912903

    mask = torch.ge(torch.abs(weights),k_th_element)
    #new_weights = torch.mul(weights, mask.int().float())

    new_weights = copy.deepcopy(weights)
    new_weights[mask==False] = 0

    return new_weights


def pdf_gennorm(x, a, m, b):
  return stats.gennorm.pdf(x,a,m,b)

def update_centers_magnitude_distance(data, R, M, iterations_kmeans):
    #TODO: allow change of m
    #M = 1
    #data = data.cpu().numpy()
    mu = np.mean(data)
    s = np.var(data)
    data_normalized = np.divide(np.subtract(data,mu),np.sqrt(s))
    a, m, b = stats.gennorm.fit(data_normalized)

    xmin, xmax = min(data_normalized), max(data_normalized)
    random_array = np.random.uniform(0, min(abs(xmin), abs(xmax)), 2 ** (R - 1))
    centers_init = np.concatenate((-random_array, random_array))
    thresholds_init = np.zeros(len(centers_init) - 1)
    for i in range(len(centers_init) - 1):
        thresholds_init[i] = 0.5 * (centers_init[i] + centers_init[i + 1])

    centers_update = np.copy(np.sort(centers_init))
    thresholds_update = np.copy(np.sort(thresholds_init))
    for i in range(iterations_kmeans):
        integ_nom = quad(lambda x: x ** (M+1) * pdf_gennorm(x, a, m, b), -np.inf, thresholds_update[0])[0]
        integ_denom = quad(lambda x: x ** M * pdf_gennorm(x, a, m, b), -np.inf, thresholds_update[0])[0]
        #centers_update[0] = np.divide(integ_nom, integ_denom)
        centers_update[0] = np.divide(integ_nom, (integ_denom + 1e-7))
        for j in range(len(centers_init) - 2):          # j=7
            integ_nom_update = \
            quad(lambda x: x ** (M+1) * pdf_gennorm(x, a, m, b), thresholds_update[j], thresholds_update[j + 1])[0]
            integ_denom_update = \
            quad(lambda x: x ** M * pdf_gennorm(x, a, m, b), thresholds_update[j], thresholds_update[j + 1])[0]
            ###
            centers_update[j + 1] = np.divide(integ_nom_update, (integ_denom_update + 1e-7))
            #if (np.abs(integ_nom_update)<0.0000000001) or (np.abs(integ_denom_update)<0.0000000001):
            #    centers_update[j + 1] = 0
            #else:
            #    centers_update[j + 1] = np.divide(integ_nom_update, integ_denom_update)  # integ_denom_update+eplison
        integ_nom_final = \
        quad(lambda x: x ** (M+1) * pdf_gennorm(x, a, m, b), thresholds_update[len(thresholds_update) - 1], np.inf)[0]
        integ_denom_final = \
        quad(lambda x: x ** M * pdf_gennorm(x, a, m, b), thresholds_update[len(thresholds_update) - 1], np.inf)[0]
        #centers_update[len(centers_update) - 1] = np.divide(integ_nom_final, integ_denom_final)
        centers_update[len(centers_update) - 1] = np.divide(integ_nom_final, (integ_denom_final+ 1e-7))
        for j in range(len(thresholds_update)):
            thresholds_update[j] = 0.5 * (centers_update[j] + centers_update[j + 1])
    #thresholds_final = np.divide(np.subtract(thresholds_update,thresholds_update[::-1]),2)
    #centers_final = np.divide(np.subtract(centers_update,centers_update[::-1]),2)
    return np.add(np.multiply(thresholds_update,np.sqrt(s)),mu), np.add(np.multiply(centers_update,np.sqrt(s)),mu)



def pdf_doubleweibull(x, a, m, scale=1):
  return stats.dweibull.pdf(x,a,m,scale)

def update_centers_magnitude_distance_weibull(data, R, M, iterations_kmeans):
    #TODO: allow change of m
    #M = 1
    #data = data.cpu().numpy()
    mu = np.mean(data)
    s = np.var(data)
    data_normalized = np.divide(np.subtract(data,mu),np.sqrt(s))
    a, m, b = stats.gennorm.fit(data_normalized)

    xmin, xmax = min(data_normalized), max(data_normalized)
    random_array = np.random.uniform(0, min(abs(xmin), abs(xmax)), 2 ** (R - 1))
    centers_init = np.concatenate((-random_array, random_array))
    thresholds_init = np.zeros(len(centers_init) - 1)
    for i in range(len(centers_init) - 1):
        thresholds_init[i] = 0.5 * (centers_init[i] + centers_init[i + 1])

    centers_update = np.copy(np.sort(centers_init))
    thresholds_update = np.copy(np.sort(thresholds_init))
    for i in range(iterations_kmeans):
        integ_nom = quad(lambda x: x ** (M+1) * pdf_doubleweibull(x, a, m, b), -np.inf, thresholds_update[0])[0]
        integ_denom = quad(lambda x: x ** M * pdf_doubleweibull(x, a, m, b), -np.inf, thresholds_update[0])[0]
        #centers_update[0] = np.divide(integ_nom, integ_denom)
        centers_update[0] = np.divide(integ_nom, (integ_denom + 1e-7))
        for j in range(len(centers_init) - 2):          # j=7
            integ_nom_update = \
            quad(lambda x: x ** (M+1) * pdf_doubleweibull(x, a, m, b), thresholds_update[j], thresholds_update[j + 1])[0]
            integ_denom_update = \
            quad(lambda x: x ** M * pdf_doubleweibull(x, a, m, b), thresholds_update[j], thresholds_update[j + 1])[0]
            ###
            centers_update[j + 1] = np.divide(integ_nom_update, (integ_denom_update + 1e-7))
            #if (np.abs(integ_nom_update)<0.0000000001) or (np.abs(integ_denom_update)<0.0000000001):
            #    centers_update[j + 1] = 0
            #else:
            #    centers_update[j + 1] = np.divide(integ_nom_update, integ_denom_update)  # integ_denom_update+eplison
        integ_nom_final = \
        quad(lambda x: x ** (M+1) * pdf_doubleweibull(x, a, m, b), thresholds_update[len(thresholds_update) - 1], np.inf)[0]
        integ_denom_final = \
        quad(lambda x: x ** M * pdf_doubleweibull(x, a, m, b), thresholds_update[len(thresholds_update) - 1], np.inf)[0]
        #centers_update[len(centers_update) - 1] = np.divide(integ_nom_final, integ_denom_final)
        centers_update[len(centers_update) - 1] = np.divide(integ_nom_final, (integ_denom_final+ 1e-7))
        for j in range(len(thresholds_update)):
            thresholds_update[j] = 0.5 * (centers_update[j] + centers_update[j + 1])
    #thresholds_final = np.divide(np.subtract(thresholds_update,thresholds_update[::-1]),2)
    #centers_final = np.divide(np.subtract(centers_update,centers_update[::-1]),2)
    return np.add(np.multiply(thresholds_update,np.sqrt(s)),mu), np.add(np.multiply(centers_update,np.sqrt(s)),mu)

















def quantization(gradients, args):

    compression_type = args.compression_type
    rate = args.R
    M_value = args.M

        
    gradient_shape = gradients.shape
    gradient_vec = gradients.reshape((-1))
    non_zero_indices = torch.nonzero(gradient_vec).reshape((-1)).cpu().numpy()
    gradient_vec = gradient_vec.cpu().numpy()

    seq = gradient_vec[non_zero_indices]

    if  compression_type=="no_compression":
        seq_dec = seq

    elif compression_type=="uniform":

        seq_enc, uni_max, uni_min= compress_uni_scalar(seq, rate)
        seq_dec = decompress_uni_scalar(seq_enc, rate, uni_max, uni_min)

    elif compression_type=="GenNorm":

        thresholds, quantization_centers = update_centers_magnitude_distance(data=seq, R=rate, M=M_value, iterations_kmeans=50)
        thresholds_sorted = np.sort(thresholds)
        labels = np.digitize(seq,thresholds_sorted)
        index_labels_false = np.where(labels == 2**rate)
        labels[index_labels_false] = 2**rate-1
        seq_dec = quantization_centers[labels]

    elif compression_type=="Weibull":

        thresholds, quantization_centers = update_centers_magnitude_distance_weibull(data=seq, R=rate, M=M_value, iterations_kmeans=100)
        thresholds_sorted = np.sort(thresholds)
        labels = np.digitize(seq,thresholds_sorted)
        index_labels_false = np.where(labels == 2**rate)
        labels[index_labels_false] = 2**rate-1
        seq_dec = quantization_centers[labels]


    np.put(gradient_vec, non_zero_indices, seq_dec)

    vec_tensor = torch.from_numpy(gradient_vec).float().to(gradients.get_device())
    gradients_out = vec_tensor.reshape(gradient_shape)


    return gradients_out

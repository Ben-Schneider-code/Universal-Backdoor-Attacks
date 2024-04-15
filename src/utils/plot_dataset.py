import matplotlib.pyplot as plt
import numpy as np

def latent_graphs(latent_space_in_basis):
    for i in range(latent_space_in_basis.shape[1]):
        arr = latent_space_in_basis[:, latent_space_in_basis.shape[1] - 1 - i]
        numpy_array_histogram(arr.cpu().numpy())
def numpy_array_histogram(array):
    plt.hist(array, bins='auto')
    plt.show()

def numpy_array_dual_histogram(array1,array2, eigen_vector):
    plt.title(eigen_vector)
    plt.hist(array1, bins='auto')
    plt.hist(array2, bins='auto')
    plt.show()

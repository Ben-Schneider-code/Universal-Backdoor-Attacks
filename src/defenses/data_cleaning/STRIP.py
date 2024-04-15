__name__ = 'strip'
__category__ = 'transformer'
__input_type__ = 'image'
__defaults_form__ = {
    'number_of_images': {
        'pretty_name': 'Number of images',
        'default_value': [100],
        'info': 'The number of images represents the number of clean images the defence can use to calculate the entropy. These images are overlayed to see how it affects the predictions of the classifier. The more images, the more accurate the result is but, the longer it takes to compute. This defence is effective with 100 or more images.'
    }
}
__defaults_dropdown__ = {
}
__defaults_range__ = {
    'false_acceptance_rate': {
        'pretty_name': 'False acceptance rate',
        'minimum': 0.0,
        'maximum': 1.0,
        'default_value': [0.01],
        'info': 'False acceptance rate. From this the threshold for acceptance is calculated. The default is 0.01.'
    },
}
__link__ = 'https://arxiv.org/pdf/1902.06531.pdf'
__info__ = '''strip, or STRong Intentional Perturbation, is a run-time based trojan attack detection system that focuses on vision system. 
strip intentionally perturbs the incoming input, for instance, by superimposing various image patterns and observing the randomness of predicted 
classes for perturbed inputs from a given deployed model—malicious or benign. A low entropy in predicted classes violates the input-dependence 
property of a benign model. It implies the presence of a malicious input—a characteristic of a trojaned input.'''.replace(
    '\n', '')

import torch
import cv2
from scipy.stats import norm
from tqdm import tqdm
import numpy as np


def run(clean_classifier, clean_data, num_imgs, fpr=0.01):
    '''Runs the strip defence

    Parameters
    ----------
    clean_classifier :
        Classifier that has not been tampered with, i.e. is clean
    test_data :
        Data that the clean classifier will be validated on as a tuple with (inputs, labels)
    execution_history :
        Dictionary with paths of attacks/defences taken to achieve classifiers, if any
    defence_params :
        Dictionary with the parameters for the defence (one value per parameter)

    Returns
    ----------
    Returns the updated execution history dictionary
    '''
    print('Instantiating a strip defence.')

    clean_classifier.predict = clean_classifier.forward
    clean_classifier.nb_classes = 1000
    clean_data = (clean_data[0].numpy(), clean_data[1].numpy())

    return STRIP_ViTA(clean_classifier, clean_data, number_of_samples=num_imgs, far=fpr)


class STRIP_ViTA():
    def __init__(self, model, clean_test_data, number_of_samples=100, far=0.01):
        """
            strip-ViTA defence class

        Parameters
        ----------
        model : classifier for audio
            IMPORTANT : it contains .predict() method
        clean_test_data : (datapoints, labels)
            clean dataset
        number_of_samples : int, optional
            number of samples used for calculating entropy. The default is 100.
        far : float, optional
            False acceptance rate. From this the threshold for acceptance is calculated. The default is 0.01.

        Returns
        -------
        None.

        """
        self.model = model
        self.clean_test_data = clean_test_data

        self.number_of_samples = min(number_of_samples, len(clean_test_data[0]))
        self.far = far

        self.entropy_bb = None

        self.defence()

    def superimpose(self, background, overlay):
        """
        Combines 2 data_cleaning points

        Parameters
        ----------
        background : datapoint
            Datapoint from clean test dataset.
        overlay : datapoint
            Datapoint generated from noise.

        Returns
        -------
        datapoint
            Weighted sum of 2 datapoints

        """
        # make sure the data_cleaning is correct type
        background = background.astype(np.float32)
        overlay = overlay.astype(np.float32)

        # return background+overlay
        return cv2.addWeighted(background, 1, overlay, 1, 0)

    def entropyCal(self, background, n):
        """
        Calculates entropy of a single datapoint

        Parameters
        ----------
        background : datapoint
            Datapoint from test dataset.
        n : int
            number of samples the function takes.

        Returns
        -------
        EntropySum : float
            Entropy

        """
        x1_add = [0] * n

        x_test = self.clean_test_data[0]

        # choose n overlay indexes
        index_overlay = np.random.randint(0, len(x_test), n)

        # do superimpose n times
        for i in range(n):
            x1_add[i] = self.superimpose(background, x_test[index_overlay[i]])

        py1_add = self.model.predict(torch.tensor(np.array(x1_add)).cuda()).detach().cpu().numpy()
        EntropySum = -np.nansum(py1_add * np.log2(py1_add))
        return EntropySum

    def defence(self):
        """
        Initializes Strip-Vita defence

        Returns
        -------
        None.

        """
        x_test = self.clean_test_data[0]

        n_test = len(x_test)
        n_sample = self.number_of_samples

        entropy_bb = [0] * n_test  # entropy for benign + benign

        # calculate entropy for clean test set
        for j in tqdm(range(n_test), desc="Entropy:benign_benign"):
            x_background = x_test[j]
            entropy_bb[j] = self.entropyCal(x_background, n_sample)

        self.entropy_bb = np.array(entropy_bb) / n_sample

        mean_entropy, std_entropy = norm.fit(self.entropy_bb)

        self.entropy_treshold = norm.ppf(self.far, loc=mean_entropy, scale=std_entropy)

    def predict(self, x_test_data):
        """
        Predicts class for the input data_cleaning.
        Also if the method find out that the datapoint is very likely to be poisoned it doesn't predict anything.

        Parameters
        ----------
        x_test_data : array of datapoints / single datapoint
            input test data_cleaning.

        Returns
        -------
        predictions : array -> with length (x_test_data) with elements : array (number_of_classes,)
            There are two possible types of output:
                1. entropy(data_cleaning) < threshold: append np.zeros(number_of_classes)
                2. entropy(data_cleaning) >= threshold: uses PytorchClassifier.predict()

        """

        trojan_x_test = x_test_data
        # print(np.array(x_test_data).shape)
        if not (np.array(x_test_data).shape == 3):
            trojan_x_test = list(trojan_x_test)

        n_test = len(trojan_x_test)
        n_sample = self.number_of_samples

        entropy_tb = [0] * n_test  # entropy for trojan + benign
        predictions = list()
        # calculate entropy for clean test set
        for j in tqdm(range(n_test), desc="Entropy:benign_benign"):
            x_background = trojan_x_test[j]
            entropy_tb[j] = self.entropyCal(x_background, n_sample)

            entropy_tb[j] = entropy_tb[j] / n_sample

            if entropy_tb[j] <= self.entropy_treshold:  # is poisoned
                predictions.append(1)
            else:  # is not poisoned
                predictions.append(0)

        return predictions

    def get_predictions(self, x_poison_data):
        """
        Applies Strip-Vita defence on poisoned dataset

        Parameters
        ----------
        x_poison_data : array of datapoints
            Poisoned input dataset.

        Returns
        -------
        posion_predictions : array of predictions
            Predictions for poisoned dataset.
        clean_predictions : array of predictions
            Predictions for clean dataset.

        """
        posion_predictions = self.predict(x_poison_data)
        # clean_predictions = self.predict(self.clean_test_data[0][:self.number_of_samples])

        return posion_predictions  # , clean_predictions

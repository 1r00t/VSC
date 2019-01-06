import numpy as np
# from unsupervised_learning.data_utils import load_CIFAR10
import matplotlib.pyplot as plt
import sys, os, pickle


#########################################################################
#
#        Helper functions
#
#########################################################################

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        datadict = u.load()
        # datadict = pickle.load(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def visualize_images(X_train, y_train):
    """
    Visualizes some examples from the dataset showing a few examples of training images from each class.
    :return:
    """
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)
    samples_per_class = 7
    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(y_train == y)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X_train[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.show()


#########################################################################
#
#        KnnNearestNeighbor
#
#########################################################################


class KnnNearestNeighbor(object):
    def __init__(self):


    def train(self, X, y):
        """ X is N x D where each row is an example. Y is 1-dimension of size N """
        # the nearest neighbor classifier simply remembers all the training data
        self.Xtr = X
        self.ytr = y

    def predict(self, X, k=1):
        """ X is N x D where each row is an example we wish to predict label for """
        num_test = X.shape[0]
        # lets make sure that the output type matches the input type
        ypred = np.zeros(num_test, dtype = self.ytr.dtype)

        # TODO
        # 1. loop over all test rows
        # 2. compute the distances of X to self.Xtr
        # 3. find the k min distances and its indices
        # 4. use indices to lookup the label in self.ytr[]
        # 5. count the majority of labels. Note: you could insert the found labels
        #    in a np.histogram with bins=[0,...,9] and find the bin with the maximum
        #    that should be the majority of the labels
        #    fill ypred[i] with the correct label

        return ypred



if __name__ == "__main__":

    # TODO Load the raw CIFAR-10 data.

    # As a sanity check, we print out the size of the training and test data.
    print('Training data shape: ', X_train.shape)
    print('Training labels shape: ', y_train.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)

    # Subsample the data for more efficient code execution in this exercise
    # use between 1000 / 100 und 5000/500 as train/test sizes

    # TODO sample 1000-5000 training images and 100-500 disjunct test images
    #

    # if you want to see some example images you can visualize them
    # visualize_images(X_train, y_train)


    # TODO: Reshape the image data into rows
    # size of the X_train, X_test matrix should be
    # (1000, 3072) (100, 3072)


    # find hyperparameters that work best on the validation set
    validation_accuracies = []
    k_array = [1, 2, 5, 10, 20, 50]
    for k in k_array:

        # use a particular value of k and evaluation on validation data
        nn = KnnNearestNeighbor()
        nn.train(X_train, y_train)
        # here we assume a modified NearestNeighbor class that can take a k as input
        y_predict = nn.predict(X_test, k = k)

        # TODO implement accuracy by calculating the number of correct labels divided by the number of the
        # whole test set -> y_predict and y_test
        print('accuracy: %f' % (acc,))

        # keep track of what k works best
        validation_accuracies.append(acc)

    print(validation_accuracies[1])
    plt.plot(k_array, validation_accuracies)
    plt.xlabel('k')
    plt.ylim((0,1.0))
    plt.ylabel('Accuracy')
    plt.show()

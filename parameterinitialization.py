import numpy as np 
import matplotlib.pyplot as plt
from init_util import *
from init_methods import *
import sklearn
import sklearn.datasets


def main():
    train_X, train_Y, test_X, test_Y = load_dataset()
    parameters = model(train_X, train_Y, initialization = "he")
    print ("On the train set:")
    predictions_train = predict(train_X, train_Y, parameters)
    print ("On the test set:")
    predictions_test = predict(test_X, test_Y, parameters)
    plt.title("Model with large random initialization")
    axes = plt.gca()
    axes.set_xlim([-1.5,1.5])
    axes.set_ylim([-1.5,1.5])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)


if __name__ == "__main__":
    main()



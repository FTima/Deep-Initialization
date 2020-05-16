from regularization_methods import *
from reg_utils import *





def main():
    train_X, train_Y, test_X, test_Y = load_2D_dataset()
    # base model
    parameters = model(train_X, train_Y)
    print ("On the training set:")
    predictions_train = predict(train_X, train_Y, parameters)
    print ("On the test set:")
    predictions_test = predict(test_X, test_Y, parameters)

    plt.title("Model without regularization")
    axes = plt.gca()
    axes.set_xlim([-0.75,0.40])
    axes.set_ylim([-0.75,0.65])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)


    #L2 Regularization
    parameters = model(train_X, train_Y, lambd = 0.7)
    print ("On the train set:")
    predictions_train = predict(train_X, train_Y, parameters)
    print ("On the test set:")

    predictions_test = predict(test_X, test_Y, parameters)
    plt.title("Model with L2-regularization")
    axes = plt.gca()
    axes.set_xlim([-0.75,0.40])
    axes.set_ylim([-0.75,0.65])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)


    #Drop out
    parameters = model(train_X, train_Y, keep_prob = 0.86, learning_rate = 0.3)

    print ("On the train set:")
    predictions_train = predict(train_X, train_Y, parameters)
    print ("On the test set:")
    predictions_test = predict(test_X, test_Y, parameters)

    plt.title("Model with dropout")
    axes = plt.gca()
    axes.set_xlim([-0.75,0.40])
    axes.set_ylim([-0.75,0.65])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
 

if __name__ == "__main__":
    main()

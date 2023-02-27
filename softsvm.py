import numpy as np
from cvxopt import solvers, matrix, spdiag
import matplotlib.pyplot as plt


# todo: complete the following functions, you may add auxiliary functions or define class to help you
def softsvm(l, trainX: np.array, trainy: np.array):
    """

    :param l: the parameter lambda of the soft SVM algorithm
    :param trainX: numpy array of size (m, d) containing the training sample
    :param trainy: numpy array of size (m, 1) containing the labels of the training sample
    :return: linear predictor w, a numpy array of size (d, 1)
    """
    m, d = trainX.shape

    H = 2 * l * spdiag(list(np.concatenate((np.ones(d), np.zeros(m)), axis=0)))

    average_coefficient = 1 / m
    u = matrix(np.concatenate((np.zeros(d), average_coefficient * np.ones(m)), axis=0), (m + d, 1))

    v = matrix(np.concatenate((np.zeros(m), np.ones(m)), axis=0), (2 * m, 1))

    # construct A from blocks
    A1 = np.zeros((m, d))
    A2 = np.identity(m)
    A3 = np.matmul(np.diag(trainy), trainX)
    A4 = A2

    A = matrix(np.block([[A1, A2],[A3, A4]]))

    sol = solvers.qp(H, u, -A, -v)

    return np.array(sol["x"])[: d]


def q2_small_sample(plot_error_bars=True):
    # load question 2 data
    data = np.load('EX2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainY = data['Ytrain']
    testY = data['Ytest']

    # don't print results during QP
    solvers.options['show_progress'] = False

    # sample size
    m = 100

    # train_mean_errors and test_mean_errors are the y-axis values
    train_mean_errors, train_max_errors, train_min_errors = [], [], []
    test_mean_errors, test_max_errors, test_min_errors = [], [], []

    x_axis = range(1, 11)

    for n in x_axis:  # for lambda in 10^n : 1 <= n <= 10
        Lambda = 10 ** n

        train_errors_of_current_lambda, test_errors_of_current_lambda = [], []  # will be averaged in the end
        max_train_error, min_train_error = 0, 1
        max_test_error, min_test_error = 0, 1

        for i in range(10):
            # Get a random m training examples from the training set
            indices = np.random.permutation(trainX.shape[0])
            training_samples_X = trainX[indices[:m]]
            training_samples_Y = trainY[indices[:m]]

            # run the soft-svm algorithm
            w = softsvm(Lambda, training_samples_X, training_samples_Y)

            # use w to label examples
            def predict_label(test_example):
                return np.sign(test_example @ w)[0]  # vector multiplication returns a vector of shape (1,1)

            # calculate train mean error at current iteration
            training_sample_predictions = [predict_label(training_sample) for training_sample in training_samples_X]
            train_error = np.mean(training_sample_predictions != training_samples_Y)
            train_errors_of_current_lambda.append(train_error)

            # data for train error bars - check max and min errors
            max_train_error = max(max_train_error, train_error)
            min_train_error = min(min_train_error, train_error)

            # calculate test mean error at current iteration
            testing_sample_predictions = [predict_label(testing_sample) for testing_sample in testX]
            test_error = np.mean(testing_sample_predictions != testY)
            test_errors_of_current_lambda.append(test_error)

            # data for test error bars - check max and min errors
            max_test_error = max(max_test_error, test_error)
            min_test_error = min(min_test_error, test_error)

        # calculate average train mean error of the 10 iterations.
        # mark the mean, max, min train errors
        average_train_mean_error_of_10_runs = sum(train_errors_of_current_lambda) / 10
        train_mean_errors.append(average_train_mean_error_of_10_runs)

        train_max_errors.append(max_train_error - average_train_mean_error_of_10_runs)
        train_min_errors.append(average_train_mean_error_of_10_runs - min_train_error)

        # calculate average test mean error of the 10 iterations.
        # mark the mean, max, min test errors
        average_test_mean_error_of_10_runs = sum(test_errors_of_current_lambda) / 10
        test_mean_errors.append(average_test_mean_error_of_10_runs)

        test_max_errors.append(max_test_error - average_test_mean_error_of_10_runs)
        test_min_errors.append(average_test_mean_error_of_10_runs - min_test_error)

    train_max_and_min_errors = [train_min_errors, train_max_errors]
    test_max_and_min_errors = [test_min_errors, test_max_errors]

    plt.plot(x_axis, train_mean_errors, 'c', label="small sample - train mean error")
    plt.plot(x_axis, test_mean_errors, 'k', label="small sample - test eman error")

    # add error bars
    if plot_error_bars:
        plt.errorbar(x_axis, train_mean_errors, train_max_and_min_errors, fmt='o', ecolor='red')
        plt.errorbar(x_axis, test_mean_errors, test_max_and_min_errors, fmt='o', ecolor='red')

    plt.title("Train and test mean errors")
    plt.xlabel("n")
    plt.ylabel("Mean error")
    plt.legend(loc="upper left")


def q2_large_sample():
    # load question 2 data
    data = np.load('EX2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainY = data['Ytrain']
    testY = data['Ytest']

    # don't print results during QP
    solvers.options['show_progress'] = False

    # sample size
    m = 1_000

    # train_mean_errors and test_mean_errors are the y-axis values
    train_mean_errors, test_mean_errors = [], []

    n_values = [1, 3, 5, 8]

    for n in n_values:
        Lambda = 10 ** n

        # Get a random m training examples from the training set
        indices = np.random.permutation(trainX.shape[0])
        training_samples_X = trainX[indices[:m]]
        training_samples_Y = trainY[indices[:m]]

        # run the soft-svm algorithm
        w = softsvm(Lambda, training_samples_X, training_samples_Y)

        # use w to label examples
        def predict_label(test_example):
            return np.sign(test_example @ w)[0]  # vector multiplication returns a vector of shape (1,1)

        # calculate train mean error
        training_sample_predictions = [predict_label(training_sample) for training_sample in training_samples_X]
        current_train_mean_error = np.mean(training_sample_predictions != training_samples_Y)
        train_mean_errors.append(current_train_mean_error)

        # calculate test mean error
        testing_sample_predictions = [predict_label(testing_sample) for testing_sample in testX]
        current_test_mean_error = np.mean(testing_sample_predictions != testY)
        test_mean_errors.append(current_test_mean_error)

    x_axis = n_values
    plt.plot(x_axis, train_mean_errors, 'o', label="large sample - train mean error")  # 'o' means no line
    plt.plot(x_axis, test_mean_errors, 'o', label="large sample - test mean error")

    plt.title("Train and test mean errors")
    plt.xlabel("n")
    plt.ylabel("Mean error")
    plt.legend(loc="upper left")


def q2_a():
    q2_small_sample()
    plt.show()


def q2_b():
    q2_small_sample(False)
    q2_large_sample()
    plt.show()


def q2_b_only_large_sample():
    q2_large_sample()
    plt.show()


def simple_test():
    # load question 2 data
    data = np.load('EX2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    m = 100
    d = trainX.shape[1]

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    # run the softsvm algorithm
    w = softsvm(10, _trainX, _trainy)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(w, np.ndarray), "The output of the function softsvm should be a numpy array"
    assert w.shape[0] == d and w.shape[1] == 1, f"The shape of the output should be ({d}, 1)"

    # get a random example from the test set, and classify it
    i = np.random.randint(0, testX.shape[0])
    predicty = np.sign(testX[i] @ w)

    # this line should print the classification of the i'th test sample (1 or -1).
    print(f"The {i}'th test sample was classified as {predicty}")


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    simple_test()
    # here you may add any code that uses the above functions to solve question 2
    # q2_a()
    # q2_b_only_large_sample()
    # q2_b()

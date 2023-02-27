import matplotlib.pyplot as plt
import numpy as np
from cvxopt import solvers, matrix

from softsvm import softsvm


# todo: complete the following functions, you may add auxiliary functions or define class to help you
def softsvmpoly(l: float, k: int, trainX: np.array, trainy: np.array):
    """

    :param l: the parameter lambda of the soft SVM algorithm
    :param k: the bandwidth parameter sigma of the RBF kernel.
    :param trainX: numpy array of size (m, d) containing the training sample
    :param trainy: numpy array of size (m, 1) containing the labels of the training sample
    :return: numpy array of size (m, 1) which describes the coefficients found by the algorithm
    """
    m, d = trainX.shape

    def K(x_1, x_2):
        return (1 + np.dot(x_1, x_2)) ** k

    G = np.array([[K(x_i, x) for x_i in trainX] for x in trainX])

    # u = (0,...,0,1/m,...,1/m)
    average_coefficient = 1 / m
    u = matrix(np.concatenate((np.zeros(m), average_coefficient * np.ones(m)), axis=0), (2 * m, 1))

    # v = (0,...,0,1,...,1)
    v = matrix(np.concatenate((np.zeros(m), np.ones(m)), axis=0), (2 * m, 1))

    empty_matrix = np.zeros((m, m))

    # defining H
    H = np.block([[2 * l * G, empty_matrix], [empty_matrix, empty_matrix]])

    epsilon = 0.001
    epsilon_matrix = epsilon * np.identity(2 * m)

    H = np.add(H, epsilon_matrix)
    H = matrix(H)

    # defining A
    A1 = np.zeros((m, m))
    A2 = np.identity(m)
    A3 = np.matmul(np.diag(trainy), G)
    A4 = A2

    A = matrix(np.block([[A1, A2], [A3, A4]]))

    solvers.options['show_progress'] = False  # don't print results during QP
    sol = solvers.qp(H, u, -A, -v)

    return np.array(sol["x"])[: m]


def find_labels_with_kernel_svm(Lambda, k, train_X: np.ndarray, train_Y: np.ndarray, test_X: np.ndarray):
    alpha = softsvmpoly(Lambda, k, train_X, train_Y)
    m = alpha.shape[0]

    def predict_label(example):
        def K(x1, x2):  # polynomial kernel
            return (1 + np.dot(x1, x2)) ** k

        # represents the row corresponding to the example in the Gram matrix
        polynomial_kernel_vector = np.reshape([K(x, example) for x in train_X], (1, m))

        return np.sign((polynomial_kernel_vector @ alpha))[0][0]

    return [predict_label(example) for example in test_X]


def q4_a():
    # load data
    data = np.load('ex2q4_data.npz')
    train_X = data['Xtrain']
    train_Y = data['Ytrain']

    m = train_X.shape[0]

    # indices 0,1 are X's features, index 2 is Y
    training_sample_with_labels = np.concatenate((train_X, np.reshape(train_Y, (m, 1))), axis=1)

    x_axis_positive_label, y_axis_positive_label = \
        zip(*[(training_example[0], training_example[1]) for training_example in training_sample_with_labels if
              training_example[2] == 1])

    x_axis_negative_label, y_axis_negative_label = \
        zip(*[(training_example[0], training_example[1]) for training_example in training_sample_with_labels if
              training_example[2] == -1])

    plt.scatter(x_axis_negative_label, y_axis_negative_label, color="red", label="-1")
    plt.scatter(x_axis_positive_label, y_axis_positive_label, color="blue", label="+1")

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15))

    plt.show()


def q4_b_kernel_svm():
    # load data
    data = np.load('ex2q4_data.npz')
    train_X = data['Xtrain']
    test_X = data['Xtest']
    train_Y = data['Ytrain']
    test_Y = data['Ytest']

    number_of_cross_validations = 5  # we perform k-cross validation on the training set
    holdout_set_size = len(train_X) / number_of_cross_validations

    lambda_and_k_options = [(1, 2), (1, 5), (1, 8), (10, 2), (10, 5), (10, 8), (100, 2), (100, 5), (100, 8)]
    average_validation_errors = []

    for Lambda, k in lambda_and_k_options:
        total_validation_error = 0  # will be averaged (divided by number of cross validations)

        for i in range(number_of_cross_validations):
            start_of_holdout_set = int(i * holdout_set_size)
            end_of_holdout_set = int(start_of_holdout_set + holdout_set_size)

            holdout_set_X = train_X[start_of_holdout_set: end_of_holdout_set]
            holdout_set_Y = train_Y[start_of_holdout_set: end_of_holdout_set]

            training_set_X = np.concatenate((train_X[:start_of_holdout_set], train_X[end_of_holdout_set:]), axis=0)
            training_set_Y = np.concatenate((train_Y[:start_of_holdout_set], train_Y[end_of_holdout_set:]), axis=0)

            # predict labels for test sample
            holdout_set_predictions = \
                find_labels_with_kernel_svm(Lambda, k, training_set_X, training_set_Y, holdout_set_X)

            # calculate validation error
            validation_error = np.mean(holdout_set_predictions != holdout_set_Y)
            total_validation_error += validation_error

        average_validation_error = total_validation_error / number_of_cross_validations
        average_validation_errors.append(average_validation_error)

        print(f"Lambda = {Lambda}, K = {k}, average validation error = {average_validation_error}")

    best_lambda, best_k = lambda_and_k_options[np.argmin(average_validation_errors)]
    print(f"best lambda is {best_lambda}, best k is {best_k}")

    test_predictions = find_labels_with_kernel_svm(best_lambda, best_k, train_X, train_Y, test_X)
    test_error = np.mean(test_predictions != test_Y)

    print(f"test error is {test_error}")


def q4_b_linear_svm():
    # load data
    data = np.load('ex2q4_data.npz')
    train_X = data['Xtrain']
    test_X = data['Xtest']
    train_Y = data['Ytrain']
    test_Y = data['Ytest']

    # don't print results during QP
    solvers.options['show_progress'] = False

    number_of_cross_validations = 5  # we perform k-cross validation on the training set
    holdout_set_size = len(train_X) / number_of_cross_validations

    Lambda_options, average_validation_errors = [1, 10, 100], []
    for Lambda in Lambda_options:
        total_validation_error = 0  # will be averaged (divided by number of cross validations)

        for i in range(number_of_cross_validations):
            start_of_holdout_set = int(i * holdout_set_size)
            end_of_holdout_set = int(start_of_holdout_set + holdout_set_size)

            holdout_set_X = train_X[start_of_holdout_set: end_of_holdout_set]
            holdout_set_Y = train_Y[start_of_holdout_set: end_of_holdout_set]

            training_set_X = np.concatenate((train_X[:start_of_holdout_set], train_X[end_of_holdout_set:]), axis=0)
            training_set_Y = np.concatenate((train_Y[:start_of_holdout_set], train_Y[end_of_holdout_set:]), axis=0)

            w = softsvm(Lambda, training_set_X, training_set_Y)

            # check validation error
            def predict_label(example):
                return np.sign(example @ w)[0]  # vector multiplication returns a vector of shape (1,1)

            holdout_set_predictions = [predict_label(example) for example in holdout_set_X]

            validation_error = np.mean(holdout_set_predictions != holdout_set_Y)
            total_validation_error += validation_error

        average_validation_error = total_validation_error / number_of_cross_validations
        average_validation_errors.append(average_validation_error)
        print(f"Lamda = {Lambda}, average validation error = {average_validation_error}")

    best_lambda = Lambda_options[np.argmin(average_validation_errors)]
    print(f"best lambda is {best_lambda}")

    w = softsvm(best_lambda, train_X, train_Y)

    def predict_label(example):
        return np.sign(example @ w)[0]  # vector multiplication returns a vector of shape (1,1)

    test_predictions = [predict_label(example) for example in test_X]

    test_error = np.mean(test_predictions != test_Y)

    print(f"test error is {test_error}")


def q4_e():
    # load data
    data = np.load('ex2q4_data.npz')
    train_X = data['Xtrain']
    train_Y = data['Ytrain']

    m = train_X.shape[0]

    Lambda = 1  # stated in forum that also allowed (instead of Lambda = 100)

    for k in [3, 5, 8]:
        predictions = find_labels_with_kernel_svm(Lambda, k, train_X, train_Y, train_X)

        # indices 0,1 are X's features, index 2 is predicted Y
        training_sample_with_labels = np.concatenate((train_X, np.reshape(predictions, (m, 1))), axis=1)

        x_axis_positive_label, y_axis_positive_label = \
            zip(*[(training_example[0], training_example[1]) for training_example in training_sample_with_labels if
                  training_example[2] == 1])

        x_axis_negative_label, y_axis_negative_label = \
            zip(*[(training_example[0], training_example[1]) for training_example in training_sample_with_labels if
                  training_example[2] == -1])

        plt.figure()  # create new graph
        plt.scatter(x_axis_negative_label, y_axis_negative_label, color="red", label="-1")
        plt.scatter(x_axis_positive_label, y_axis_positive_label, color="blue", label="+1")

        # plt.legend(loc='upper left', bbox_to_anchor=(0.5, 1.15))
        plt.title(f"k = {k}")

    plt.show()


def q4_f(*question_parts):
    data = np.load('ex2q4_data.npz')
    train_X = data['Xtrain']
    test_X = data['Xtest']
    train_Y = data['Ytrain']

    Lambda, k = 1, 5
    m, d = train_X.shape

    alpha = softsvmpoly(Lambda, k, train_X, train_Y)

    # returns I^d_k (all possible monomials with length d of degree <= k)
    def generate_all_possible_monomials():
        output = []

        # auxiliary recursive function
        def generate_all_possible_monomials2(current_power, total_power, current_index, monomial):
            if current_power + total_power > k:
                return

            elif current_index == d:
                output.append(monomial)
                return

            new_monomial = monomial.copy()
            new_monomial.append(current_power)

            generate_all_possible_monomials2(current_power + 1, total_power, current_index, monomial)
            generate_all_possible_monomials2(0, total_power + current_power, current_index + 1, new_monomial)

        generate_all_possible_monomials2(0, 0, 0, [])
        return output

    all_possible_monomials = generate_all_possible_monomials()

    # returns n! / (t(1)! * ... * t(d)!)
    def calculate_multinomial_coefficient(n, monomial):
        def factorial(number: int) -> int:
            return int(np.prod(range(1, number + 1)))

        return int(factorial(n) / np.prod([factorial(number) for number in monomial]))

    def psi(x):
        # calculates the coordinate in psi(x) that corresponds with the monomial t
        # formula: sqrt(B(k, t)) * (x(1) ^ t(i)) * ... * (x(d) ^ t(d))
        def psi_i(t):
            # calculate B(k, t)
            B_k_t = calculate_multinomial_coefficient(k, t)

            # calculate the product (x(1) ^ t(i)) * ... * (x(d) ^ t(d))
            product_of_coordinates = np.prod([x_i ** t_i for x_i, t_i in zip(x, t)])
            return np.sqrt(B_k_t) * product_of_coordinates

        return np.array([psi_i(t) for t in all_possible_monomials])

    # represents (psi(x_1), ..., psi(x_m)).
    psi_vector = np.array([psi(x) for x in train_X])

    # multiply row i by alpha_i
    psi_vector_multiplied_by_alpha = np.matmul(np.diag(alpha.transpose()[0]), psi_vector)

    # w is the sum of all rows
    w = psi_vector_multiplied_by_alpha.sum(axis=0)

    if "ii" in question_parts:
        print(f"w = \n{w}\n ")

    if "iii" in question_parts:
        print("<w, psi(x)> = ")
        print(" +\n".join(str(w_i * np.sqrt(calculate_multinomial_coefficient(k, t))) +
                          " * x(1) ^ {0} * x(2) ^ {1}".format(t[0], t[1]) for w_i, t in zip(w, all_possible_monomials)))

    if "iv" in question_parts:
        merged_training_and_test_X = np.concatenate((train_X, test_X), axis=0)

        def predict_label(example):
            return np.sign(np.dot(w, psi(example)))

        predictions = [predict_label(example) for example in merged_training_and_test_X]
        predictions = np.array(predictions).reshape((len(predictions), 1))

        # indices 0,1 are X's features, index 2 is predicted Y
        merged_training_and_test_with_predictions = np.concatenate(
            (merged_training_and_test_X, predictions), axis=1)

        x_axis_positive_label, y_axis_positive_label = \
            zip(*[(example[0], example[1]) for example in merged_training_and_test_with_predictions if
                  example[2] == 1])

        x_axis_negative_label, y_axis_negative_label = \
            zip(*[(example[0], example[1]) for example in merged_training_and_test_with_predictions if
                  example[2] == -1])

        plt.scatter(x_axis_negative_label, y_axis_negative_label, color="red", label="-1")
        plt.scatter(x_axis_positive_label, y_axis_positive_label, color="blue", label="+1")

        plt.show()


def simple_test():
    # load question 2 data
    data = np.load('EX2q2_mnist.npz')
    train_X = data['Xtrain']
    train_Y = data['Ytrain']

    m = 100

    # Get a random m training examples from the training set
    indices = np.random.permutation(train_X.shape[0])
    _trainX = train_X[indices[:m]]
    _trainy = train_Y[indices[:m]]

    # run the softsvmpoly algorithm
    w = softsvmpoly(10, 5, _trainX, _trainy)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(w, np.ndarray), "The output of the function softsvmbf should be a numpy array"
    assert w.shape[0] == m and w.shape[1] == 1, f"The shape of the output should be ({m}, 1)"

    print("Passed successfully!")


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    # simple_test()
    # here you may add any code that uses the above functions to solve question 4
    # q4_a()
    # q4_b_kernel_svm()
    # q4_b_linear_svm()
    # q4_e()
    q4_f("ii", "iii", "iv")

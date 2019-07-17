import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class NeuralNetwork:

    def __init__(self):
        self.w = np.zeros((1, 1))
        self.b = 0

    def initialize(self, dims):
        self.w = np.zeros((dims, 1))
        self.b = 0

    @staticmethod
    def propagate(w, b, data_set_x, solution_set_y):
        m = data_set_x.shape[1]
        a = sigmoid(np.dot(w.T, data_set_x) + b)
        cost = -1/m * (np.dot(solution_set_y, np.log(a.T)) + np.dot(1 - solution_set_y, np.log((1 - a).T)))
        dw = (1/m) * (np.dot(data_set_x, (a - solution_set_y).T))
        db = (1/m) * (np.sum(a - solution_set_y))
        gradients = {"dw": dw, "db": db}
        cost = np.squeeze(cost)
        return gradients, cost

    def train_model(self, data_set_x, solution_set_y, learning_rate, iterations, disp = False):
        for i in range(iterations):
            gradients, cost = NeuralNetwork.propagate(self.w, self.b, data_set_x, solution_set_y)

            if disp:
                print('Running gradient descent... iteration %i' % i)
                print('w = ' + str(self.w))
                print('b = ' + str(self.b))
                print('cost = ' + str(cost))

            dw = gradients["dw"]
            db = gradients["db"]

            self.w = self.w - learning_rate * dw
            self.b = self.b - learning_rate * db

        print('Gradient descent with %i iterations complete. ' % iterations)
        print('w = '+str(self.w))
        print('b = '+str(self.b))
        print('Current cost = '+str(cost))

    def predict(self, test_data_set):
        m = test_data_set.shape[1]
        self.w = self.w.reshape(test_data_set.shape[0], 1)
        y_predictions = np.zeros((1, m))
        a = sigmoid(np.dot(self.w.T, test_data_set) + self.b)
        for i in range(a.shape[1]):
            if a[0][i] <= 0.5:
                y_predictions[0][i] = 0
            else:
                y_predictions[0][i] = 1

        return y_predictions


NN = NeuralNetwork()
NN.initialize(2)
data_x = np.array([[1., 2., -1.],
                   [3., 4., -3.2]])

sol_y = np.array([[1, 0, 1]])

NN.train_model(data_x, sol_y, 0.005, 500, True)

test_data = np.array([[1., -1.1, -3.2],
                      [1.2, 2., 0.1]])

prediction = NN.predict(test_data)
print('Predictions = '+str(prediction))



print('This is working!')

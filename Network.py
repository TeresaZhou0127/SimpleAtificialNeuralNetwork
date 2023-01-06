import numpy as np


def sigmoid(x: int) -> float:
    """
    An activation function which changes an arbitrary number into a number within 0
    and 1
    :param x: an arbitrary integer
    :return: a float which is within 0 and 1
    """
    return 1 / (1 + np.exp(-x))


def dev_sigmoid(x: int) -> float:
    """
    Calculate the derivative of the sigmoid activation function
    :param x: the input value
    :return: the derivative value of the sigmoid function at the point of the input
    """
    f = sigmoid(x)
    return f * (1 - f)


def mse(true, pred) -> float:
    """
    Calculate the mean squared error of the given data
    :param true: a numpy array which contains the real values
    :param pred: a numpy array which contains the predicted values
    :return: the mse of the two datasets
    """
    return ((true - pred) ** 2).mean()

class Network:
    """
    A neural network with 3 input neurons, one hidden layer with 3 hidden neurons and 1
    output neuron. After training this network with the help of Stochastic Gradient Descent
    algorithm, the accuracy of this network doing predictions based on the training data will
    be improved.
    """
    def __init__(self):
        """
        Set up the values of the weights and bias for each neurons
        """
        # Neuron: input_1
        self.w11 = np.random.normal()
        self.w12 = np.random.normal()
        self.w13 = np.random.normal()

        # Neuron: input_2
        self.w21 = np.random.normal()
        self.w22 = np.random.normal()
        self.w23 = np.random.normal()

        # Neuron: input_3
        self.w31 = np.random.normal()
        self.w32 = np.random.normal()
        self.w33 = np.random.normal()

        # Neuron: hidden_1 & hidden_2 & hidden_3
        self.wh1 = np.random.normal()
        self.wh2 = np.random.normal()
        self.wh3 = np.random.normal()

        self.bh1 = np.random.normal()
        self.bh2 = np.random.normal()
        self.bh3 = np.random.normal()

        # Neuron: output_1
        self.bo1 = np.random.normal()

    def feedforward(self, array):
        """
        The process of feedforward; to be more specific, processing the given input
        based on the neuron's weights and bias and return the output value
        :param array: a numpy array with three values
        :return: the output value after processing
        """
        h1 = sigmoid(self.w11 * array[0] + self.w21 * array[1] + self.w31 * array[2] + self.bh1)
        h2 = sigmoid(self.w12 * array[0] + self.w22 * array[1] + self.w32 * array[2] + self.bh2)
        h3 = sigmoid(self.w12 * array[0] + self.w22 * array[1] + self.w32 * array[2] + self.bh3)

        o1 = sigmoid(self.wh1 * h1 + self.wh2 * h2 + self.wh3 * h3 + self.bo1)
        return o1

    def training(self, training_data, trues):
        """
        Train the neural network based on the Stochastic Gradient Descent algorithm.
        :param training_data: a dataset that contains all the information used to do the
                              following predictions
        :param trues: real values of the parameter to be predicted by the network
        :return: the predicted results and show how the results get more and more accurate
                 during the training
        """
        # Set the learning rate to be 0.08
        learning_rate = 0.08
        # Set the training rounds to be 500
        trains = 500

        for train in range(trains):
            for data, true in zip(training_data, trues):
                h1_value = self.w11 * data[0] + self.w21 * data[1] + self.w31 * data[2] + self.bh1
                h1 = sigmoid(h1_value)
                h2_value = self.w12 * data[0] + self.w22 * data[1] + self.w32 * data[2] + self.bh2
                h2 = sigmoid(h2_value)
                h3_value = self.w12 * data[0] + self.w22 * data[1] + self.w32 * data[2] + self.bh3
                h3 = sigmoid(h3_value)
                o1_value = self.wh1 * h1 + self.wh2 * h2 + self.wh3 * h3 + self.bo1
                o1 = sigmoid(o1_value)
                pred = o1

                # because loss = (true - pred) ** 2
                d_loss_d_pred = (-1) * 2 * (true - pred)

                # Neuron: hidden_1
                d_h1_d_w11 = data[0] * dev_sigmoid(h1_value)
                d_h1_d_w21 = data[1] * dev_sigmoid(h1_value)
                d_h1_d_w31 = data[2] * dev_sigmoid(h1_value)
                d_h1_d_bh1 = dev_sigmoid(h1_value)

                # Neuron: hidden_2
                d_h2_d_w12 = data[0] * dev_sigmoid(h2_value)
                d_h2_d_w22 = data[1] * dev_sigmoid(h2_value)
                d_h2_d_w32 = data[2] * dev_sigmoid(h2_value)
                d_h2_d_bh2 = dev_sigmoid(h2_value)

                # Neuron: hidden_3
                d_h3_d_w13 = data[0] * dev_sigmoid(h3_value)
                d_h3_d_w23 = data[1] * dev_sigmoid(h3_value)
                d_h3_d_w33 = data[2] * dev_sigmoid(h3_value)
                d_h3_d_bh3 = dev_sigmoid(h3_value)

                # Neuron: output_1
                d_o1_d_wh1 = h1 * dev_sigmoid(o1_value)
                d_o1_d_wh2 = h2 * dev_sigmoid(o1_value)
                d_o1_d_wh3 = h3 * dev_sigmoid(o1_value)
                d_o1_d_bo1 = dev_sigmoid(o1_value)

                # Some other derivatives
                d_pred_d_h1 = self.wh1 * dev_sigmoid(o1_value)
                d_pred_d_h2 = self.wh2 * dev_sigmoid(o1_value)
                d_pred_d_h3 = self.wh3 * dev_sigmoid(o1_value)

                # Update the weights and bias of the neurons based on the
                # Stochastic Gradient Descent rule
                # w <- w - learning_rate * (d_loss_d_w)
                # b <- b - learning_rate * (d_loss_d_b)
                # Neuron: hidden_1
                self.w11 -= learning_rate * (d_loss_d_pred * d_pred_d_h1 * d_h1_d_w11)
                self.w21 -= learning_rate * (d_loss_d_pred * d_pred_d_h1 * d_h1_d_w21)
                self.w31 -= learning_rate * (d_loss_d_pred * d_pred_d_h1 * d_h1_d_w31)
                self.bh1 -= learning_rate * (d_loss_d_pred * d_pred_d_h1 * d_h1_d_bh1)

                # Neuron: hidden_2
                self.w12 -= learning_rate * (d_loss_d_pred * d_pred_d_h2 * d_h2_d_w12)
                self.w22 -= learning_rate * (d_loss_d_pred * d_pred_d_h2 * d_h2_d_w22)
                self.w32 -= learning_rate * (d_loss_d_pred * d_pred_d_h2 * d_h2_d_w32)
                self.bh2 -= learning_rate * (d_loss_d_pred * d_pred_d_h2 * d_h2_d_bh2)

                # Neuron: hidden_3
                self.w13 -= learning_rate * (d_loss_d_pred * d_pred_d_h3 * d_h3_d_w13)
                self.w23 -= learning_rate * (d_loss_d_pred * d_pred_d_h3 * d_h3_d_w23)
                self.w33 -= learning_rate * (d_loss_d_pred * d_pred_d_h3 * d_h3_d_w33)
                self.bh3 -= learning_rate * (d_loss_d_pred * d_pred_d_h3 * d_h3_d_bh3)

                # Neuron: output_1
                self.wh1 -= learning_rate * (d_loss_d_pred * d_o1_d_wh1)
                self.wh2 -= learning_rate * (d_loss_d_pred * d_o1_d_wh2)
                self.wh3 -= learning_rate * (d_loss_d_pred * d_o1_d_wh3)
                self.bo1 -= learning_rate * (d_loss_d_pred * d_o1_d_bo1)

                if train % 5 == 0:
                    preds = np.apply_along_axis(self.feedforward(), 1, training_data)
                    loss = mse(trues, preds)
                    print(loss)








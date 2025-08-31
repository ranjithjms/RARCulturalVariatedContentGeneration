import numpy
import numpy as np
from matplotlib import pyplot as plt
import argparse
import time
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from CVCG import config as cfg

class Existing_RNN:
    def __init__(self, input=None, label=None, n_ins=2, hidden_layer_sizes=[3, 3], n_outs=2, rng=None):

        xrange = []
        self.x = input
        self.y = label
        sigmoid = ""

        self.sigmoid_layers = []
        self.rbm_layers = []
        self.n_layers = len(hidden_layer_sizes)  # = len(self.rbm_layers)

        if rng is None:
            rng = numpy.random.RandomState(1234)

        assert self.n_layers > 0

        # construct multi-layer
        for i in xrange(self.n_layers):
            # layer_size
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layer_sizes[i - 1]

            # layer_input
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].sample_h_given_v()

            # construct sigmoid_layer
            sigmoid_layer = self.HiddenLayer(input=layer_input,
                                             n_in=input_size,
                                             n_out=hidden_layer_sizes[i],
                                             rng=rng,
                                             activation=sigmoid)
            self.sigmoid_layers.append(sigmoid_layer)

            # construct rbm_layer
            rbm_layer = self.RBM(input=layer_input,
                            n_visible=input_size,
                            n_hidden=hidden_layer_sizes[i],
                            W=sigmoid_layer.W,  # W, b are shared
                            hbias=sigmoid_layer.b)
            self.rbm_layers.append(rbm_layer)

        # layer for output using Logistic Regression
        self.log_layer = self.LogisticRegression(input=self.sigmoid_layers[-1].sample_h_given_v(),
                                            label=self.y,
                                            n_in=hidden_layer_sizes[-1],
                                            n_out=n_outs)

        # finetune cost: the negative log likelihood of the logistic regression layer
        self.finetune_cost = self.log_layer.negative_log_likelihood()

    def pretrain(self, lr=0.1, k=1, epochs=100):
        xrange = []
        # pre-train layer-wise
        for i in xrange(self.n_layers):
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[i - 1].sample_h_given_v(layer_input)
            rbm = self.rbm_layers[i]

            for epoch in xrange(epochs):
                rbm.contrastive_divergence(lr=lr, k=k, input=layer_input)
                # cost = rbm.get_reconstruction_cross_entropy()
                # print >> sys.stderr, \
                #        'Pre-training layer %d, epoch %d, cost ' %(i, epoch), cost

    def finetune(self, lr=0.1, epochs=100):
        layer_input = self.sigmoid_layers[-1].sample_h_given_v()

        # train log_layer
        epoch = 0
        done_looping = False
        while (epoch < epochs) and (not done_looping):
            self.log_layer.train(lr=lr, input=layer_input)
            # self.finetune_cost = self.log_layer.negative_log_likelihood()
            # print >> sys.stderr, 'Training epoch %d, cost is ' % epoch, self.finetune_cost

            lr *= 0.95
            epoch += 1

    def predict(self, x):
        xrange = []
        layer_input = x

        for i in xrange(self.n_layers):
            sigmoid_layer = self.sigmoid_layers[i]
            layer_input = sigmoid_layer.output(input=layer_input)

        out = self.log_layer.predict(layer_input)
        return out

    def test_dbn(self, pretrain_lr=0.1, pretraining_epochs=1000, k=1, \
                 finetune_lr=0.1, finetune_epochs=200):
        x = numpy.array([[1, 1, 1, 0, 0, 0],
                         [1, 0, 1, 0, 0, 0],
                         [1, 1, 1, 0, 0, 0],
                         [0, 0, 1, 1, 1, 0],
                         [0, 0, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 0]])
        y = numpy.array([[1, 0],
                         [1, 0],
                         [1, 0],
                         [0, 1],
                         [0, 1],
                         [0, 1]])

        rng = numpy.random.RandomState(123)

        # construct RNN
        rnn = Existing_RNN(input=x, label=y, n_ins=6, hidden_layer_sizes=[3, 3], n_outs=2, rng=rng)

        # pre-training (TrainUnsupervised)
        rnn.pretrain(lr=pretrain_lr, k=1, epochs=pretraining_epochs)

        # fine-tuning (SupervisedFineTuning)
        rnn.finetune(lr=finetune_lr, epochs=finetune_epochs)

        # test
        x = numpy.array([[1, 1, 0, 0, 0, 0],
                         [0, 0, 0, 1, 1, 0],
                         [1, 1, 1, 1, 1, 0]])

        print
        rnn.predict(x)

    def training(self, iptrdata, iptrcls):
        parser = argparse.ArgumentParser(description='Train Existing RNN for Image Captioning')

        parser.add_argument('-train', help='Train data', type=str, required=True)
        parser.add_argument('-val', help='Validation data (1vs9 for validation on 10 percents of training data)', type=str)
        parser.add_argument('-test', help='Test data', type=str)

        parser.add_argument('-e', help='Number of epochs', type=int, default=1000)
        parser.add_argument('-p', help='Crop of early stop (0 for ignore early stop)', type=int, default=10)
        parser.add_argument('-b', help='Batch size', type=int, default=128)

        parser.add_argument('-pre', help='Pre-trained weight', type=str)


        train_inputs = []
        train_outputs = []
        time.sleep(54)
        if len(train_inputs) > 0:
            if (train_inputs.ndim != 4):
                raise ValueError("The training data input has {num_dims} but it must have 4 dimensions. The first dimension is the number of training samples, the second & third dimensions represent the width and height of the sample, and the fourth dimension represents the number of channels in the sample.".format(num_dims=train_inputs.ndim))
            if (train_inputs.shape[0] != len(train_outputs)):
                raise ValueError(
                            "Mismatch between the number of input samples and number of labels: {num_samples_inputs} != {num_samples_outputs}.".format(num_samples_inputs=train_inputs.shape[0], num_samples_outputs=len(train_outputs)))

            network_predictions = []
            network_error = 0
            for epoch in range(self.epochs):
                print("Epoch {epoch}".format(epoch=epoch))
                for sample_idx in range(train_inputs.shape[0]):
                    # print("Sample {sample_idx}".format(sample_idx=sample_idx))
                    self.feed_sample(train_inputs[sample_idx, :])

                    try:
                        predicted_label = \
                            self.numpy.where(self.numpy.max(self.last_layer.layer_output) == self.last_layer.layer_output)[0][0]
                    except IndexError:
                        print(self.last_layer.layer_output)
                        raise IndexError("Index out of range")
                    network_predictions.append(predicted_label)

                    network_error = network_error + abs(predicted_label - train_outputs[sample_idx])

                    self.update_weights(network_error)

        model = Sequential([
            Dense(units=64, activation='relu', input_shape=(180, 180, 3)),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(10, activation='softmax')
        ])

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        model.save("..\\Models\\EDBN.h5")

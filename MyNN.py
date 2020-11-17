class NeuralNetwork:

    def __init__(self, shape):
        self.shape = shape
        self.w1 = [
                    [1, 2],
                    [3, 4]
                  ]
        self.w2 = [
                    [2],
                    [3]
                  ]
        self.u2_activation = self.lin_fun
        self.u3_activation = self.lin_fun

        self.weights = [[[1] * shape[x + 1] for y in range(shape[x])] for x in range(len(shape) - 1)]
        self.outputs = [[0] * shape[x] for x in range(len(shape))]
        self.before_activation = [[0] * shape[x] for x in range(len(shape))]

        #self.weights[0] = [[1, 2], [3, 4]]
        #self.weights[1] = [[2], [3]]
        self.activation_functions = [self.lin_fun, self.lin_fun]

        print(self.w1)

        print(self.w2)

        print(self.outputs)

        print(self.weights)

    def lin_fun(self, input):
        result = []
        for i in range(len(input)):
            result.append(2 * input[i])
        return result

    def feedForward(self, input):

        self.outputs[0] = input
        for layer in range(len(self.shape)-1):
            output_layer = self.outputs[layer + 1]
            for output_neuron in range(self.shape[layer + 1]):
                output_layer[output_neuron] = 0
                for input_neuron in range(self.shape[layer]):
                    output_layer[output_neuron] += self.outputs[layer][input_neuron] * self.weights[layer][input_neuron][output_neuron]
            self.before_activation[layer + 1] = self.outputs[layer + 1]
            self.outputs[layer + 1] = self.activation_functions[layer](self.outputs[layer + 1])

        u2 = [
             self.w1[0][0] * input[0] + self.w1[1][0] * input[1],
             self.w1[0][1] * input[0] + self.w1[1][1] * input[1]
             ]
        o2 = self.u2_activation(u2)

        u3 = [
               self.w2[0][0] * o2[0] + self.w2[1][0] * o2[1]
             ]
        o3 = self.u3_activation(u3)
        print(input)
        print(u2)
        print(o2)
        print(u3)
        print(o3)



network = NeuralNetwork([3, 2, 1]);

# print(network.shape)
#
# print(network.w1[0][1])
network.feedForward([1, 2, 3])

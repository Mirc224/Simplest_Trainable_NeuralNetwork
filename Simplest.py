import random
import math

class NeuralNetwork:
    def __init__(self, shape):
        self.shape = shape
        self.weights = [[[0] * shape[x] for y in range(shape[x+1])] for x in range(len(shape) - 1)]
        self.outputs = [[0] * shape[x] for x in range(len(shape))]
        self.before_activation = [[0] * shape[x] for x in range(len(shape))]

        for layer in self.weights:
            for output_neuron in layer:
                for input_neuron in range(len(output_neuron)):
                    output_neuron[input_neuron] = round(random.uniform(0, 1),3)
        self.activation_functions = [self.linear] * (len(shape) - 1 )
        self.derivation_functions = [self.linear_der] * (len(shape) - 1 )

    def linear_der(self, input):
        result = []
        for i in range(len(input)):
            result.append(1)
        return result

    def linear(self, input):
        result = []
        for i in range(len(input)):
            result.append(input[i])
        return result

    def sigmoid(self, input):
        result = []
        for i in range(len(input)):
            result.append(1/(1 + math.pow(math.e, input[i])))
        return result

    def sigmoid_derivate(self, input):
        non_derivate_result = self.sigmoid(input)
        result = []
        for i in range(len(non_derivate_result)):
            result.append(non_derivate_result[i] * (1 - non_derivate_result[i]))
        return result

    def lin_fun(self, input):
        result = []
        for i in range(len(input)):
            result.append(2 * input[i])
        return result

    def calculate_cost(self, target):
        result = []
        output_layer_number = len(self.shape) - 1
        for i in range(len(self.outputs[output_layer_number])):
            result.append(0.5*math.pow((target[i] - self.outputs[output_layer_number][i]), 2))
        return result

    def predict(self, input):
        self.outputs[0] = input
        for layer in range(len(self.shape)-1):
            output_layer = self.outputs[layer + 1]
            for output_neuron in range(self.shape[layer + 1]):
                output_layer[output_neuron] = 0
                for input_neuron in range(self.shape[layer]):
                    output_layer[output_neuron] += self.outputs[layer][input_neuron] * self.weights[layer][output_neuron][input_neuron]
            self.before_activation[layer + 1] = self.outputs[layer + 1]
            self.outputs[layer + 1] = self.activation_functions[layer](self.outputs[layer + 1])
        return self.outputs[len(self.shape)-1]

    def back_propagate(self, target):
        gradinets = [[] for _ in range(len(self.shape) -1)]
        sigma = []
        output_layer_number = len(self.shape) - 1
        derivation_output = self.derivation_functions[output_layer_number-1](self.outputs[output_layer_number])
        for i in range(len(target)):
            sigma.append((target[i]-self.outputs[output_layer_number][i])*derivation_output[i])

        partials_derivates = []

        activation_previous_layer = self.outputs[output_layer_number - 1]
        for i in range(len(sigma)):
            partials_derivates.append([])
            for j in range(len(activation_previous_layer)):
                partials_derivates[i].append(sigma[i] * activation_previous_layer[j])

        gradinets[output_layer_number - 1] = partials_derivates
        actual_layer = output_layer_number - 1

        while actual_layer > 0:
            temp_sigma = sigma.copy()
            layer_weights = self.weights[actual_layer]
            sigma = [0] * len(self.outputs[actual_layer])
            derivation_output = self.derivation_functions[actual_layer](self.outputs[actual_layer])

            for j in range(len(sigma)):
                for i in range(len(temp_sigma)):
                    sigma[j] += temp_sigma[i] * layer_weights[i][j]

            for i in range(len(temp_sigma)):
                for j in range(len(layer_weights)):
                    sigma[i] += temp_sigma[j] * layer_weights[j][i]
                sigma[i] *= derivation_output[i]

            partials_derivates = []
            activation_previous_layer = self.outputs[actual_layer - 1]
            for i in range(len(sigma)):
                partials_derivates.append([])
                for j in range(len(activation_previous_layer)):
                    partials_derivates[i].append(sigma[i] * activation_previous_layer[j])
            gradinets[actual_layer - 1] = partials_derivates
            actual_layer = actual_layer - 1

            for layer in range(len(gradinets)):
                for i in range(len(gradinets[layer])):
                    for j in range(len(gradinets[layer][i])):
                        self.weights[layer][i][j] += 0.001 * gradinets[layer][i][j]


    def train(self, input, target):
        print('Input')
        print(input)
        print('Target')
        print(target)
        print('Prediction')
        print(self.predict(input))
        print('Cost')
        print(self.calculate_cost(target))
        self.back_propagate(target)

nn = NeuralNetwork([3,2,1])
print(nn.weights)


for i in range(10):
    number = random.randint(1,20)
    nn.train([number,number, number], [number])

print(nn.predict([323,323,323]))





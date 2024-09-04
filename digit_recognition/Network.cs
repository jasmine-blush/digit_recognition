namespace digit_recognition
{
    internal class Network
    {
        private readonly List<Layer> _layers;

        internal Network(int[] layerSizes)
        {
            _layers = [];
            for(int i = 0; i < layerSizes.Length; i++)
            {
                _layers.Add(new Layer(layerSizes[i], new SigmoidActivation()));
            }

            Random rng = new();
            for(int i = 1; i < _layers.Count; i++)
            {
                Layer currLayer = _layers[i];
                Layer prevLayer = _layers[i - 1];
                foreach(Neuron neuron in currLayer.Neurons)
                {
                    neuron.Bias = rng.NextDouble() * 2.0 - 1; // random bias between -1 and 1
                    for(int j = 0; j < prevLayer.Neurons.Count; j++)
                    {
                        neuron.Weights.Add(rng.NextDouble() * 2.0 - 1); // same
                    }
                }
            }
        }

        internal void Train(double[][] trainingData, double[][] trainingTargets, int epochs, double learningRate, bool doOutput)
        {
            for(int epoch = 1; epoch <= epochs; epoch++)
            {
                int correct = 0;
                for(int i = 0; i < trainingData.Length; i++)
                {
                    double[] results = FeedForward(trainingData[i]);

                    double[] delta = CalculateDelta(results, trainingTargets[i], _layers.Last().Activation);
                    UpdateWeightsAndBiases(delta, learningRate);

                    int predictedClass = results.ToList().IndexOf(results.Max());
                    int actualClass = trainingTargets[i].ToList().IndexOf(trainingTargets[i].Max());
                    if(predictedClass == actualClass)
                        correct++;
                }

                if(doOutput)
                    OutputResults("Epoch " + epoch, correct, trainingTargets.Length);
            }
        }

        internal int Test(double[][] testData, double[] testLabels)
        {
            int correct = 0;
            for(int i = 0; i < testData.Length; i++)
            {
                double[] testResults = FeedForward(testData[i]);

                int predictedClass = testResults.ToList().IndexOf(testResults.Max());
                int actualClass = (int)testLabels[i];
                if(predictedClass == actualClass)
                    correct++;
            }
            return correct;
        }

        internal static void OutputResults(string name, int correct, int total)
        {
            Console.WriteLine(string.Format("Accuracy ({0}): {1}/{2} correct, {3}%",
                name,
                correct,
                total,
                (float)correct / total * 100));
        }

        private double[] FeedForward(double[] inputs)
        {
            for(int i = 0; i < inputs.Length; i++)
            {
                _layers[0].Neurons[i].Value = inputs[i];
            }

            for(int i = 1; i < _layers.Count; i++)
            {
                Layer currLayer = _layers[i];
                Layer prevLayer = _layers[i - 1];

                foreach(Neuron neuron in currLayer.Neurons)
                {
                    double weightedSum = 0;
                    for(int j = 0; j < neuron.Weights.Count; j++)
                    {
                        weightedSum += neuron.Weights[j] * prevLayer.Neurons[j].Value;
                    }
                    neuron.Value = weightedSum + neuron.Bias;
                    neuron.Value = currLayer.Activation.Calculate(neuron.Value);
                }
            }

            return _layers.Last().Neurons.Select(neuron => neuron.Value).ToArray();
        }

        private static double[] CalculateDelta(double[] result, double[] target, IActivationFunction activation)
        {
            double[] delta = new double[result.Length];
            for(int i = 0; i < result.Length; i++)
            {
                delta[i] = (target[i] - result[i]) * activation.CalculateDerivative(result[i]);
            }
            return delta;
        }

        private void UpdateWeightsAndBiases(double[] delta, double learningRate)
        {
            for(int i = _layers.Count - 1; i > 0; i--)
            {
                Layer currLayer = _layers[i];
                Layer prevLayer = _layers[i - 1];
                for(int j = 0; j < currLayer.Neurons.Count; j++)
                {
                    Neuron neuron = currLayer.Neurons[j];
                    neuron.Bias += learningRate * delta[j];

                    for(int k = 0; k < prevLayer.Neurons.Count; k++)
                    {
                        neuron.Weights[k] += learningRate * delta[j] * prevLayer.Neurons[k].Value;
                    }
                }

                double[] newDelta = new double[prevLayer.Neurons.Count];
                for(int j = 0; j < prevLayer.Neurons.Count; j++)
                {
                    newDelta[j] = 0;
                    for(int k = 0; k < currLayer.Neurons.Count; k++)
                    {
                        newDelta[j] += currLayer.Neurons[k].Weights[j] * delta[k];
                    }
                    newDelta[j] *= prevLayer.Activation.CalculateDerivative(prevLayer.Neurons[j].Value);

                }
                delta = newDelta;
            }
        }
    }
}


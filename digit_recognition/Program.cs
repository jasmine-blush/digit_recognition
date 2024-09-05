using digit_recognition;

int[] layers = [784, 128, 64, 10];
int trainingAmount = 60000;
int epochs = 20;
double learningRate = 0.001;
int testAmount = 200;

double[] labels = MNISTLoader.LoadTrainingLabels(trainingAmount);
double[][] trainingLabels = labels.Select(label => Enumerable.Range(0, 10).Select(i => i == label ? 1.0 : 0.0).ToArray()).ToArray();
double[][] trainingImages = MNISTLoader.LoadTrainingImages(trainingAmount);

double[] testLabels = MNISTLoader.LoadTestLabels(testAmount);
double[][] testImages = MNISTLoader.LoadTestImages(testAmount);


int[] correctSum = [0, 0];
int iterations = 20;
for(int i = 0; i < iterations; i++)
{
    Console.WriteLine("Iteration " + (i + 1));
    int[] correctResults = TestNetworkAccuracy();
    correctSum[0] += correctResults[0];
    correctSum[1] += correctResults[1];
    Network.OutputResults("Training Data", correctResults[0], labels.Length);
    Network.OutputResults("Test Data", correctResults[1], testLabels.Length);
    Console.WriteLine();
}
Console.WriteLine("Average Results");
Network.OutputResults("Training Data", correctSum[0] / iterations, labels.Length);
Network.OutputResults("Test Data", correctSum[1] / iterations, testLabels.Length);


int[] TestNetworkAccuracy()
{
    Network neuralNet = new(layers);

    neuralNet.Train(trainingImages, trainingLabels, epochs, learningRate, true);

    int[] correctResults = [0, 0];
    correctResults[0] = neuralNet.Test(trainingImages, labels);
    correctResults[1] = neuralNet.Test(testImages, testLabels);
    return correctResults;
}
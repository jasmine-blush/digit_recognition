namespace digit_recognition
{
    internal class Layer
    {
        internal List<Neuron> Neurons;
        internal IActivationFunction Activation;

        internal Layer(int size, IActivationFunction activationFunction)
        {
            Neurons = [];
            for(int i = 0; i < size; i++)
            {
                Neurons.Add(new Neuron());
            }
            Activation = activationFunction;
        }
    }
}

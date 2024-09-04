namespace digit_recognition
{
    internal class Neuron
    {
        internal double Value;
        internal double Bias;
        internal List<double> Weights; //Weights to neurons from previous layer

        public Neuron()
        {
            Weights = [];
        }
    }
}

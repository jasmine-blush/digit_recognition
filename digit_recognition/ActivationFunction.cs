namespace digit_recognition
{
    public interface IActivationFunction
    {
        double Calculate(double x);
        double CalculateDerivative(double x);
    }

    internal class SigmoidActivation : IActivationFunction
    {
        public double Calculate(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        public double CalculateDerivative(double x)
        {
            double activation = Calculate(x);
            return activation * (1.0 - activation);
        }
    }

    internal class ReLUActivation : IActivationFunction
    {
        public double Calculate(double x)
        {
            return Math.Max(0.0, x);
        }

        public double CalculateDerivative(double x)
        {
            return x > 0.0 ? 1.0 : 0.0;
        }
    }
}

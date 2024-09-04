namespace digit_recognition
{
    internal static class MNISTLoader
    {
        internal static double[] LoadTrainingLabels(int amount)
        {
            return LoadLabels("./train-labels.idx1-ubyte", amount);
        }

        internal static double[][] LoadTrainingImages(int amount)
        {
            return LoadImages("./train-images.idx3-ubyte", amount);
        }

        internal static double[] LoadTestLabels(int amount)
        {
            return LoadLabels("./t10k-labels.idx1-ubyte", amount);
        }

        internal static double[][] LoadTestImages(int amount)
        {
            return LoadImages("./t10k-images.idx3-ubyte", amount);
        }

        private static double[] LoadLabels(string path, int amount)
        {
            using BinaryReader br = new(new FileStream(path, FileMode.Open));
            int magicNumber = br.ReadInt32();
            int itemCount = ReverseBytes(br.ReadInt32());

            amount = amount > itemCount ? itemCount : amount;

            double[] labels = new double[amount];
            for(int i = 0; i < amount; i++)
            {
                labels[i] = br.ReadByte();
            }
            return labels;
        }

        private static double[][] LoadImages(string path, int amount)
        {
            using BinaryReader br = new(new FileStream(path, FileMode.Open));
            int magicNumber = br.ReadInt32();
            int itemCount = ReverseBytes(br.ReadInt32());
            int imageWidth = ReverseBytes(br.ReadInt32());
            int imageHeight = ReverseBytes(br.ReadInt32());

            amount = amount > itemCount ? itemCount : amount;

            double[][] images = new double[amount][];
            for(int i = 0; i < amount; i++)
            {
                images[i] = new double[imageWidth * imageHeight];
                for(int j = 0; j < (imageWidth * imageHeight); j++)
                {
                    images[i][j] = br.ReadByte() / 255.0; // Normalize inputs
                }
            }
            return images;
        }

        private static int ReverseBytes(int value)
        {
            byte[] bytes = BitConverter.GetBytes(value);
            Array.Reverse(bytes);
            return BitConverter.ToInt32(bytes);
        }
    }
}

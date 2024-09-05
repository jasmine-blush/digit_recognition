namespace digit_recognition
{
    internal static class MNISTLoader
    {
        private static int _imageWidth = 0;
        private static int _imageHeight = 0;

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
            _imageWidth = ReverseBytes(br.ReadInt32());
            _imageHeight = ReverseBytes(br.ReadInt32());

            amount = amount > itemCount ? itemCount : amount;

            double[][] images = new double[amount][];
            for(int i = 0; i < amount; i++)
            {
                images[i] = new double[_imageWidth * _imageHeight];
                for(int j = 0; j < (_imageWidth * _imageHeight); j++)
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

        internal static double[] AugmentImage(double[] inputImage)
        {
            Random rng = new();

            double[,] image = new double[_imageHeight, _imageWidth];
            for(int i = 0; i < _imageHeight; i++)
            {
                for(int j = 0; j < _imageWidth; j++)
                {
                    image[i, j] = inputImage[i * _imageWidth + j];
                }
            }

            image = RotateImage(image, rng.Next(51) - 25); //random rotation between -25 and 25°

            double[] outputImage = new double[_imageWidth * _imageHeight];
            for(int i = 0; i < outputImage.Length; i++)
            {
                outputImage[i] = image[i / _imageWidth, i % _imageWidth];
            }
            return outputImage;
        }

        private static double[,] RotateImage(double[,] image, double angle)
        {
            int centerX = image.GetLength(1) / 2;
            int centerY = image.GetLength(0) / 2;
            double radianAngle = angle * Math.PI / 180;

            double[,] rotatedImage = new double[image.GetLength(0), image.GetLength(1)];
            for(int i = 0; i < rotatedImage.GetLength(0); i++)
            {
                for(int j = 0; j < rotatedImage.GetLength(1); j++)
                {
                    rotatedImage[i, j] = -1;
                }
            }

            for(int y = 0; y < image.GetLength(0); y++)
            {
                for(int x = 0; x < image.GetLength(1); x++)
                {
                    int offsetX = x - centerX;
                    int offsetY = y - centerY;

                    int newX = (int)Math.Round(offsetX * Math.Cos(radianAngle) - offsetY * Math.Sin(radianAngle)) + centerX;
                    int newY = (int)Math.Round(offsetX * Math.Sin(radianAngle) + offsetY * Math.Cos(radianAngle)) + centerY;

                    if(newX >= 0 && newX < image.GetLength(1) && newY >= 0 && newY < image.GetLength(0))
                    {
                        rotatedImage[newY, newX] = image[y, x];
                    }
                }
            }

            rotatedImage = Interpolate(rotatedImage);

            return rotatedImage;
        }

        private static double[,] Interpolate(double[,] image)
        {
            double[,] interpolatedImage = new double[image.GetLength(0), image.GetLength(1)];
            Array.Copy(image, interpolatedImage, image.GetLength(0) * image.GetLength(1));

            for(int y = 0; y < image.GetLength(0); y++)
            {
                for(int x = 0; x < image.GetLength(1); x++)
                {
                    if(image[y, x] == -1)
                    {
                        List<double> surroundingValues = [];
                        if(x - 1 >= 0 && y - 1 >= 0)
                            surroundingValues.Add(image[y - 1, x - 1]);
                        if(x + 1 < image.GetLength(1) && y - 1 >= 0)
                            surroundingValues.Add(image[y - 1, x + 1]);
                        if(x - 1 >= 0 && y + 1 < image.GetLength(0))
                            surroundingValues.Add(image[y + 1, x - 1]);
                        if(x + 1 < image.GetLength(1) && y + 1 < image.GetLength(0))
                            surroundingValues.Add(image[y + 1, x + 1]);

                        for(int i = 0; i < surroundingValues.Count; i++)
                        {
                            if(surroundingValues[i] == -1)
                            {
                                surroundingValues[i] = 0;
                            }
                        }
                        double interpolatedValue = surroundingValues.Sum(x => x) / surroundingValues.Count;
                        interpolatedImage[y, x] = interpolatedValue;
                    }
                }
            }

            for(int y = 0; y < image.GetLength(0); y++)
            {
                for(int x = 0; x < image.GetLength(1); x++)
                {
                    if(interpolatedImage[y, x] == -1)
                    {
                        interpolatedImage[y, x] = 0;
                    }
                }
            }
            return interpolatedImage;
        }
    }
}

public class ImageClassifier {
    // Creates a feature vector (1D array) from the given picture.
    // Must be greyscale
    public static double[] extractFeatures(Picture picture) {
        double[] featureVector = new double[picture.height() * picture.width()];
        int z = 0;      // counter

        // iterates through the 2d array width-wise
        for (int i = 0; i < picture.height(); i++) {
            for (int j = 0; j < picture.width(); j++) {
                featureVector[z] = picture.get(j, i).getBlue();
                z++;
            }
        }
        return featureVector;
    }

    // Trains and tests a multi-perceptron's ability to classify images
    public static void main(String[] args) {
        // TRAINING
        In trainFile = new In(args[0]);
        int m = trainFile.readInt();        // m
        int width = trainFile.readInt();    // width
        int height = trainFile.readInt();   // height

        MultiPerceptron perceptrons = new MultiPerceptron(m, width * height);
        while (!trainFile.isEmpty()) {
            Picture image = new Picture(trainFile.readString());
            double[] features = extractFeatures(image);
            int label = trainFile.readInt();
            perceptrons.trainMulti(features, label);
        }

        // TESTING
        In testFile = new In(args[1]);
        double errors = 0.0;
        int total = 0;

        // clears first three integers
        for (int i = 0; i < 3; i++)
            testFile.readInt();

        while (!testFile.isEmpty()) {
            // reads image and makes array representation
            String fileName = testFile.readString();
            Picture image = new Picture(fileName);
            double[] features = extractFeatures(image);

            // predicts label and checks if it is correct, if wrong print data
            int label = testFile.readInt();
            int prediction = perceptrons.predictMulti(features);
            if (prediction != label) {
                System.out.println(fileName + ", label = " + label +
                                           ", predict = " + prediction);
                errors += 1.0; // increment if an error is found
            }
            total++;
        }
        StdOut.println("test error rate = " + errors / total);
    }
}

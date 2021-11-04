public class MultiPerceptron {
    // array of perceptrons
    private Perceptron[] perceptrons;

    // Creates a multi-perceptron object with m classes and n inputs.
    // It creates an array of m perceptrons, each with n inputs.
    public MultiPerceptron(int m, int n) {
        perceptrons = new Perceptron[m];
        for (int i = 0; i < m; i++) {
            perceptrons[i] = new Perceptron(n);
        }
    }

    // Returns the number of classes m.
    public int numberOfClasses() {
        return perceptrons.length;
    }

    // Returns the number of inputs n (length of the feature vector).
    public int numberOfInputs() {
        return perceptrons[0].numberOfInputs();
    }

    // Returns the predicted label (between 0 and m-1) for the given input.
    public int predictMulti(double[] x) {
        double sum = perceptrons[0].weightedSum(x);
        int index = 0;
        // iterates through all perceptrons searching for highest weighted sum
        for (int i = 1; i < perceptrons.length; i++) {
            double tempSum = perceptrons[i].weightedSum(x);
            if (tempSum > sum) {
                sum = tempSum;
                index = i;
            }
        }
        return index;
    }

    // Trains this multi-perceptron on the labeled (between 0 and m-1) input.
    public void trainMulti(double[] x, int label) {
        perceptrons[label].train(x, 1);                 // one
        for (int i = 0; i < perceptrons.length; i++) {  // vs all
            if (i != label)
                perceptrons[i].train(x, -1);
        }
    }

    // Returns a String representation of this MultiPerceptron, with
    // the string representations of the perceptrons separated by commas
    // and enclosed in parentheses.
    // Example with m = 2 and n = 3: ((2.0, 0.0, -2.0), (3.0, 4.0, 5.0))
    public String toString() {
        String statement = "(";
        // concatenates all perceptrons
        for (int i = 0; i < perceptrons.length - 1; i++) {
            statement += perceptrons[i].toString() + ", ";
        }
        // adds last perceptron and returns
        return statement + perceptrons[perceptrons.length - 1] + ")";
    }

    // Tests this class by directly calling all instance methods.
    public static void main(String[] args) {
        int m = 2;
        int n = 3;

        double[] training1 = { 3.0, 4.0, 5.0 };  // class 1
        double[] training2 = { 2.0, 0.0, -2.0 };  // class 0
        double[] training3 = { -2.0, 0.0, 2.0 };  // class 1
        double[] training4 = { 5.0, 4.0, 3.0 };  // class 0


        MultiPerceptron perceptron = new MultiPerceptron(m, n);
        StdOut.println(perceptron);
        StdOut.println("Number of classes: " + perceptron.numberOfClasses());
        StdOut.println("Number of inputs: " + perceptron.numberOfInputs());
        perceptron.trainMulti(training1, 1);
        StdOut.println(perceptron);
        perceptron.trainMulti(training2, 0);
        StdOut.println(perceptron);
        perceptron.trainMulti(training3, 1);
        StdOut.println(perceptron);
        perceptron.trainMulti(training4, 0);
        StdOut.println(perceptron);

        perceptron.predictMulti(training4);
    }
}

#include <bits/stdc++.h>
using namespace std;

using Vector = vector<double>;
using Matrix = vector<vector<double>>;

double randomWeight()
{
    return ((double)rand() / RAND_MAX) * 2.0 - 1.0;
}

double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double dSigmoid(double y)
{
    return y * (1.0 - y); // derivative using output
}

class Layer
{
public:
    int inSize, outSize;
    Matrix weights;
    Vector bias;
    Vector output;
    Vector delta;

    Layer(int in, int out)
    {
        inSize = in;
        outSize = out;

        weights.resize(out, Vector(in));
        bias.resize(out);
        output.resize(out);
        delta.resize(out);

        for (int i = 0; i < out; i++)
        {
            for (int j = 0; j < in; j++)
                weights[i][j] = randomWeight();
            bias[i] = randomWeight();
        }
    }

    Vector forward(const Vector &input)
    {
        for (int i = 0; i < outSize; i++)
        {
            double sum = bias[i];
            for (int j = 0; j < inSize; j++)
                sum += weights[i][j] * input[j];
            output[i] = sigmoid(sum);
        }
        return output;
    }
};

class NeuralNetwork
{
public:
    vector<Layer> layers;
    double lr;

    NeuralNetwork(const vector<int> &architecture, double learningRate)
    {
        lr = learningRate;
        for (int i = 1; i < architecture.size(); i++)
            layers.emplace_back(architecture[i - 1], architecture[i]);
    }

    Vector forward(Vector input)
    {
        for (auto &layer : layers)
            input = layer.forward(input);
        return input;
    }

    void backward(const Vector &input, const Vector &target)
    {

        Layer &outLayer = layers.back();
        for (int i = 0; i < outLayer.outSize; i++)
        {
            double error = target[i] - outLayer.output[i];
            outLayer.delta[i] = error * dSigmoid(outLayer.output[i]);
        }

        for (int l = layers.size() - 2; l >= 0; l--)
        {
            Layer &curr = layers[l];
            Layer &next = layers[l + 1];

            for (int i = 0; i < curr.outSize; i++)
            {
                double error = 0.0;
                for (int j = 0; j < next.outSize; j++)
                    error += next.weights[j][i] * next.delta[j];

                curr.delta[i] = error * dSigmoid(curr.output[i]);
            }
        }

        Vector layerInput = input;

        for (int l = 0; l < layers.size(); l++)
        {
            Layer &layer = layers[l];

            if (l != 0)
                layerInput = layers[l - 1].output;

            for (int i = 0; i < layer.outSize; i++)
            {
                for (int j = 0; j < layer.inSize; j++)
                    layer.weights[i][j] += lr * layer.delta[i] * layerInput[j];
                layer.bias[i] += lr * layer.delta[i];
            }
        }
    }

    void train(vector<Vector> &X, vector<Vector> &Y, int epochs)
    {
        for (int e = 0; e < epochs; e++)
        {
            double loss = 0.0;

            for (int i = 0; i < X.size(); i++)
            {
                Vector output = forward(X[i]);

                for (int j = 0; j < output.size(); j++)
                    loss += pow(Y[i][j] - output[j], 2);

                backward(X[i], Y[i]);
            }

            if (e % 1000 == 0)
                cout << "Epoch " << e << " | Loss: " << loss << endl;
        }
    }
};

vector<vector<double>> readCSV(const string &filename)
{
    vector<vector<double>> data;
    ifstream file(filename);
    string line;

    // skip header
    getline(file, line);

    while (getline(file, line))
    {
        stringstream ss(line);
        string value;
        vector<double> row;

        while (getline(ss, value, ','))
            row.push_back(stod(value));

        data.push_back(row);
    }
    return data;
}
void splitXY(
    const vector<vector<double>> &data,
    vector<Vector> &X,
    vector<Vector> &Y,
    int labelIndex)
{
    for (auto &row : data)
    {
        Vector x, y;

        for (int i = 0; i < row.size(); i++)
        {
            if (i == labelIndex)
                y.push_back(row[i]);
            else
                x.push_back(row[i]);
        }

        X.push_back(x);
        Y.push_back(y);
    }
}
void normalize(vector<Vector> &X)
{
    int n = X.size();
    int m = X[0].size();

    Vector mean(m, 0.0), stddev(m, 0.0);

    for (auto &row : X)
        for (int j = 0; j < m; j++)
            mean[j] += row[j];

    for (int j = 0; j < m; j++)
        mean[j] /= n;

    for (auto &row : X)
        for (int j = 0; j < m; j++)
            stddev[j] += pow(row[j] - mean[j], 2);

    for (int j = 0; j < m; j++)
    {
        stddev[j] = sqrt(stddev[j] / n);
        if (stddev[j] == 0)
            stddev[j] = 1;
    }

    for (auto &row : X)
        for (int j = 0; j < m; j++)
            row[j] = (row[j] - mean[j]) / stddev[j];
}
int main()
{
    srand(time(0));

    auto data = readCSV("pima-indians-diabetes.csv");

    vector<Vector> X, Y;
    splitXY(data, X, Y, data[0].size() - 1);ṇṇ

    normalize(X);

    NeuralNetwork nn({(int)X[0].size(), 8, 1}, 0.01);

    nn.train(X, Y, 20000);

    for (int i = 0; i < 5; i++)
    {
        Vector out = nn.forward(X[i]);
        cout << "Pred: " << out[0] << "  Actual: " << Y[i][0] << endl;
    }
}
// int main()
// {
//     srand(time(0));

//     NeuralNetwork nn({2, 4, 1}, 0.1);

//     vector<Vector> X = {
//         {0, 0},
//         {0, 1},
//         {1, 0},
//         {1, 1}};

//     vector<Vector> Y = {
//         {0},
//         {1},
//         {1},
//         {0}};

//     nn.train(X, Y, 10000);

//     cout << "\nXOR Predictions:\n";
//     for (auto &x : X)
//     {
//         Vector out = nn.forward(x);
//         cout << x[0] << " XOR " << x[1] << " = " << out[0] << endl;
//     }

//     return 0;
// }

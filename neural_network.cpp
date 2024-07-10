#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

namespace py = pybind11;

Eigen::VectorXd relu(const Eigen::VectorXd& input) {
    return input.array().max(0.0);
}

Eigen::VectorXd drelu(const Eigen::VectorXd& input) {
    return (input.array() > 0).cast<double>();
}

class Layer {
private:
    Eigen::MatrixXd weights;
    Eigen::VectorXd biases;
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> activation_function;
public:
    Layer(Eigen::MatrixXd weights, Eigen::VectorXd biases, std::string activation = "relu") 
        : weights(std::move(weights)), biases(std::move(biases)) {
        if (activation == "relu") {
            activation_function = relu;
        } else {
            throw std::invalid_argument("Unsupported activation function");
        }
    }

    Eigen::VectorXd forward(const Eigen::VectorXd& input) {
        return activation_function(weights * input + biases);
    }

    void set_weights(const Eigen::MatrixXd& weights) { this->weights = weights; }
    void set_biases(const Eigen::VectorXd& biases) { this->biases = biases; }
    void set_parameters(const Eigen::MatrixXd& weights, const Eigen::VectorXd& biases) {
        set_weights(weights);
        set_biases(biases);
    }
    const Eigen::MatrixXd get_weights() const { return weights; }
    const Eigen::VectorXd get_biases() const { return biases; }
    const std::function<Eigen::VectorXd(const Eigen::VectorXd&)> get_activation_function() const { return activation_function; }
};

class NeuralNetwork {
private:
    std::vector<Layer> layers;
    std::vector<Eigen::VectorXd> cache;
    std::vector<std::pair<Eigen::MatrixXd, Eigen::VectorXd>> gradient_table;
    std::vector<std::pair<Eigen::MatrixXd, Eigen::VectorXd>> momentum_table;

public:

    /**
    * @brief Adds a hidden layer with an ReLU activation function to the neural network
    */
    void add_layer(Layer& layer) {
        if (!layers.empty() && (layers.back().get_weights().rows() != layer.get_weights().cols())) {
            throw std::invalid_argument("Layer dimension do not fit the last layer");
        }
        layers.push_back(layer);
        gradient_table.resize(layers.size());
        momentum_table.resize(layers.size());

        momentum_table.back() = {Eigen::MatrixXd::Zero(layer.get_weights().rows(), layer.get_weights().cols()),
                                 Eigen::VectorXd::Zero(layer.get_biases().size())};

    }

    void train(const Eigen::VectorXd& x, const Eigen::VectorXd& y, const float& lr, const float& momentum) {
        if (layers.front().get_weights().cols() != x.size()){
            throw std::invalid_argument("Input size does not match the first layer");
        }
        Eigen::VectorXd h = x;
        cache.push_back(h);
        for (Layer& layer : layers) {
            h = layer.forward(h);
            cache.push_back(h);
        }
        py::print("Starting back propagation");
        back_prop(y, h);
        py::print("Back propagation successful, starting gradient descent");
        gradient_descent(lr, momentum);
        py::print("Gradient descent successful");
        cache.clear();
    }

    void back_prop(const Eigen::VectorXd& y, const Eigen::VectorXd& y_hat) {
        int num_layers = layers.size();
        Eigen::VectorXd g = -2 * (y-y_hat);
        for (int i=0; i<num_layers; ++i) {
            Eigen::VectorXd g_a = Eigen::VectorXd(g.array() * drelu(cache[num_layers-i]).array());
            Eigen::VectorXd g_b = g_a;
            Eigen::MatrixXd g_W = g_a * cache[num_layers-i-1].transpose();
            g = layers[num_layers-i-1].get_weights().transpose() * g_a;
            gradient_table[num_layers-i-1] = {g_W, g_b};
        }
    }

    void gradient_descent(const float& lr, const float& momentum = 0) {
        int num_layers = layers.size();
        if (momentum_table.size() != num_layers) {
            momentum_table.resize(num_layers);
            for (int i = 0; i < num_layers; ++i) {
                momentum_table[i] = {Eigen::MatrixXd::Zero(layers[i].get_weights().rows(), layers[i].get_weights().cols()),
                                     Eigen::VectorXd::Zero(layers[i].get_biases().size())};
            }
        }

        for (int i=0; i<num_layers; ++i) {
            Eigen::MatrixXd v_W = -lr * gradient_table[i].first;
            Eigen::VectorXd v_b = -lr * gradient_table[i].second;
            if (!momentum_table.empty()) {
                v_W += momentum * momentum_table[i].first;
                v_b += momentum * momentum_table[i].second;
            }
            momentum_table[i] = {v_W, v_b};
            Eigen::MatrixXd new_weight = layers[i].get_weights() + v_W;
            Eigen::VectorXd new_bias = layers[i].get_biases() + v_b;
            layers[i].set_parameters(new_weight, new_bias);
        }
    }

    /**
    * @brief Predicts the target variables from the given input variables.
    * @return predicted target vector
    */
    Eigen::VectorXd predict(const Eigen::VectorXd& input) {
        if (layers[0].get_weights().cols() != input.size()){
            throw std::invalid_argument("Input size does not match the first layer");
        }
        std::stringstream ss;
        Eigen::VectorXd h = input;
        for(Layer& layer : layers) {
            ss << layer.get_weights();
            ss << "\n \n";
            h = layer.forward(h);
        }
        py::print(ss.str());
        return h;
    }

    /**
    * @brief Returns the Mean Squared Error loss
    */
    double mse_loss(const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
        Eigen::VectorXd y_hat = predict(x);
        return (y - y_hat).array().square().mean();
    }

};

PYBIND11_MODULE(neural_network, m) {
    py::class_<Layer>(m, "Layer")
        .def(py::init<Eigen::MatrixXd&, Eigen::VectorXd&, std::string>(),
             py::arg("weights"), py::arg("biases"), py::arg("activation") = "relu");
    py::class_<NeuralNetwork>(m, "NeuralNetwork")
        .def(py::init<>())
        .def("add_layer", &NeuralNetwork::add_layer)
        .def("train", &NeuralNetwork::train)
        .def("predict", &NeuralNetwork::predict)
        .def("mse_loss", &NeuralNetwork::mse_loss);
}

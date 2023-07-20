use rand::Rng;

pub struct NeuralNetwork {
    input_layer_size: usize,
    hidden_layer_size: usize,
    output_layer_size: usize,
    weights_ih: Vec<Vec<f64>>,
    weights_ho: Vec<Vec<f64>>,
    bias_h: Vec<f64>,
    bias_o: Vec<f64>,
}

impl NeuralNetwork {
    pub fn new(
        input_layer_size: usize,
        hidden_layer_size: usize,
        output_layer_size: usize,
    ) -> Self {
        let mut rng = rand::thread_rng();

        // Initialize weights and biases with random values
        let weights_ih: Vec<Vec<f64>> = (0..hidden_layer_size)
            .map(|_| {
                (0..input_layer_size)
                    .map(|_| rng.gen_range(-1.0..1.0))
                    .collect()
            })
            .collect();
        let weights_ho: Vec<Vec<f64>> = (0..output_layer_size)
            .map(|_| {
                (0..hidden_layer_size)
                    .map(|_| rng.gen_range(-1.0..1.0))
                    .collect()
            })
            .collect();
        let bias_h: Vec<f64> = (0..hidden_layer_size)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        let bias_o: Vec<f64> = (0..output_layer_size)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();

        NeuralNetwork {
            weights_ih,
            weights_ho,
            bias_h,
            bias_o,
            input_layer_size: input_layer_size,
            hidden_layer_size: hidden_layer_size,
            output_layer_size: output_layer_size,
        }
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    pub fn feedforward(&self, input: &Vec<f64>) -> [Vec<f64>; 2] {
        // Calculate hidden layer values
        let mut hidden = vec![0.0; self.hidden_layer_size];
        for i in 0..self.hidden_layer_size {
            let mut sum = self.bias_h[i];
            for j in 0..self.input_layer_size {
                sum += self.weights_ih[i][j] * input[j];
            }
            hidden[i] = NeuralNetwork::sigmoid(sum);
        }

        // Calculate output layer values
        let mut output = vec![0.0; self.output_layer_size];
        for i in 0..self.output_layer_size {
            let mut sum = self.bias_o[i];
            for j in 0..self.hidden_layer_size {
                sum += self.weights_ho[i][j] * hidden[j];
            }
            output[i] = NeuralNetwork::sigmoid(sum);
        }

        [hidden, output]
    }

    pub fn train(&mut self, input: &Vec<f64>, target: &Vec<f64>, learning_rate: f64) {
        // Feedforward
        let [hidden, predicted] = self.feedforward(input);

        // Calculate output errors
        let mut output_errors = vec![0.0; self.output_layer_size];
        for i in 0..self.output_layer_size {
            let error = target[i] - predicted[i];
            output_errors[i] = predicted[i] * (1.0 - predicted[i]) * error;
        }

        // Calculate hidden layer errors
        let mut hidden_errors = vec![0.0; self.hidden_layer_size];
        for i in 0..self.hidden_layer_size {
            let mut error = 0.0;
            for j in 0..self.output_layer_size {
                error += output_errors[j] * self.weights_ho[j][i];
            }
            hidden_errors[i] = hidden[i] * (1.0 - hidden[i]) * error;
        }

        // Update weights and biases
        for i in 0..self.hidden_layer_size {
            for j in 0..self.input_layer_size {
                self.weights_ih[i][j] += learning_rate * hidden_errors[i] * input[j];
            }
        }

        for i in 0..self.output_layer_size {
            for j in 0..self.hidden_layer_size {
                self.weights_ho[i][j] += learning_rate * output_errors[i] * hidden[j];
            }
        }

        for i in 0..self.hidden_layer_size {
            self.bias_h[i] += learning_rate * hidden_errors[i];
        }

        for i in 0..self.output_layer_size {
            self.bias_o[i] += learning_rate * output_errors[i];
        }
    }
}

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DataProcessor:
    def __init__(self, file_path, input_columns, target_column):
        self.file_path = file_path
        self.input_columns = input_columns
        self.target_column = target_column
        self.data = self._load_data()
        self.inputs, self.target = self._prepare_data()

    def _load_data(self):
        return pd.read_excel(self.file_path)[self.input_columns + [self.target_column]].dropna()

    def _prepare_data(self):
        inputs = self.data[self.input_columns].values
        target = self.data[self.target_column].values
        normalized_inputs = (inputs - inputs.mean(axis=0)) / inputs.std(axis=0)
        return normalized_inputs, target


class ModelConfig:
    @staticmethod
    def get_hidden_layers():
        layers = int(input("Specify the number of hidden layers: "))
        return [int(input(f"Enter the number of nodes in hidden layer {i + 1}: ")) for i in range(layers)]


class NeuralNetwork:
    def __init__(self, input_dim, hidden_dims, output_dim):
        self.layers = self._initialize_weights(input_dim, hidden_dims, output_dim)

    def _initialize_weights(self, input_dim, hidden_dims, output_dim):
        weights = []
        prev_dim = input_dim
        for dim in hidden_dims:
            weights.append(np.random.rand(prev_dim, dim))
            prev_dim = dim
        weights.append(np.random.rand(prev_dim, output_dim))
        return weights

    def forward(self, X):
        for weight in self.layers:
            X = np.maximum(0, X @ weight)
        return X


class PSOParticle:
    def __init__(self, input_dim, hidden_dims, output_dim):
        self.mlp = NeuralNetwork(input_dim, hidden_dims, output_dim)
        self.position = [w.copy() for w in self.mlp.layers]
        self.velocity = [np.random.uniform(-0.1, 0.1, w.shape) for w in self.position]
        self.best_position = self.position
        self.best_score = float("inf")


class ParticleSwarmOptimizer:
    def __init__(self, X, y, hidden_layers, num_particles=20, iterations=150):
        self.X = X
        self.y = y
        self.hidden_layers = hidden_layers
        self.num_particles = num_particles
        self.iterations = iterations
        self.global_best_position = None
        self.global_best_score = float("inf")

    def optimize(self):
        particles = [PSOParticle(self.X.shape[1], self.hidden_layers, 1) for _ in range(self.num_particles)]

        for _ in range(self.iterations):
            for particle in particles:
                predictions = particle.mlp.forward(self.X).flatten()
                error = np.mean(np.abs(self.y - predictions))

                if error < particle.best_score:
                    particle.best_score = error
                    particle.best_position = [w.copy() for w in particle.position]
                if error < self.global_best_score:
                    self.global_best_score = error
                    self.global_best_position = [w.copy() for w in particle.position]

                inertia, cog, soc = 0.5, 2.0, 2.0
                for i in range(len(particle.position)):
                    r1, r2 = np.random.rand(*particle.position[i].shape), np.random.rand(*particle.position[i].shape)
                    particle.velocity[i] = (
                        inertia * particle.velocity[i]
                        + cog * r1 * (particle.best_position[i] - particle.position[i])
                        + soc * r2 * (self.global_best_position[i] - particle.position[i])
                    )
                    particle.position[i] += particle.velocity[i]

        return self.global_best_position


class CrossValidator:
    def __init__(self, model_cls, X, y, hidden_layers, folds=10):
        self.model_cls = model_cls
        self.X = X
        self.y = y
        self.hidden_layers = hidden_layers
        self.folds = folds
        self.fold_size = len(X) // folds
        self.mae_results = []

    def run(self):
        for fold in range(self.folds):
            test_idx = list(range(fold * self.fold_size, (fold + 1) * self.fold_size))
            train_idx = list(set(range(len(self.X))) - set(test_idx))

            X_train, y_train = self.X[train_idx], self.y[train_idx]
            X_test, y_test = self.X[test_idx], self.y[test_idx]

            optimizer = self.model_cls(X_train, y_train, self.hidden_layers)
            best_weights = optimizer.optimize()

            model = NeuralNetwork(X_train.shape[1], self.hidden_layers, 1)
            model.layers = best_weights
            y_pred = model.forward(X_test).flatten()

            mae = np.mean(np.abs(y_test - y_pred))
            self.mae_results.append(mae)
            print(f"Fold {fold + 1}/{self.folds} - Mean Absolute Error: {mae}")

        return np.mean(self.mae_results)

    def plot_mae(self):
        plt.bar(range(1, self.folds + 1), self.mae_results, color='skyblue')
        plt.title("Mean Absolute Error per Fold")
        plt.xlabel("Fold Number")
        plt.ylabel("Mean Absolute Error")
        plt.show()


# Execution
file_path = 'AirQualityUCI.xlsx'
input_columns = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']
target_column = 'C6H6(GT)'

data = DataProcessor(file_path, input_columns, target_column)
hidden_layers = ModelConfig.get_hidden_layers()
cross_validator = CrossValidator(ParticleSwarmOptimizer, data.inputs, data.target, hidden_layers)

average_mae = cross_validator.run()
print(f"Average MAE: {average_mae}")
cross_validator.plot_mae()

final_optimizer = ParticleSwarmOptimizer(data.inputs, data.target, hidden_layers)
best_weights = final_optimizer.optimize()
final_model = NeuralNetwork(data.inputs.shape[1], hidden_layers, 1)
final_model.layers = best_weights
predictions = final_model.forward(data.inputs).flatten()

plt.plot(data.target, label="Actual Benzene Concentration", color="blue")
plt.plot(predictions, label="Predicted Benzene Concentration", color="orange")
plt.title("Benzene Concentration Prediction")
plt.xlabel("Sample Index")
plt.ylabel("Benzene Concentration (µg/m³)")
plt.legend()
plt.show()

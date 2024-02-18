import numpy as np
from sklearn.utils import shuffle
from PIL import Image
import os

def load_custom_data():
    data_folder = 'C:/Users/arthu/OneDrive/Documents/extracted_images/first_steps'
    images = []
    labels = []
    i=0
    for label in os.listdir(data_folder):
        label_path = os.path.join(data_folder, label)

        if os.path.isdir(label_path):
            for image_file in os.listdir(label_path):
                image_path = os.path.join(label_path, image_file)

                # Charger l'image
                image = Image.open(image_path)

                # Aplatir l'image en un vecteur
                image_vector = np.array(image).flatten()

                # Ajouter l'image et l'étiquette aux listes
                images.append(image_vector)
                labels.append(i)
            i += 1

    # Convertir les listes en tableaux NumPy
    X = np.array(images, dtype='float64')/255
    y = np.array(labels, dtype='int')

    # Mélanger les données
    X, y = shuffle(X, y)

    return X, y

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(x):
    """Dérivée de la fonction sigmoid."""
    return sigmoid(x) * (1.0 - sigmoid(x))


def to_one_hot(y, k):
    """Convertit un entier en vecteur "one-hot".

    to_one_hot(5, 10) -> (0, 0, 0, 0, 1, 0, 0, 0, 0)

    """
    one_hot = np.zeros(k, dtype=int)
    one_hot[y] = 1
    return one_hot




class Layer:
    """Une seule couche de neurones."""
    def __init__(self, size, input_size):
        self.size = size
        self.input_size = input_size

        # Les poids sont représentés par une matrice de n lignes
        # et m colonnes. n = le nombre de neurones, m = le nombre de
        # neurones dans la couche précédente.
        self.weights = np.random.randn(size, input_size)

        # Un biais par neurone
        self.biases = np.random.randn(size)

    # Résultat du calcul de chaque neurone.
    # Il est important de noter que `data` est un vecteur (normalement, de
    # longueur `self.input_size`, et que nous retournons un vecteur de
    # taille `self.size`.
    def forward(self, data):
        aggregation = self.aggregation(data)
        activation = self.activation(aggregation)
        return activation

    # Calcule la somme des entrées pondérées + biais pour chaque neurone.
    # Plutôt que d'utiliser une boucle for, nous tirons parti du calcul
    # matriciel qui permet d'effectuer toutes ces opérations d'un coup.
    def aggregation(self, data):
        result = np.dot(self.weights, data) + self.biases
        return result

    # Passe les valeurs aggrégées dans la moulinette de la fonction
    # d'activation.
    # `x` est un vecteur de longueur `self.size`, et nous retournons un
    # vecteur de même dimension.
    def activation(self, x):
        return sigmoid(x)

    # Dérivée de la fonction d'activation.
    def activation_prime(self, x):
        return sigmoid_prime(x)

    # Mise à jour des poids à partir du gradient (algo du gradient)
    def update_weights(self, gradient, learning_rate):
        self.weights -= learning_rate * gradient

    def update_biases(self, gradient, learning_rate):
        self.biases -= learning_rate * gradient


class Network:
    """Un réseau constitué de couches de neurones."""
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.layers = []

    def add_layer(self, size):
        if len(self.layers) > 0:
            input_dim = self.layers[-1].size
        else:
            input_dim = self.input_dim

        self.layers.append(Layer(size, input_dim))

    # Propage les données d'entrée d'une couche à l'autre.
    def feedforward(self, input_data):
        activation = input_data
        for layer in self.layers:
            activation = layer.forward(activation)
        return activation

    # Retourne l'index du neurone de sortie qui a la plus haute valeur, ce
    # qui revient à indiquer quelle classe est sélectionnée par le réseau.
    def predict(self, input_data):
        return np.argmax(self.feedforward(input_data))

    # Évalue la performance du réseau à partir d'un set d'exemples.
    # Retourne un nombre entre 0 et 1.
    def evaluate(self, X, Y):
        results = [1 if self.predict(x) == y else 0 for (x, y) in zip(X, Y)]
        accuracy = sum(results) / len(results)
        return accuracy

    # Fonction d'entraînement du modèle.
    # rétropropagation sur un certain nombre d'exemples (batch_size) avant
    # de calculer un gradient moyen, et de mettre à jour les poids.
    def train(self, X, Y, steps=30, learning_rate=0.3, batch_size=100):
        n = Y.size
        for i in range(steps):
            # Mélangeons les données parce que… parce que.
            X, Y = shuffle(X, Y)
            for batch_start in range(0, n, batch_size):
                X_batch, Y_batch = X[batch_start:batch_start + batch_size], Y[batch_start:batch_start + batch_size]
                self.train_batch(X_batch, Y_batch, learning_rate)

    # Cette fonction combine les algos du retropropagation du gradient +
    # gradient descendant.
    def train_batch(self, X, Y, learning_rate):
        # Initialise les gradients pour les poids et les biais.
        weight_gradient = [np.zeros(layer.weights.shape) for layer in self.layers]
        bias_gradient = [np.zeros(layer.biases.shape) for layer in self.layers]

        # On fait tourner l'algo de rétropropagation pour calculer les
        # gradients un certain nombre de fois. On fera la moyenne ensuite.
        for (x, y) in zip(X, Y):
            x = x.astype('float64')  # Conversion ajoutée ici
            new_weight_gradient, new_bias_gradient = self.backprop(x, y)
            weight_gradient = [wg + nwg for wg, nwg in zip(weight_gradient, new_weight_gradient)]
            bias_gradient = [bg + nbg for bg, nbg in zip(bias_gradient, new_bias_gradient)]

        # C'est ici qu'on calcule les moyennes des gradients calculés
        avg_weight_gradient = [wg / Y.size for wg in weight_gradient]
        avg_bias_gradient = [bg / Y.size for bg in bias_gradient]

        # Il ne reste plus qu'à mettre à jour les poids et biais en
        # utilisant l'algo du gradient descendant.
        for layer, weight_gradient, bias_gradient in zip(self.layers,
                                                         avg_weight_gradient,
                                                         avg_bias_gradient):
            layer.update_weights(weight_gradient, learning_rate)
            layer.update_biases(bias_gradient, learning_rate)

    # L'algorithme de rétropropagation du gradient.
    # C'est là que tout le boulot se fait.
    def backprop(self, x, y):

        # On va effectuer une passe vers l'avant, une passe vers l'arrière
        # On profite de la passe vers l'avant pour stocker les calculs
        # intermédiaires, qui seront réutilisés par la suite.
        aggregations = []
        activation = x
        activations = [activation]

        # Propagation pour obtenir la sortie
        for layer in self.layers:
            aggregation = layer.aggregation(activation)
            aggregations.append(aggregation)
            activation = layer.activation(aggregation)
            activations.append(activation)

        # Calculons la valeur delta (δ) pour la dernière couche
        # en appliquant les équations détaillées plus haut.
        target = to_one_hot(int(y), 3)
        delta = self.get_output_delta(aggregation, activation, target)
        deltas = [delta]

        # Phase de rétropropagation pour calculer les deltas de chaque
        # couche
        # On utilise une implémentation vectorielle des équations.
        nb_layers = len(self.layers)
        for l in reversed(range(nb_layers - 1)):
            layer = self.layers[l]
            next_layer = self.layers[l + 1]
            activation_prime = layer.activation_prime(aggregations[l])
            delta = activation_prime * np.dot(next_layer.weights.transpose(), delta)
            deltas.append(delta)

        # Nous sommes parti de l'avant-dernière couche pour remonter vers
        # la première. deltas[0] contient le delta de la dernière couche.
        # Nous l'inversons pour faciliter la gestion des indices plus tard.
        deltas = list(reversed(deltas))

        # On utilise maintenant les deltas pour calculer les gradients.
        weight_gradient = []
        bias_gradient = []
        for l in range(len(self.layers)):

            prev_activation = activations[l].astype('float64')  # Conversion ajoutée ici
            weight_gradient.append(np.outer(deltas[l].astype('float64'), prev_activation))
            bias_gradient.append(deltas[l].astype('float64'))

        return weight_gradient, bias_gradient

    def get_output_delta(self, z, a, target):
        return a - target


if __name__ == '__main__':
    # Découpons notre base de données en deux.
    # Une partie pour l'entraînement du réseau, l'autre pour vérifier
    X, Y = load_custom_data()
    X_train, Y_train = X[:20000], Y[:20000]
    X_test, Y_test = X[20000:], Y[20000:]
    #
    net = Network(input_dim=2025)
    net.add_layer(200)
    net.add_layer(3)

    accuracy = net.evaluate(X_test, Y_test)
    print('Performance initiale : {:.2f}%'.format(accuracy * 100.0))

    for i in range(50):
        net.train(X_train, Y_train, steps=1, learning_rate=0.3)
        accuracy = net.evaluate(X_test, Y_test)
        print('Nouvelle performance : {:.2f}%'.format(accuracy * 100.0))






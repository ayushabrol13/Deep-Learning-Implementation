# Importing the libraries
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
import pickle

class MultiLayerPerceptron:
    def __init__(self, input_layer, hidden_layer1, hidden_layer2, output_layer, learning_rate=0.01):
        
        # Initializing the weights and biases
        self.weights1 = np.random.rand(input_layer, hidden_layer1)
        self.bias1 = np.random.rand(1, hidden_layer1)
        self.weights2 = np.random.rand(hidden_layer1, hidden_layer2)
        self.bias2 = np.random.rand(1, hidden_layer2)
        self.weights3 = np.random.rand(hidden_layer2, output_layer)
        self.bias3 = np.random.rand(1, output_layer)
        self.learning_rate = learning_rate
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        x[x<=0] = 0
        x[x>0] = 1
        return x

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def fit(self, X, y):     
        for j in range(X.shape[0]):
            # Forward pass
            self.input_layer = X[j].reshape(X[j].shape[0], 1) # (4, 1)
            self.output_layer1 = self.relu(np.dot(self.weights1.T, self.input_layer) + self.bias1.T) # (4, 1)
            self.output_layer2 = self.relu(np.dot(self.weights2.T, self.output_layer1) + self.bias2.T) # (5, 1)
            self.output_layer = self.softmax(np.dot(self.weights3.T, self.output_layer2) + self.bias3.T) # (3, 1)


            # Backward pass
            val = y[j]
            val = val.reshape(val.shape[0], 1) # (3, 1)
            self.error = (val - self.output_layer) # (3, 1)
    
            self.output_layer_error = self.error # (3, 1)
            self.output_layer2_error = self.relu_derivative(self.output_layer2) * np.dot(self.weights3, self.output_layer_error) 
            self.output_layer1_error = self.relu_derivative(self.output_layer1) * np.dot(self.weights2, self.output_layer2_error) 
            

            # Updating the weights and biases
            self.weights3 += self.learning_rate * np.dot(self.output_layer2, self.output_layer_error.T)
            self.bias3 += self.learning_rate * np.sum(self.output_layer_error, axis=0, keepdims=True) 
            self.weights2 += self.learning_rate * np.dot(self.output_layer1, self.output_layer2_error.T)
            self.bias2 += self.learning_rate * np.sum(self.output_layer2_error, axis=0, keepdims=True) 
            self.weights1 += self.learning_rate * np.dot(self.input_layer, self.output_layer1_error.T) 
            self.bias1 += self.learning_rate * np.sum(self.output_layer1_error.T, axis=0, keepdims=True) 



    def train(self, X, y, epochs):
        train_loss = []
        print("Training Started...")
        for i in range(epochs):
            self.fit(X, y)
            print("Epoch:", i+1, "Error: ", np.mean(np.abs(self.error))) 
            print("---------------------------------")
            train_loss.append(np.mean(np.abs(self.error)))
        print("Training Completed...")

        return train_loss
    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            self.input_layer = X[i].reshape(X[i].shape[0], 1) # (4, 1)
            self.output_layer1 = self.relu(np.dot(self.weights1.T, self.input_layer) + self.bias1.T) # (4, 1)
            self.output_layer2 = self.relu(np.dot(self.weights2.T, self.output_layer1) + self.bias2.T) # (5, 1)
            self.output_layer = self.relu(np.dot(self.weights3.T, self.output_layer2) + self.bias3.T) # (3, 1)

            predictions.append(self.output_layer)
        return np.array(predictions)
    

# Load the iris dataset
iris_df = datasets.load_iris()

# Separating the features and the target
X = iris_df.data
y = iris_df.target

y_encoded = np.zeros((y.size, y.max()+1))
y_encoded[np.arange(y.size),y] = 1

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Initializing the model
input_layer = X_train.shape[1]
hidden_layer1 = 4
hidden_layer2 = 5
output_layer = 3
epochs = 100

model = MultiLayerPerceptron(input_layer, hidden_layer1, hidden_layer2, output_layer)
train_loss = model.train(X_train, y_train, epochs)

# Saving the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved successfully...")

# Plotting the loss
fig, ax = plt.subplots()
ax.plot(np.arange(epochs), train_loss)
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
ax.set_title("Training Loss")
plt.savefig('training_loss.png')

# Predicting the test data
predictions = model.predict(X_test)
predictions = np.argmax(predictions, axis=1).flatten()
test_classes = np.argmax(y_test, axis=1)

# Calculating the accuracy
accuracy = accuracy_score(test_classes, predictions)*100

print("Accuracy on the testing data: ", accuracy, "%")

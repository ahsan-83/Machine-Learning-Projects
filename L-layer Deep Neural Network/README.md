# L-layer Deep Neural Network : Cat Image Classification

<img src="https://raw.githubusercontent.com/ahsan-83/Machine-Learning-Projects/main/L-layer%20Deep%20Neural%20Network/images/deep_neural_network.jpg" width="800"/>

L-layer Deep Neural Network Binary Classification Model is developed from scratch with python for Cat vs non-cat image classification.

## L-layer Deep Neural Network Model Architecture

- Initialize weight and bias parameters for  L-layer deep neural network
- Compute Linear Forward Activation *LINEAR->RELU* for $L-1$ layers and *LINEAR->SIGMOID* for last layer
- Compute the loss with Binary Cross Entropy cost function
- Compute gradients of loss function respect to parameters of hidden layers in Linear Backward Activation
- Update hidden layer parameters for gradient descent using learning rate

<img src="https://raw.githubusercontent.com/ahsan-83/Machine-Learning-Projects/main/L-layer%20Deep%20Neural%20Network/images/model_architecture.png" width="800">

## L-layer Deep Neural Network Model Implementation

**Initialize Parameters**

Parameter initialization (Weights and Biases) for L-layer neural network implemented in `initialize_parameters_deep`

- Random initialization of weights using `np.random.randn(shape)`
- Zeros initialization of biases using `np.zeros(shape)`

**Forward Propagation**

- Linear Forward computes the equation $Z^{[l]} = W^{[l]}A^{[l-1]} +b^{[l]}$ in `linear_forward` where $A^{[0]} = X$. 
- Linear Activation Forwad computes activation function of Linear Forward in *LINEAR->ACTIVATION* layer <br/><br/> $$A^{[l]} = g(Z^{[l]}) = g(W^{[l]}A^{[l-1]} +b^{[l]})$$
- **Sigmoid**: $\sigma(Z) = \sigma(W A + b) = \frac{1}{ 1 + e^{-(W A + b)}}$. 
- **ReLU**: $A = RELU(Z) = max(0, Z)$. 
- L-layer Model Forward Propagation involves *LINEAR->RELU* activation in  $L-1$ layers and *LINEAR->SIGMOID* activation in last layer as implemented in `L_model_forward`
- Forward Propagation prediction output $\hat{Y}$ is denoted as **`AL`** $A^{[L]} = \sigma(Z^{[L]}) = \sigma(W^{[L]} A^{[L-1]} + b^{[L]})$

<img src="https://raw.githubusercontent.com/ahsan-83/Machine-Learning-Projects/main/L-layer%20Deep%20Neural%20Network/images/forward_propagation.png" width="800">

**Cost Function**

- Forward Propagation prediction output is evaluated using binary cross entropy cost function formula as implemented in `compute_cost`

$$ J = -\frac{1}{m} \sum\limits_{i = 1}^{m} (y^{(i)}\log\left(a^{[L] (i)}\right) + (1-y^{(i)})\log\left(1- a^{[L](i)}\right))$$

**Backward Propagation**

- Backward Propagation computes gradients of the loss function with respect to the parameters. 

<img src="https://raw.githubusercontent.com/ahsan-83/Machine-Learning-Projects/main/L-layer%20Deep%20Neural%20Network/images/backward_propagation.png" width="700">

- Linear Backward computes  $dW^{[l]}, db^{[l]}, dA^{[l-1]}$ from Linear Forward equation $Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$ <br/> assuming $dZ^{[l]} = \frac{\partial \mathcal{L} }{\partial Z^{[l]}}$ is already computed for layer $l$

<img src="https://raw.githubusercontent.com/ahsan-83/Machine-Learning-Projects/main/L-layer%20Deep%20Neural%20Network/images/linear_backward_propagation.png" width="400">

$$ dW^{[l]} = \frac{\partial \mathcal{J} }{\partial W^{[l]}} = \frac{1}{m} dZ^{[l]} A^{[l-1] T}$$

$$ db^{[l]} = \frac{\partial \mathcal{J} }{\partial b^{[l]}} = \frac{1}{m} \sum_{i = 1}^{m} dZ^{[l](i)}$$

$$ dA^{[l-1]} = \frac{\partial \mathcal{L} }{\partial A^{[l-1]}} = W^{[l] T} dZ^{[l]}$$

- Linear Activation Backward computes gradient $dZ^{[l]} = \frac{\partial \mathcal{L} }{\partial Z^{[l]}} = dA^{[l]} * g'(Z^{[l]})$ from Linear Activation Forward equarion $A^{[l]} = g(Z^{[l]})$

- Gradient $dZ^{[l]}$ computed for different activation function is used to get $dW^{[l]}, db^{[l]}, dA^{[l-1]}$ from `linear_backward` function

- **`sigmoid_backward`** and **`relu_backward`** computes $dZ^{[l]}$ for Sigmoid and RELU activation function

- `L_model_backward` computes gradients of loss function respect to parameters in each hidden layer of deep neural network backward. 

<img src="https://raw.githubusercontent.com/ahsan-83/Machine-Learning-Projects/main/L-layer%20Deep%20Neural%20Network/images/backward_propagation_layers.png" width="500">

At first, derivative of loss function respect to prediction **`dAL`** is computed with equation 

$$ \frac{\partial \mathcal{L} }{\partial a} = -\frac{y}{a} + \frac{(1-y)}{(1-a)}$$

- $dW^{[L]}, db^{[L]}, dA^{[L-1]}$ are computed from **`dAL`** for *LINEAR->SIGMOID* layer using `linear_activation_backward`

- $dW^{[l]}, db^{[l]}, dA^{[l-1]}$ are computed from $dA^{[l]}$ for *LINEAR->RELU* $L-1$ layers using `linear_activation_backward`

**Parameter Update**

- Neural network parameters are updated from gradients as follows

$$ W^{[l]} = W^{[l]} - \alpha \text{ } dW^{[l]} $$

$$ b^{[l]} = b^{[l]} - \alpha \text{ } db^{[l]} $$

- Learning rate $\alpha$ updates parameters $W^{[l]}$ and $b^{[l]}$ for gradient descent in every layer $l$

**L-layer Deep Neural Network Model** 

Neural network model is developed by stacking above steps

1. Initialize Parameters
2. Loop for number of iterations
    * Forward Propagation
    * Compute Cost
    * Backward Propagation
    * Parameter Update
3. Predict output with trained parameters 

**`L_layer_model`** trains L-layer neural network with labeled dataset (X,Y) and output hidden layer parameters

**L-layer Deep Neural Network Model Prediction**

After training model with train dataset we get trained parameters for hidden layer units. <br/>

Model prediction evaluated for test dataset using model parameters in `L_model_forward` as implemented in `L_layer_model_prediction`

- Loss is calculated from model prediciton with `compute_cost` function
- Accuracy is calculated from model prediciton with equation

$$ Accuracy = \frac{1}{m} \sum\limits_{i = 1}^{m} (\hat Y^{(i)} == Y^{(i)})$$

- Precision, Recall and F1 Score is calculated from model prediciton with equations

$$ Precision = \frac{True Positive}{(True Positive + False Positive)}$$

$$ Recall = \frac{True Positive}{(True Positive + False Negative)}$$

$$ F1 Score = \frac{2 \times Precision \times Recall}{(Precision + Recall)}$$

## Cat Image Dataset

"Cat vs non-Cat" dataset stored as `train_catvnoncat.h5` and `test_catvnoncat.h5` contains 64 x 64 dimension labelled images.

Image shape : (64, 64, 3)

- Reshape train and test image dataset before training model

<img src="https://raw.githubusercontent.com/ahsan-83/Machine-Learning-Projects/main/L-layer%20Deep%20Neural%20Network/images/cat_image_reshape.png"/>

- Normalize image data to have feature values in range [0,1]

## Model Training

L-layer Deep Neural Network Binary Classification Model contains multiple layers for Cat Image Classification.

Model contains 3 hidden layers with [20, 7, 5] units, input layer with 12288 units as flattened image size and output layer with 1 unit.

- Learning Rate = 0.0075
- Optimization Algo = Gradient Discent
- Loss = Binary Crossentropy
- Model Training Iteration = 3000

**Model Training Accuracy and Loss**

Model training accuracy 99.5 %

<img src="https://raw.githubusercontent.com/ahsan-83/Machine-Learning-Projects/main/L-layer%20Deep%20Neural%20Network/images/model_loss.png"/>


## Model Evaluation

Cat vs non-cat classification model is evaluated by computing Accuracy, Precision, Recall, F1 Score for test dataset from `L_layer_model_prediction`

Metrics | Result
--- | --- |
Accuracy | 78.0 %
Precision |  0.823
Recall |  0.848
F1 Score |  0.835

- Cat vs non-cat classification model achieved 78% accuracy and 0.83 F1 Score


















































import numpy as np
import matplotlib.pyplot as plt

def softmax_rows(X):
  """
  Softmax of X
  """
  N, C = X.shape
  X_new = X - np.reshape(np.max(X,1), (N,1))
  exp = np.exp(X_new)
  return exp/np.reshape(exp.sum(1), (N,1))

def init_two_layer_model(input_size, hidden_size, output_size):
  """
  Initialize the weights and biases for a two-layer fully connected neural
  network. The net has an input dimension of D, a hidden layer dimension of H,
  and performs classification over C classes. Weights are initialized to small
  random values and biases are initialized to zero.

  Inputs:
  - input_size: The dimension D of the input data
  - hidden_size: The number of neurons H in the hidden layer
  - ouput_size: The number of classes C

  Returns:
  A dictionary mapping parameter names to arrays of parameter values. It has
  the following keys:
  - W1: First layer weights; has shape (D, H)
  - b1: First layer biases; has shape (H,)
  - W2: Second layer weights; has shape (H, C)
  - b2: Second layer biases; has shape (C,)
  """
  # initialize a model
  model = {}
  model['W1'] = 0.00001 * np.random.randn(input_size, hidden_size)
  model['b1'] = np.zeros(hidden_size)
  model['W2'] = 0.00001 * np.random.randn(hidden_size, output_size)
  model['b2'] = np.zeros(output_size)
  return model

def two_layer_net(X, model, y=None, reg=0.0):
  """
  Compute the loss and gradients for a two layer fully connected neural network.
  The net has an input dimension of D, a hidden layer dimension of H, and
  performs classification over C classes. We use a softmax loss function and L2
  regularization the the weight matrices. The two layer net should use a ReLU
  nonlinearity after the first affine layer.

  The two layer net has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each
  class.

  Inputs:
  - X: Input data of shape (N, D). Each X[i] is a training sample.
  - model: Dictionary mapping parameter names to arrays of parameter values.
    It should contain the following:
    - W1: First layer weights; has shape (D, H)
    - b1: First layer biases; has shape (H,)
    - W2: Second layer weights; has shape (H, C)
    - b2: Second layer biases; has shape (C,)
  - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
    an integer in the range 0 <= y[i] < C. This parameter is optional; if it
    is not passed then we only return scores, and if it is passed then we
    instead return the loss and gradients.
  - reg: Regularization strength.

  Returns:
  If y not is passed, return a matrix scores of shape (N, C) where scores[i, c]
  is the score for class c on input X[i].

  If y is not passed, instead return a tuple of:
  - loss: Loss (data loss and regularization loss) for this batch of training
    samples.
  - grads: Dictionary mapping parameter names to gradients of those parameters
    with respect to the loss function. This should have the same keys as model.
  """

  # unpack variables from the model dictionary
  W1,b1,W2,b2 = model['W1'], model['b1'], model['W2'], model['b2']
  N, D = X.shape

  # compute the forward pass
  hidden = np.dot(X, W1) + b1 # N x H + H x 1 (Broadcasting)
  mask = hidden < 0
  hidden[mask] = 0 # RelU
  relu_out = hidden # N x H
  scores = np.dot(relu_out, W2) + b2 # N x C
  #############################################################################
  # TODO: Perform the forward pass, computing the class scores for the input. #
  # Store the result in the scores variable, which should be an array of      #
  # shape (N, C).                                                             #
  #############################################################################
  
  # If the targets are not given then jump out, we're done
  if y is None:
    return scores

  # compute the loss
  softmax_out = softmax_rows(scores)
  softmax_loss  = -(1/N)*np.sum(np.log(softmax_out[np.arange(N), y] + 1e-6)) # Softmax loss
  regularization_loss = np.sum(W1*W1) + np.sum(W2*W2)
  regularization_loss *= 0.5*reg
  loss = softmax_loss + regularization_loss

  #############################################################################
  # TODO: Finish the forward pass, and compute the loss. This should include  #
  # both the data loss and L2 regularization for W1 and W2. Store the result  #
  # in the variable loss, which should be a scalar. Use the Softmax           #
  # classifier loss. So that your results match ours, multiply the            #
  # regularization loss by 0.5                                                #
  #############################################################################
  # pass
  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################

  # compute the gradients
  grads = {}
  backprop_start = softmax_out # N x C
  backprop_start[np.arange(N), y] -= 1
  grads['W2'] = (1/N) * np.dot(relu_out.T, backprop_start) # H x N, N x C = H x C
  grads['W2'] += reg*W2
  grads['b2'] = (1/N)*np.sum(backprop_start, axis = 0)
  grad_relu_out = np.dot(backprop_start, model['W2'].T) # N * C, C * H = N * H
  grad_relu_in = grad_relu_out
  grad_relu_in[mask] = 0
  grads['W1'] = (1/N) * np.dot(X.T, grad_relu_in) # D * N, N * H = D * H
  grads['W1'] += reg*W1
  grads['b1'] = (1/N) * np.sum(grad_relu_in, axis = 0)

  #############################################################################
  # TODO: Compute the backward pass, computing the derivatives of the weights #
  # and biases. Store the results in the grads dictionary. For example,       #
  # grads['W1'] should store the gradient on W1, and be a matrix of same size #
  #############################################################################
  # pass
  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################

  return loss, grads


if __name__ == '__main__':
  # pass
  N = 3
  C = 5
  W = np.random.randn(N, C)
  # print(W)
  # 
  print(np.reshape(np.max(W,1), (N,1)))
  W = W - np.reshape(np.max(W,1), (N,1))
  # print(W)

  # B = np.array([1,2,3,4,5])
  # print(W + B)
  # W = np.array([[1,2,3], [-1,2,3]])
  # W[W < 0] = 0
  # print(W)

  # model = init_two_layer_model(2,5,10)
  # for matrix in model:
  #   print(model[matrix])
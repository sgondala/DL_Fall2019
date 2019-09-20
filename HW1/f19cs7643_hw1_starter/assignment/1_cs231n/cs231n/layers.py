import numpy as np

def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
  We multiply this against a weight matrix of shape (D, M) where
  D = \prod_i d_i

  Inputs:
  x - Input data, of shape (N, d_1, ..., d_k)
  w - Weights, of shape (D, M)
  b - Biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  N = x.shape[0]
  X = np.reshape(x, (N, -1))
  out = np.dot(X, w) + b
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  # pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  dx, dw, db = None, None, None
  x, w, b = cache
  N = x.shape[0]
  X = np.reshape(x, (N, -1))
  dw = np.dot(X.T, dout)
  dx = np.dot(dout, w.T)
  dx = np.reshape(dx, x.shape)
  db = np.sum(dout, axis = 0)
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  mask = x < 0
  out = x.copy()
  out[mask] = 0
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  mask = x < 0
  dx = dout
  dx[mask] = 0
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  # pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx

def convolve(data_point, filter, stride):
  """
  Input:
  - data_point: Shape is (C, H, W) or (H, W)
  - filter: Shape is (C, HH, WW) or (HH, WW)
  - stride: Integer
  Output:
  - Integer whose value is convolution
  """
  output = 0
  r = 0
  if len(filter.shape) == 2:
    filter_here = np.reshape(filter, (1, *filter.shape))
    data_point_here = np.reshape(data_point, (1, *data_point.shape))
  elif len(filter.shape) == 3:
    filter_here = filter
    data_point_here = data_point
  
  assert filter_here.shape[0] == data_point_here.shape[0]

  C, HH, WW = filter_here.shape
  _, H, W = data_point_here.shape
  output = np.zeros((1 + (H - HH) // stride, 1 + (W - WW) // stride))
  r_index = 0
  while r + HH <= H:
    c = 0
    c_index = 0
    while c + WW <= W:
      value_here = np.sum(filter_here[:, :, :] * data_point_here[:, r:r + HH, c:c + WW])
      output[r_index][c_index] = value_here
      c += stride
      c_index += 1
    r += stride
    r_index += 1
  return output

def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  assert x.shape[1] == w.shape[1]
  stride = conv_param['stride']
  pad = conv_param['pad']
  N, C, H, W = x.shape
  F, _, HH, WW = w.shape
  out = np.zeros((N, F, 1 + (H + 2 * pad - HH) // stride, 1 + (W + 2 * pad - WW) // stride))
  index_point = 0
  for data_point in x:
    data_point_padded = np.pad(data_point, ((0,0), (pad,pad), (pad,pad)), 'constant', constant_values=(0))
    index_filter = 0
    for filter in w:
      value_for_filter = convolve(data_point_padded, filter, stride)
      value_for_filter += b[index_filter]
      out[index_point][index_filter] = value_for_filter
      index_filter += 1
    index_point += 1
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  x, w, b, conv_param = cache
  print("X shape", x.shape)
  print("W shape", w.shape)
  print("dout shape", dout.shape)

  stride = conv_param['stride']
  pad = conv_param['pad']
  
  dw = np.zeros_like(w)

  for index_point in range(len(x)):
    data_point_padded = np.pad(x[index_point], ((0,0), (pad,pad), (pad,pad)), 'constant', constant_values=(0))
    dout_here = dout[index_point]
    for index_filter in range(len(w)):
      dout_filter = dout_here[index_filter]
      for index_channel in range(len(data_point_padded)):
        channel = data_point_padded[index_channel]
        dw_channel = convolve(channel, dout_filter, stride)
        dw[index_filter][index_channel] += dw_channel

  db = dout.sum(0).sum(1).sum(1)
  
  dx = np.zeros_like(x)
  for index_point in range(len(dout)):
    dout_point = dout[index_point]
    data_point = x[index_point]
    for index_filter in range(len(w)):
      current_filter = w[index_filter]
      dout_filter = dout_point[index_filter]
      for index_channel in range(len(current_filter)):
        filter_channel = current_filter[index_channel].copy()
        filter_channel = np.flip(filter_channel, (0,1))
        pad_size = (filter_channel.shape[0] - 1) // 2
        dout_filter_padded = np.pad(dout_filter, ((pad_size,pad_size), (pad_size,pad_size)), 'constant', constant_values=(0))
        convolution_out = convolve(dout_filter_padded, filter_channel, stride)
        dx[index_point][index_channel] += convolution_out

  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx

if __name__ == '__main__':
  data_point = np.array([[[0, 1, 2, 3], [4,5,6,7], [8,9,10,11], [12,13,14,15]]])
  filter = np.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]]])
  print(convolve(data_point, filter, 2))
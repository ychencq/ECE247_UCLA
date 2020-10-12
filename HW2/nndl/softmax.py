import numpy as np

class Softmax(object):

  def __init__(self, dims=[10, 3073]):
    self.init_weights(dims=dims)

  def init_weights(self, dims):
    """
	Initializes the weight matrix of the Softmax classifier.  
	Note that it has shape (C, D) where C is the number of 
	classes and D is the feature size.
	"""
    self.W = np.random.normal(size=dims) * 0.0001

  def loss(self, X, y):
    # Initialize the loss to zero.
    loss = 0.0
    # ================================================================ #
    # YOUR CODE HERE:
	#   Calculate the normalized softmax loss.  Store it as the variable loss.
    #   (That is, calculate the sum of the losses of all the training 
    #   set margins, and then normalize the loss by the number of 
	#	training examples.)
    # ================================================================ #
    num_train = X.shape[0] 
    scores = np.dot(self.W, X.T)
    for i in range(num_train):
      score = scores[:, i]
      score -= np.max(score)
      loss -= score[y[i]]
      sum_i = np.sum(np.exp(score))       
      loss += np.log(sum_i)
    loss /= num_train    
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #
    return loss

  def loss_and_grad(self, X, y):
    # Initialize the loss and gradient to zero.
    loss = 0.0
    grad = np.zeros_like(self.W)  
    # ================================================================ #
    # YOUR CODE HERE:
	#   Calculate the softmax loss and the gradient. Store the gradient
	#   as the variable grad.
    # ================================================================ #
    num_train = X.shape[0]
    scores = np.dot(self.W, X.T)
    for i in range(num_train):
      score = scores[:, i]
      score -= np.max(score)
      loss -= score[y[i]]
      sum_i = np.sum(np.exp(score))       
      loss += np.log(sum_i)

      for j in range(10):
        grad[j] += (np.exp(score[j]) / sum_i) * X[i]
      grad[y[i]] -= X[i]
    loss /= num_train
    grad /= num_train    
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #
    return loss, grad

  def grad_check_sparse(self, X, y, your_grad, num_checks=10, h=1e-5):
    """
    sample a few random elements and only return numerical
    in these dimensions.
    """
  
    for i in np.arange(num_checks):
      ix = tuple([np.random.randint(m) for m in self.W.shape])
  
      oldval = self.W[ix]
      self.W[ix] = oldval + h # increment by h
      fxph = self.loss(X, y)
      self.W[ix] = oldval - h # decrement by h
      fxmh = self.loss(X,y) # evaluate f(x - h)
      self.W[ix] = oldval # reset
  
      grad_numerical = (fxph - fxmh) / (2 * h)
      grad_analytic = your_grad[ix]
      rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
      print('numerical: %f analytic: %f, relative error: %e' % (grad_numerical, grad_analytic, rel_error))

  def fast_loss_and_grad(self, X, y):
    """
    A vectorized implementation of loss_and_grad. It shares the same
	inputs and ouptuts as loss_and_grad.
    """
    loss = 0.0
    grad = np.zeros(self.W.shape) # initialize the gradient as zero
  
    # ================================================================ #
    # YOUR CODE HERE:
	#   Calculate the softmax loss and gradient WITHOUT any for loops.
    # ================================================================ #
    
    num_train = X.shape[0]
    scores = np.dot(self.W, X.T)
    scores -= np.max(scores, axis=0, keepdims=True)
    soft_max = np.exp(scores) / np.sum(np.exp(scores), axis=0, keepdims=True)
    true_y = soft_max[y, range(num_train)]
    loss = np.sum(-np.log(true_y.clip(np.finfo(float).eps,)))
    loss /= num_train

    soft_max[y, range(num_train)] -= 1
    grad = np.dot(soft_max, X)
    grad /= num_train
    
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return loss, grad

  def train(self, X, y, learning_rate=1e-3, num_iters=100,
            batch_size=200, verbose=False):
    num_train, dim = X.shape
    num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes

    self.init_weights(dims=[np.max(y) + 1, X.shape[1]])	# initializes the weights of self.W

    # Run stochastic gradient descent to optimize W
    loss_history = []

    for it in np.arange(num_iters):
      X_batch = None
      y_batch = None

      # ================================================================ #
      # YOUR CODE HERE:
      #   Sample batch_size elements from the training data for use in 
      #	  gradient descent.  After sampling,
      #     - X_batch should have shape: (dim, batch_size)
	  #     - y_batch should have shape: (batch_size,)
	  #   The indices should be randomly generated to reduce correlations
	  #   in the dataset.  Use np.random.choice.  It's okay to sample with
	  #   replacement.
      # ================================================================ #
      indices = np.random.choice(X.shape[0], batch_size)
      X_batch = X[indices]
      y_batch = y[indices]
      # ================================================================ #
      # END YOUR CODE HERE
      # ================================================================ #

      # evaluate loss and gradient
      loss, grad = self.fast_loss_and_grad(X_batch, y_batch)
      loss_history.append(loss)

      # ================================================================ #
      # YOUR CODE HERE:
      #   Update the parameters, self.W, with a gradient step 
      # ================================================================ #
      self.W = self.W - grad* learning_rate
	  # ================================================================ #
      # END YOUR CODE HERE
      # ================================================================ #
      if verbose and it % 100 == 0:
        print('iteration {} / {}: loss {}'.format(it, num_iters, loss))
    return loss_history

  def predict(self, X):
    """
    Inputs:
    - X: N x D array of training data. Each row is a D-dimensional point.

    Returns:
    - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
      array of length N, and each element is an integer giving the predicted
      class.
    """
    y_pred = np.zeros(X.shape[1])
    # ================================================================ #
    # YOUR CODE HERE:
    #   Predict the labels given the training data.
    # ================================================================ #
    y_pred = np.argmax(np.dot(self.W, X.T),axis=0)
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return y_pred


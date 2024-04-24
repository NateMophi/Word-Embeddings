import numpy as np
from utils2 import get_dict

N = 3 #Word Embedding Size
V = 5 #Vocab Size

# Define first matrix of weights
W1 = np.array([[ 0.41687358,  0.08854191, -0.23495225,  0.28320538,  0.41800106],
               [ 0.32735501,  0.22795148, -0.23951958,  0.4117634 , -0.23924344],
               [ 0.26637602, -0.23846886, -0.37770863, -0.11399446,  0.34008124]])

# Define second matrix of weights
W2 = np.array([[-0.22182064, -0.43008631,  0.13310965],
               [ 0.08476603,  0.08123194,  0.1772054 ],
               [ 0.1871551 , -0.06107263, -0.1790735 ],
               [ 0.07055222, -0.02015138,  0.36107434],
               [ 0.33480474, -0.39423389, -0.43959196]])

# Define first vector of biases
b1 = np.array([[ 0.09688219],
               [ 0.29239497],
               [-0.27364426]])

# Define second vector of biases
b2 = np.array([[ 0.0352008 ],
               [-0.36393384],
               [-0.12775555],
               [-0.34802326],
               [-0.07017815]])

print(f"shape & size of W1: {W1.shape, W1.size} (NxV)")
print(f"shape & size of W2: {W2.shape, W2.size} (Nx1)")
print(f"shape & size of b1: {b1.shape, b1.size} (VxN)")
print(f"shape & size of b2: {b2.shape, b2.size} (Vx1)")

words = ["i", "am", "happy", "because", "i", "am", "learning"]
word2Ind, Ind2word = get_dict(words)

def get_windows(words, C):
    i = C
    while i < len(words) - C:
        center_word = words[i]
        context_words = words[(i-C):i] + words[(i+1):(i+C+1)]
        yield context_words, center_word
        i+=1

# Word to 1-Hot Vector func as seen previously
def word_to_one_hot_vector(word, word2Ind, V):
    one_hot_vector = np.zeros(V)
    one_hot_vector[word2Ind[word]]=1
    return one_hot_vector

# Context word to vector func
def context_word_to_vector(context_words, word2Ind, V):
    context_words_vectors = [word_to_one_hot_vector(w, word2Ind, V) for w in context_words]
    context_words_vectors = np.mean(context_words_vectors, axis=0)
    return context_words_vectors

# Genearator func
def get_training(words, C, wordInd, V):
    for context_words, center_word in get_windows(words, C):
        yield context_word_to_vector(context_words, word2Ind, V), word_to_one_hot_vector(center_word, word2Ind, V)

    
training_examples = get_training(words, 2, word2Ind, V)
xArray, yArray = next(training_examples)

# Context words Vec
print(xArray)
# 1-Hot Vec of Center word
print(yArray)

x = xArray.copy()
y = yArray.copy()
x.shape=(V,1)
y.shape=(V,1)
print(x)
print(y)

# Activation Functions again
def ReLU(z):
    res = z.copy()
    res[res < 0]=0
    return res
def softmax(z):
    e = np.exp(z)
    sum_e = np.sum(e)
    return e/sum_e

# Hidden Layer
z1 = np.dot(W1, x)+b1
h = ReLU(z1)
print("\n")
print(z1)
print(h, "\n")

z2 = np.dot(W2, h)+b2
print(z2)
y_hat = softmax(z2)
print(y_hat)

# Cross-Entropy Loss
def crossEntropyLoss(y_pred, y_actual):
    loss = np.sum(y_actual * -np.log(y_pred))
    return loss

print(crossEntropyLoss(y_hat, y))
print("\n\n\n\n\n")
## Backward Propagation
# Compute vector with partial derivatives of loss function with respect to b2
grad_b2 = y_hat - y
grad_W2 = np.dot(y_hat - y, h.T)
grad_b1 = ReLU(np.dot(W2.T, y_hat-y))
grad_W1 = np.dot(ReLU(np.dot(W2.T, y_hat-y)), x.T)
print(f"grad_b2: {grad_b2}")
print(f"grad_W2: {grad_W2}")
print(f"grad_b1: {grad_b1}")
print(f"grad_W1: {grad_W1}")
print("\n")
print(f"shape & size of grad_b2: {grad_b2.shape, grad_b2.size} (NxV)")
print(f"shape & size of grad_W2: {grad_W2.shape, grad_W2.size} (Nx1)")
print(f"shape & size of grad_b1: {grad_b1.shape, grad_b1.size} (VxN)")
print(f"shape & size of grad_W1: {grad_W1.shape, grad_W1.size} (Vx1)")
print("\n")
# Gradient Descent
alpha = 0.03
W1_new = W1 - alpha*grad_W1
print('old value of W1:')
print(W1)
print()
print('new value of W1')
print(W1_new)

W2_new = W2 - alpha*grad_W2
print('old value of W2:')
print(W2)
print()
print('new value of W2:')
print(W2_new)

b1_new = b1 - alpha*grad_b1
print('old value of b1:')
print(b1)
print()
print('new value of b1:')
print(b1_new)

b2_new = b2 - alpha*grad_b2
print('old value of b2:')
print(b2)
print()
print('new value of b2:')
print(b2_new)

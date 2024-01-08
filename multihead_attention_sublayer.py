import numpy as np
from scipy.special import softmax

# inputs of a multihead attention sublayer
print("Step 1: Input: 3 inputs, d_model=4")  # instead of d_model==512
x = np.array([[1.0, 0.0, 1.0, 0.0],  # Input 1
              [0.0, 2.0, 0.0, 2.0],  # Input 2
              [1.0, 1.0, 1.0, 1.0]])  # Input 3
print("x: ", x)

print("Step 2: weights 3 dimensions x d_model=4")
print("w_Query")
w_query = np.array([[1, 0, 1],
                    [1, 0, 0],
                    [0, 0, 1],
                    [0, 1, 1]])
print(w_query)

print("w_Key")
w_key = np.array([[0, 0, 1],
                  [1, 1, 0],
                  [0, 1, 0],
                  [1, 1, 0]])
print(w_key)

print("w_Value")
w_value = np.array([[0, 2, 0],
                    [0, 3, 0],
                    [1, 0, 3],
                    [1, 1, 0]])
print(w_value)

print("Step 3: Matrix multiplication to obtain Q,K,V")
print("Query: x * w_query")
Q = np.matmul(x, w_query)
print(Q)

print("Key: x * w_key")
K = np.matmul(x, w_key)
print(K)

print("Value: x * w_value")
V = np.matmul(x, w_value)
print(V)

print("Step 4: Scaled Attention Scores")
# actually k_d = math.sqrt(K.shape[1])
k_d = 1  # square root of k_d=3 rounded down to 1 for this example
attention_scores = (Q @ K.transpose()) / k_d
print(attention_scores)

print("Step 5: Scaled softmax attention_scores for each vector")
attention_scores[0] = softmax(attention_scores[0])
attention_scores[1] = softmax(attention_scores[1])
attention_scores[2] = softmax(attention_scores[2])
print(attention_scores[0])
print(attention_scores[1])
print(attention_scores[2])

print("Step 6: attention value obtained by score1/k_d * V")
print(V[0])
print(V[1])
print(V[2])
print("Attention 1")
attention1 = attention_scores[0].reshape(-1, 1)
attention1 = attention_scores[0][0] * V[0]
print(attention1)
print("Attention 2")
attention2 = attention_scores[0][1] * V[1]
print(attention2)
print("Attention 3")
attention3 = attention_scores[0][2] * V[2]
print(attention3)

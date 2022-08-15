import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy

########################################
tf.compat.v1.disable_v2_behavior()  # disable tf 2 features
tf.compat.v1.disable_eager_execution()
tf1 = tf.compat.v1  # tf v1 reached functions
a = tf1.placeholder(tf.float32)
b = tf1.placeholder(tf.float32)
add = tf1.add(a, b)
sess = tf1.Session()
binding = {a: 1.5, b: 2.5}
c = sess.run(add, feed_dict=binding)
########################################

seed = 7
numpy.random.seed(seed)

dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input and output variables
X = dataset[:, 0:8]
Y = dataset[:, 8]   # label

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=seed)

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # between o and 1

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=150, batch_size=10)  # CPU and GPU works in here

scores = model.evaluate(X, Y)

if __name__ == '__main__':
    print('\n%s: %.2f%%' % (model.metrics_names[1], scores[1]*100))

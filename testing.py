import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

for test in range(len(x_test)):
    for row in range(28):
        for x in range(28):
            if x_test[test][row][x] != 0:
                x_test[test][row][x] = 1



model = tf.keras.models.load_model('my_model.model')
store = len(x_test[0:5])

l = np.random.randint(-1,len(x_test))
predictions = model.predict(x_test[l:l+5])    #here it has converted into categorical

count = 0
for x in range(len(predictions)):
    guess = np.argmax(predictions[x])
    actual = y_test[l+x]
    print("I guess the number is ",guess)
    print("In Actual the number is ",actual)
    if guess!=actual:
        count = count+1
    plt.imshow(x_test[x+l])
    plt.show()

print("The program got", count, 'wrong, out of', store)
print(str(100 - ((count/store)*100)) + '% correct')

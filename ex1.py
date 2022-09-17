import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras

def save_training_plots(history, model_name):
    #  "Accuracy"
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f'plots/accuracy_{model_name}.png')
    plt.show()
    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f'plots/loss_{model_name}.png')
    plt.show()


def run_ex1_b():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape((60000, 28, 28, 1)) / 255.
    x_test = x_test.reshape((10000, 28, 28, 1)) / 255.
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    model = keras.Sequential()
    model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(rate=0.25))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dropout(rate=0.5))
    model.add(keras.layers.Dense(10, activation="softmax"))
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adadelta(learning_rate=1),
                  metrics='accuracy')

    model.summary()

    history = model.fit(x_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)
    model.save('models/ex1_b.h5')

    model = keras.models.load_model('models/ex1_b.h5')
    save_training_plots(history,"ex1_b_with_droput")

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print("Loss:", loss)
    print("Accuracy:", accuracy)


if __name__ == '__main__':
    # run_training()
    run_ex1_b()

    '''
    Question answers:
    
    Q6. TODO 
    
    Q7. Overfitting happens when the model gets optimized to run as best as possible on the train dataset, but it does 
    not generalize well. This can be easily seen in the loss curves as a training loss that is far lower then the 
    validation loss. On a higher level, we can say that the model learns by heart the correct answers for the training 
    labels, but this makes it useless when new data is presented. This is exactly what happened before adding droput to 
    the model. We can clearly see that although the training loss was decreasing (proof that the model was predicting 
    the training examples better and better), the validation loss mostly stayed the same. This clearly indicates that 
    the model was not trying to recognize digits ut just memorize the training examples.
    
    Dropout is a technique used to prevent overfitiing. It involves randomly ignoring(aka 'dropping out') a number of 
    layer outputs throughout the network. This can be controlled by giving the probability of dropping the output of a 
    node(same for every node). Dropout corrects overfitting in a few ways. Firstly, it reduces the number of node outputs 
    that are used during training, thus decreasing the model's ability to straight up learn the training examples by 
    heart. Secondly, because some nodes are de-activated randomly throughout the training process, this makes the model 
    more robust by forcing it to keep important information in each node, so that if a few nodes disappear, the rest of 
    them still are able to do a good job in predicting. And it turns out that actually making nodes look at only the 
    important features of the last layer is a good way to make them look for features that help in generalizing on 
    new examples.
    
    Q8.
    a. On one of our machines, the first epoch takes 5 seconds, while the others take 3 seconds to complete.
    b. Now, after adding dropout, we almost see the perfect loss curve. Both the training and the validation loss are 
    decreasing and they seem to get closer and closer. This indicates that the model focuses on the right features and 
    is trying to generalize on the detection task it has to do. Given that the validation and training loss do not 
    actually meet and plateau their decrease around the same value, I would recommend to train for a 1-2 more epochs in 
    order to get a better model, but this is not the point of the exercise.
    c. If we train for 8 epochs and let the model reach its full potential, then we see an improvement of around 1% on 
    the test dataset (it varies because the starting weights are not set, but chosen random as it is the keras default).
    Also we see far less of an overfitting tendency that before using dropout. Both of these are proofs that the model 
     actually generalizes better.
    '''

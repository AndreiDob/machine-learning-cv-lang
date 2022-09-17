import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers


def run_training():
    # Model / data parameters
    num_classes = 10
    input_shape = (28, 28, 1)

    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.summary()

    batch_size = 128
    epochs = 15

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])


def show_training_plots(history):
    #  "Accuracy"
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
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

    history = model.fit(x_train, y_train, batch_size=128, epochs=8, verbose=1, validation_split=0.2)
    show_training_plots(history)

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
    a. When using dropout, the training time is decreased. This makes sense as although in the feed-forward stage the 
    computations are the same, in the backpropagation stage, the weight updates for the nodes whose output was ignored 
    are no longer computed and therefore the time is reduced.
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

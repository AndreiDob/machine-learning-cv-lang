# Execise 1

### Question 1
*Give a list of applications where automatic recognition of hand-written numbers would be useful.*

Automatic recognition of hand-written numbers can be useful in many applications. For example, in the banking sector, it can be used to read parts of the Dutch IBAN numbers, reference codes, or other digits that are presented on handwritten forms. In the postal sector, it can be used to read the postal code and house number on an envelope. In the education sector, it can be used to read the student numbers on test papers

### Question 2
*Describe how the accuracy on the training and validation sets progress differently across epochs, and what this tells us about the generalisation of the model.*

The accuracy on the training set increases with each epoch, while the accuracy on the validation set's consistent increase in accuracy stops after the 4th epoch. This tells us that the model might be overfitting the training set and that it might not be generalising well. This means the model might be recognising properties unique to an image in the training data rather than the digit it should be trying to detect. 

Plots:
![Loss Plot] (plots/q2-p1.png)
![Accuracy Plot] (plots/q2-p2.png)

### Question 3
*Explain whether you think this accuracy is sufficient for some uses of automatic hand-written digit classification, and why.*

In the assignment, we found an accuracy of 92,46% on the test set. For considering just one number this accuracy is sufficient in some use cases, but if we take one of the examples for *question one* we have use cases where multiple numbers need to be recognised for the use case (e.g. the IBAN number, or postal code). Assuming that the test-set and its accuracy is a good representation of the numbers in the actual use case the odds of it having all four numbers correctly is 0,9246^4 = 0,7308. This would result in too high of an error rate for the postal sector. For the banking sector, this error rate is even higher than an IBAN has 10 digits, so the probability of labelling all numbers correctly is 0,9246 ^ 10 = 0,4566. This results in too high of an error rate.

### Question 4
*Explain how linear activation of units limits the possible computations this model can perform*

Linear activation only allows for the multiplication of the input signals with weights from a node, and summing these multiplications of all the inputs (w1 * x1 + w2 * x2 + ...). This means another (mathematical) function on the input of a node is not possible. One interesting property is that the first derivative of these functions is always a constant. This entails that the changing of an input by a certain amount always results in the same increment on the output. By only using linear activation, the activation function of a node can not be a sigmoid function or a tanh function. This limits the possible computations this model can perform.

### Question 5
*Describe how this training history differs from the previous model, for the training and validation sets. Describe what this tells us about the generalisation of the model.*

Just like in the previous model the accuracy keeps increasing on the training set. However, in this model, the accuracy only stops increasing after the 8th set on the validation set. This tells us that the model is not overfitting the training set as much and as fast as the previous model. This means that the model is generalising better.

![Loss Plot] (plots/q5-p1.png)
![Accuracy Plot] (plots/q5-p2.png)

### Question 6
*Explain whether you think this is sufficient for the uses of automatic hand-written digit classification you listed in Question 1, and why.*

An accuracy of 98.96% was found for this model. For digit recognition such as house numbers, this is most likely sufficient. However, if we repeat the calculations from question three we find that the expected accuracy of correctly reading a zip code is 0,9896^4 = 0,9590 which is significantly better than the previous model. This would still mean 5% of postal codes are read incorrectly, which is assumably still not enough for a production-scale application. For IBANs with 10 digits, the expected probability would be 09896^10 = 0,9007. This is again better than the previous model, but would still result in a high error rate for a production-scale application. Human validation would still be required for almost all use cases where multiple numbers need to be read.

### Question 7

Overfitting happens when the model gets optimized to run as best as possible on the train dataset, but it does 
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

### Question 8
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





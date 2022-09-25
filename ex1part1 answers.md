# Execise one

### Question one
*Give a list of applications where automatic recognition of hand-written numbers would be useful.*

Automatic recognition of hand-written numbers can be useful in many applications. For example, in the banking sector, it can be used to read parts of the Dutch IBAN numbers, reference codes, or other digits that are presented on handwritten forms. In the postal sector, it can be used to read the postal code and house number on an envelope. In the education sector, it can be used to read the student numbers on test papers

### Question two
*Describe how the accuracy on the training and validation sets progress differently across epochs, and what this tells us about the generalisation of the model.*

The accuracy on the training set increases with each epoch, while the accuracy on the validation set's consistent increase in accuracy stops after the 4th epoch. This tells us that the model might be overfitting the training set and that it might not be generalising well. This means the model might be recognising properties unique to an image in the training data rather than the digit it should be trying to detect. 

Plots:
![Loss Plot] (plots/q2-p1.png)
![Accuracy Plot] (plots/q2-p2.png)

### Question three
*Explain whether you think this accuracy is sufficient for some uses of automatic hand-written digit classification, and why.*

In the assignment, we found an accuracy of 92,46% on the test set. For considering just one number this accuracy is sufficient in some use cases, but if we take one of the examples for *question one* we have use cases where multiple numbers need to be recognised for the use case (e.g. the IBAN number, or postal code). Assuming that the test-set and its accuracy is a good representation of the numbers in the actual use case the odds of it having all four numbers correctly is 0,9246^4 = 0,7308. This would result in too high of an error rate for the postal sector. For the banking sector, this error rate is even higher than an IBAN has 10 digits, so the probability of labelling all numbers correctly is 0,9246 ^ 10 = 0,4566. This results in too high of an error rate.

### Question four
*Explain how linear activation of units limits the possible computations this model can perform*

Linear activation only allows for the multiplication of the input signals with weights from a node, and summing these multiplications of all the inputs (w1 * x1 + w2 * x2 + ...). This means another (mathematical) function on the input of a node is not possible. One interesting property is that the first derivative of these functions is always a constant. This entails that the changing of an input by a certain amount always results in the same increment on the output. By only using linear activation, the activation function of a node can not be a sigmoid function or a tanh function. This limits the possible computations this model can perform.

### Question five
*Describe how this training history differs from the previous model, for the training and validation sets. Describe what this tells us about the generalisation of the model.*

Just like in the previous model the accuracy keeps increasing on the training set. However, in this model, the accuracy only stops increasing after the 8th set on the validation set. This tells us that the model is not overfitting the training set as much and as fast as the previous model. This means that the model is generalising better.

![Loss Plot] (plots/q5-p1.png)
![Accuracy Plot] (plots/q5-p2.png)

### Question six
*Explain whether you think this is sufficient for the uses of automatic hand-written digit classification you listed in Question 1, and why.*

An accuracy of 98.96% was found for this model. For digit recognition such as house numbers, this is most likely sufficient. However, if we repeat the calculations from question three we find that the expected accuracy of correctly reading a zip code is 0,9896^4 = 0,9590 which is significantly better than the previous model. This would still mean 5% of postal codes are read incorrectly, which is assumably still not enough for a production-scale application. For IBANs with 10 digits, the expected probability would be 09896^10 = 0,9007. This is again better than the previous model, but would still result in a high error rate for a production-scale application. Human validation would still be required for almost all use cases where multiple numbers need to be read.


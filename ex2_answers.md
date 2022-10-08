### Question nine

a. On one of our machine which has a GPU with 4Gb of memory the epoch run time was around 7.5 seconds. However, as 
    in the case of the first exercise, the first epoch took longer at 10-12s.
    
b. The factor that causes the first epoch to take longer is that Keras supports lazy execution. The model creation 
    and compilation code are not executed until it is absolutely required which is right before the first training 
    epoch. That increased time for the first epoch includes building the TensorFlow computational graph. The other 
    epochs reuse this graph, so the overhead time is not present anymore in them.
    
c. One main difference we can observe is that the decrease in training loss is more gradual. This is explained by 
    the lower learning rate that was chosen. Another thing is that the validation loss is initially high, but it 
    steadily decreases more than it did in the exercise 1 trainings. One explanation would be that the task is harder 
    so it is harder to learn how to generalize well, therefore taking more epochs to achieve this. We also see that 
    sometimes the validation loss increases for one epoch. This can be attributed to a high learning rate, thus instead 
    of going towards the optima point, the weights make the loss jump on the other side of the "loss valley". The reason 
    it actually comes down is that the optimizer used has adaptive learning rate and detects the jump and decreases the 
    learning rate accordingly
    
d. Because this is a more complex task, in which the overall image is extremely important for the detection(not only
    parts of it), the deeper networks perform way better. This is because they have a bigger receptive field, thus they 
    look more at the overall image rather than search for specific close-by features. Another impacting factor is 
    that the exercise2 network has more convolutional layers, meaning that it can extract more features and 
    combinations of features before actually figuring out how to combine them to predict classes(in the fully connected 
    layers).

### Question ten
In this paper the researchers describe their experiments in finding good CNN architectures that also mimic biological neurology.
They have begun by artificially creating high-variation image datasets for object recognition and classification. Then, by using multiple electrode arrays, they measured the IT(Inferior Temporal cortex) neurons' responses to those images from humans. Further, they evaluated thousands of CNNs using high-throughput computational methods. They meausure for these CNNs the categorization performance and how well the outputs of the networks matches up with the outputs of the human-measured IT neurons.
As a result, they have found out that the best performing models on the categorization task were also the ones who predicted the output of the IT neurons the best. However, the models that were selected only on the IT predictibity, did not necessarily have good task performance. This result implies that the performance and IT prediction corellation cannot be explained only by simple mechanical considerations. 
 Further, they have also showed that doing well on categorization tasks does not necessarily mean that the IT prediction will be good. To do this, they have trained models by giving them access to the categorization results during inference and those models actually performed worse at predicting the IT neural responses than the ones that were straight-up learning the categorization task from scratch. 
After finding which model architecture perform better at the categorization task, they analysed what features these models share. It turns out that filter mean and spread, together with the ratio of max-pooling to average-pooling are the model parameters that were the most sensitive and diverse. This hints to the heterogenities observed in higher ventral cortex areas, but it is not enough of a proof for this, so future research still hase to be conducted into this topic. During their experimentation the researchers have also found that the penultimate layer of the best categorization model accurately predicts the input of the IT, the V4 neural structure. This gives a strong evidence that V4 actually is an intermediate layer that contributes in the processing of data until it reaches IT. This result puts a whole new perspective on the way we analyse brains.
The traditional way of analysing brains is from the beginning of the processing flow(eg. retina) to the deeper levels. It assumes that you need to know the first levels to understand the others. This paper, in light of its experimental results, argues that the reverse is also true. It proposes that we should also focus on understanding the deeper levels, because the lower level brain structures have to be made in such a way that they support the computation in the deeper levels. Therefore, if we would also focus on the deeper layers, it might give as valuable insights into how the lower layers from the cognitive path work.



# Classification-Loss-Comparison
Densenet121 model is used for comparison of Triple loss, Center loss and Cross-entropy loss models. </br>
**Cross-Entropy Loss** Function, also called logarithmic loss, log loss or logistic loss. Each predicted class probability is compared to the actual class desired output 0 or 1 and a score/loss is calculated that penalizes the probability based on how far it is from the actual expected value. </br>
**Triplet loss** is a loss function for machine learning algorithms where a reference input (called anchor) is compared to a matching input (called positive) and a non-matching input (called negative). It reduces the embedding vector distance between anchor and positive class and increase the distance between anchor and negative class. </br>
**Center loss** reduces the distance of each data point to its class center. It is not as difficult to train as triplet loss and performance is not based on the selection process of the training data points(triplets). Combining it with a softmax loss, prevents embeddings from collapsing. </br></br>

### Model: Densenet121
Took Densenet 121 model pretrained on imagenet. 

### Dataset: Tiny Imagenet
Tiny imagenet is a dataset of 200 classes. Training set contains 500 image in each class, a total of 1 lakh images.

## Training and Evaluation

1. The training and saving files for crossentropy loss model and center loss model can be run directly.
2. Before running the triplet loss model file, run preprocessing.ipynb file.
3. The file will help create a dataset.pkl file which contains images and their labels in proper sequence. After this, we can safely run triplet loss model file.
4. After saving all the models, we can run the ‘Evaluation of saved models’ ipynb file.
5. Note that catboost classifier is used as final classifier for triplet loss model, it needs to be installed before use. (!pip install catboost)

## Results

![image](https://user-images.githubusercontent.com/65457437/156496591-d862bf40-241a-4449-8f51-846f1e612851.png)

1. Note that Triplet loss (margin = 25.0) model is trained for 70 epochs and fed with catboost classifier (a state of the art classifier).
2. Center loss model is trained for 35 epochs and Crossentropy loss model is trained for 25 epochs only. 
3. I have also observed that accuracy of triplet loss model trained for 25 epochs is very low (close to 6%). So I trained it for more epochs.

## WHICH ONE IS BETTER AND WHY?

1. We can clearly see the supremacy of Cross entropy loss model over other models. It produced very good results even after getting trained for low number of epochs.
2. From the update training log of triplet loss, we can see that increasing the number of epochs would have allowed it to train better.
3. I would prefer to have cross entropy loss model as it gives best results even after getting trained for low number of epochs. Time taken by this model per epoch is also very low compared to other models. ( Reason: In triplet loss model we compute model(image) for all the triplet, anchor, positive and negative for every iteration, hence time taken by it is highest. In center loss model, the center loss function also has trainable parameters. So updating it with back-propagation takes time).
3. Second loss preference would depend on availability of time and computational resources. If we have sufficient time and resources, we can go for triplet loss (with 200-250 epochs). Else, center-loss would produce better results (for <50 epochs).


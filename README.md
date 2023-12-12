# Machine Learning Intern Assignment - Body Scratch
#### 09/10/2023
#### Bianca Chiusano 

## Description of the Task : Pose Classification
- The **purpose** of this assignment is to classify the different poses based on the 17 key points of the body.
- **Input**: the list of numpy arrays which stored heatmap which is the output of ViTPose. The shape of each numpy file is [1, 17, 64, 48] which corresponds to 17 key points in the body. All numpy files in the same folder belong to the same class. There are a total of 17 classes (action_down, action_inside, action_new, ….). 
- **Task**: Apply classification algorithm to classify the pose of the input heatmap

## Approach
### CNN
- I decided to go for a Convolutional Neural Network because it seemed to be the best NN fo image like data.
- The approach of a CNN: Convolution + ReLU --> Pooling --> Convolution + ReLU --> Pooling -->...--> Flatten --> Fully connected --> softmax 
![alt text](https://miro.medium.com/v2/resize:fit:1358/1*XbuW8WuRrAY5pC4t-9DZAQ.jpeg)
- I later understood that the heatmaps provided should not have been treated as images. So instead from what I understood Convulation operation and Max pooling are not internded for pose classification, as the pose keypoints from ViTPose are not image features.

- Important Note: This was my first time approaching such task. First time working with numpy data, cnns and tensorflow. It took some time and much research but I believe what I have is a start. It was nice to learn so much.

### Files (Explanation)
- **first_trial.ipynb**: is the first notebook I started to create a CNN from scratch.. I spent a couple of days on this, researching on and off to really understand what a CNN is and how the layers work. Eventually I realised that a faster and more efficient way to create the model would be using frameworks such as Tensorflow and Pytorch. I kept this notebook to show you my initial approach and how it changed. 
- **cnn_notebook.ipynb**: this is the final notebook, the main one.
    - I got all the heatmaps from each folder to create a big numpy array of (40553, 17,64,48) and creates a list of class names for the labels for classification
    - Split the data into train and test and then made it so that the shape was good for a tensorflow model.
    - I created 3 different models: Model_images that is actually a CNN for image classification which I then realised was not the purpose of this task. Model_1 is for pose classification, and Model_2 has one more Dense layer
    - For each model I train it on the train data and evaluate it on the test data
    - **Imports**: Sklearn, Numpy, Pandas, Tensorflow, Keras
### Models:
- Model_images: Conv operation, Maxpooling, and dense layers - doesn't really work for this task
- Model_1: Flattens the data, then has just one dense layer to create a fully connected layer with 64 neurons and the ReLU activation function to introduce Non-linearity
    - Accuracy for training: 0.99 
- Model_2: Has one more dense layer that Model_1
    - Accuracy for training: 0.99

## Future
### Limitations
- The model seems to overfit a lot during training with an accuracy of 0.99 and sometimes 1.0 but then an accuracy in testing of only 0.03
### Experiments
- I didn't manage to conduct any experiments yet because of the low testing accuracy. However If I had to run some I would experiment with the amount of layers in the CNN, the number of neurons in the fully connected layers and the number of epochs and batch size. Then choose the most optimal hyperparameters.
### Deployment
- Tensorflow hub or Hugging Face
- https://www.tensorflow.org/hub


## Sources
#### Documentation
- Sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
- Tensor: https://www.tensorflow.org/api_docs/python/tf/Tensor
- Sequential: https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
- Conv2D: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D#args
- Model and Layers: https://www.tensorflow.org/js/guide/models_and_layers
- Tensorflow CNN: https://www.tensorflow.org/tutorials/images/cnn
- CIFAR10: https://keras.io/api/datasets/cifar10/

#### Videos
- Simple explanation of CNN: https://www.youtube.com/watch?v=zfiSAzpy9NM&t=358s
- Image classification using CNN: https://www.youtube.com/watch?v=7HPwo4wnJeA
- Tensorflow CNN: https://www.youtube.com/watch?v=eMMZpas-zX0&t=513s
- CNN from scratch: https://www.youtube.com/watch?v=Lakz2MoHy6o&t=1513s


#### Extra
- https://github.com/ViTAE-Transformer/ViTPose
- ViTPose: https://arxiv.org/pdf/2204.12484.pdf
- ChatGPT, Perplexity
- Hugging Face: https://huggingface.co/spaces/hysts/ViTPose_video

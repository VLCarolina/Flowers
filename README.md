                                                                        # FLOWERS
                                                                     #ALEXNET PROJECT
                                                                 
![Alt text]([image_url](https://www.imagewoof.com/wp-content/uploads/2020/09/summer-flowers-1590678054.jpg))

#OBJECTIVE

The objective of an AlexNet Flowers project is to create a deep learning model that can accurately classify images of flowers into their respective species. The model should be trained using the AlexNet architecture, which is a convolutional neural network (CNN) that is capable of learning and extracting meaningful features from images.

The specific objectives of the project could include:

Collecting a large dataset of images of flowers (30 different types), with each image labeled with its corresponding species.
Preprocessing the images to ensure they are all the same size and format, and splitting the dataset into training and validation sets.
Implementing the AlexNet architecture using a deep learning framework such as TensorFlow or PyTorch.
Training the model on the flower dataset, tuning hyperparameters as needed to achieve high accuracy.
Evaluating the model's performance on the validation set to ensure it is not overfitting.
Testing the model on a separate test set to determine its real-world performance.
Visualizing the results of the model's predictions, such as generating a confusion matrix or creating a web application to classify user-provided images of flowers.

#ALEXNET ARCHITECTURE
AlexNet is a convolutional neural network architecture designed for image classification. It was developed by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton in 2012 and achieved significant success in the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) that year.

The AlexNet architecture consists of eight layers in total, including five convolutional layers and three fully connected layers. The first five layers are convolutional layers with various filter sizes, strides, and padding. These layers are responsible for extracting features from the input image. The first two convolutional layers have 96 filters each, while the next two have 256 filters each. The fifth convolutional layer has 384 filters.

The three fully connected layers at the end of the network are responsible for classification. The first fully connected layer has 4096 neurons, followed by a dropout layer to prevent overfitting. The second fully connected layer also has 4096 neurons, followed by another dropout layer. The final fully connected layer has as many neurons as there are classes in the classification problem.

AlexNet also incorporates several other techniques to improve performance, including overlapping pooling, local response normalization, and data augmentation.

Overall, the architecture of AlexNet is characterized by its large number of filters and neurons, its use of multiple GPUs for training, and its incorporation of techniques to improve generalization and prevent overfitting.

#FLOWERS' PROJECT
- This project contains 30 different types of flowers.
- I collected 5 pictures of each flower.
- Each type of flower has one folder inside the main folders Train and Valid, when the pictures of each type of flower are divided: 4 of 5 pictures are in the Train folder and 1 is in the Valid folder. 
- The folders are uploaded to the google drive account.
- Creating the Alexnet project in Github (steps):
    - Linking the Google Drive folder with the Flowers Notebook    
    - Loading labeled images from folders
    - Create loaders
    - Set the Alexnet structure
    - Ask the coding for names of categories
    - Define inputs and outputs
    - Doing the torch
    - Train Alexnet model and validation data
    - Prediction 

Doing all this process we create data that contain information about 30 different types of flowers and when we put information about one new picture the system the Alexnet model runs and star to find similarities pixel by pixel between the new picture and the data created before. No matter the size because we standardized the size and no matter the position.
Alexnet does a simulation and works like the human eyes doing a picture's scan pixel by pixel to find similarities or recognize images and associate with images that one person knows or saw before. As human eyes, no matter the picture's position our eyes and Alexnet net do a process to recognize the picture wherever its position is.


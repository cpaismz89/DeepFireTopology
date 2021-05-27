# Deep Learning
import tensorflow.keras

# Visualization
import matplotlib.pyplot as plt

# Plotting training class
from IPython.display import clear_output

# Image Processing
from imutils import paths, build_montages
import imutils
import cv2
from skimage import io

# Numerical
import numpy as np
import random

# Training Plot class
class TrainingPlot(tensorflow.keras.callbacks.Callback):
    # Constructor
    def __init__(self, show_interval=10):
        self._show_interval = show_interval
    
    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []
    
    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        if epoch % self._show_interval == 0:
            # Append the logs, losses and accuracies to the lists
            self.logs.append(logs)
            self.losses.append(logs.get('loss'))
            self.acc.append(logs.get('acc'))
            self.val_losses.append(logs.get('val_loss'))
            self.val_acc.append(logs.get('val_acc'))

            # Before plotting ensure at least 2 epochs have passed
            if len(self.losses) > 1:

                # Clear the previous plot
                clear_output(wait=True)
                N = np.arange(1, len(self.losses) + 1) * self._show_interval

                # You can chose the style of your preference
                # print(plt.style.available) to see the available options
                plt.style.use("seaborn")

                # Plot train loss, train acc, val loss and val acc against epochs passed
                plt.figure()
                plt.plot(N, self.losses, label = "train_loss")
                plt.plot(N, self.acc, label = "train_acc")
                plt.plot(N, self.val_losses, label = "val_loss")
                plt.plot(N, self.val_acc, label = "val_acc")
                plt.title("Training Loss and Accuracy [Epoch {}]".format(epoch + self._show_interval))
                plt.xlabel("Epoch #")
                plt.ylabel("Loss/Accuracy")
                plt.legend()
                plt.show()

# Plot the training loss and accuracy
def plot_summary(H,
                 N,
                 xlim=None,
                 ylim=None,
                 size=None,
                 ax=None,
                 save=False,
                 plotname=None,
                 ):

    # Initialize if no ax is provided
    if ax is None:
        # Style
        plt.style.use("ggplot")

        # Initialize figure
        plt.figure(figsize=size)

    # Plot curves
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    
    # Title
    plt.title("Training Loss and Accuracy on Dataset")
    
    # Labels
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    
    # Axes Limits
    if xlim is not None:
        plt.xlim([xlim[0], xlim[1]])

    if ylim is not None:
        plt.ylim([ylim[0], ylim[1]])
    
    # Legend and save
    plt.legend(loc="lower left")
    if save:
        plt.savefig("plot.png" if plotname is None else plotname)        
        
# Plot LR summary
def plot_lr_summary(clr,
                    xlim=None,
                    ylim=None,
                    size=None,
                    ax=None,
                    plotname=None,
                    ):

    # Initialize if no ax is provided
    if ax is None:
        # Style
        plt.style.use("ggplot")

        # Initialize figure
        plt.figure(figsize=size)

    # Plot curve
    plt.plot(np.arange(0, len(clr.history['lr'])), clr.history["lr"], label="learning_rate")
    
    # Title
    plt.title("Cyclical Learning Rate (CLR)")
    
    # Labels
    plt.xlabel("Training Iterations")
    plt.ylabel("Learning Rate")

    # Axes Limits
    if xlim is not None:
        plt.xlim([xlim[0], xlim[1]])

    if ylim is not None:
        plt.ylim([ylim[0], ylim[1]])
    
    # Legend and save
    plt.legend(loc="lower left")
    plt.savefig("clr_plot.png" if plotname is None else plotname)    

# Montage data custom function
def montage_data(sample=9, 
                 dataset=None, 
                 save=False, 
                 text='Sample Training Set', 
                 textCoord=(0,0), 
                 textColor=None, 
                 outFileName=None,
                 imgShape=(128,128),
                 montageShape=(3,3),
                 fontScale=1,
                 seed=42,
                 show=True):
    
    # Read training dataset and build a montage
    imagePaths = sorted(list(paths.list_images(dataset)))
    imagePaths = imagePaths[:sample]

    # Random shuffle
    random.shuffle(imagePaths)

    # initialize the list of images
    images = []

    # loop over the list of image paths and load them into memory
    for imagePath in imagePaths:
        # load the image and update the list of images
        image = cv2.imread(imagePath)
        images.append(image)

    # Build montage
    montages = imutils.build_montages(image_list=images, 
                                      montage_shape=montageShape, 
                                      image_shape=imgShape)

    # Display montage
    for montage in montages:
        # font 
        font = cv2.FONT_HERSHEY_SIMPLEX 

        # org 
        org = textCoord

        # fontScale 
        fontScale = fontScale

        # Blue color in BGR 
        color = (0, 0, 255) if textColor is None else textColor

        # Line thickness of 2 px 
        thickness = 2

        # Using cv2.putText() method 
        montage = cv2.putText(montage, text, org, font,  
                              fontScale, color, thickness, cv2.LINE_AA) 
        if save:
            cv2.imwrite(outFileName, montage)
        if show:
            cv2.imshow("Montage", montage)
            cv2.waitKey(0)
        
    return montage
   
# Show image from url
def show_img_url(url, 
                 size=(15,10),
                 show=False):
    # download the image using scikit-image
    image = io.imread(url)
    plt.rcParams['figure.figsize'] = size[0], size[1]
    if show:
        plt.imshow(image)
        
    return image

# Show collage of predicted
def predicted_collage(images, 
                      labels, 
                      model, 
                      seed=42, 
                      sampleSize=12,
                      imagesize=(128,128),
                      collagesize=(4,3),
                      show=False,
                      readImage=False):
    # Seed for random sampling
    random.seed(seed)
    idx = np.random.randint(0, len(images), 1)[0]

    # If paths, read images and update images with objects
    if readImage:
        aux = []
        for image in images[idx:idx + sampleSize]:
            image = cv2.imread(image)
            image = cv2.resize(image, imagesize)
            aux.append(image)
        images = aux
        
    # Sample
    Sampledimages = images[idx:idx + sampleSize]
    Sampledlabels = labels[idx:idx + sampleSize]

    # Predictions
    predictions = model.predict(Sampledimages).argmax(1)
    PredictedImages = []

    # Read sample images
    for idx,image in enumerate(Sampledimages):
        # Text
        text = 'Correct' if predictions[idx] == Sampledlabels.argmax(1)[idx] else 'Incorrect'

        # font 
        font = cv2.FONT_HERSHEY_SIMPLEX 

        # org 
        org = (0,50)

        # fontScale 
        fontScale = 1

        # Blue color in BGR 
        color = (0, 0, 255) if text == 'Incorrect' else (0, 255, 0)

        # Line thickness of 2 px 
        thickness = 2

        # Using cv2.putText() method 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, imagesize)
        image = cv2.putText(image, text, org, 
                            font, fontScale, color, 
                            thickness, cv2.LINE_AA) 
        PredictedImages.append(image)
        
        # Show
        if show:
            cv2.imshow('image',image,)
            cv2.waitKey(0)

    # Make collage
    collage = build_montages(image_list=PredictedImages, 
                             image_shape=(128,128), 
                             montage_shape=collagesize)

    # Show
    plt.imshow(cv2.cvtColor(collage[0], cv2.COLOR_BGR2RGB))
    
    # Return collage
    return collage

# Show collage
def collage(images, 
            labels, 
            seed=42, 
            sampleSize=12,
            imagesize=(128,128),
            collagesize=(4,3),
            show=False,
            readImage=False):
    
    # Seed for random sampling
    random.seed(seed)
    idx = np.random.randint(0, len(images), 1)[0]

    # If paths, read images and update images with objects
    if readImage:
        aux = []
        for image in images[idx:idx + sampleSize]:
            image = cv2.imread(image)
            image = cv2.resize(image, imagesize)
            aux.append(image)
        images = aux
        
    # Sample
    Sampledimages = images[idx:idx + sampleSize]
    Sampledlabels = labels[idx:idx + sampleSize]

    # Container
    ProcessedImages = []
    
    # Read sample images
    for idx,image in enumerate(Sampledimages):
        # Text
        text = labels[idx]

        # font 
        font = cv2.FONT_HERSHEY_SIMPLEX 

        # org 
        org = (0,50)

        # fontScale 
        fontScale = .8

        # Label color in BGR 
        color = (0, 255, 0)

        # Line thickness of 2 px 
        thickness = 2

        # Using cv2.putText() method 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, imagesize)
        image = cv2.putText(image, text, org, 
                            font, fontScale, color, 
                            thickness, cv2.LINE_AA) 
        ProcessedImages.append(image)
        
        # Show
        if show:
            cv2.imshow('image',image,)
            cv2.waitKey(0)

    # Make collage
    collage = build_montages(image_list=ProcessedImages, 
                             image_shape=(128,128), 
                             montage_shape=collagesize)

    # Show
    plt.imshow(cv2.cvtColor(collage[0], cv2.COLOR_BGR2RGB))
    
    # Return collage
    return collage
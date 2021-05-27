# Importations
from keras.applications import ResNet50, VGG16, imagenet_utils
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import to_categorical
from imutils import paths
import numpy as np
import pickle
import random
import os


# Extract features for each set VGG
def ExtractFeatures_VGG(TRAIN='training', 
                        TEST='evaluation',
                        VAL='validation',
                        BATCH_SIZE=32,
                        BASE_PATH=os.path.join('TransferLearningKeras', 'dataset'),
                        BASE_CSV_PATH=os.path.join('TransferLearningKeras', 'output'),
                        LE_PATH=os.path.join('TransferLearningKeras', 'output'),
                       ):
    
    # load the VGG16 network and initialize the label encoder
    print("[INFO] loading network...")
    model = VGG16(weights="imagenet", include_top=False)    # No top layer
    le = None

    # Loop over the different sets
    for split in (TRAIN, TEST, VAL):
        # Get all images (paths)
        print("[INFO] processing '{} split'...".format(split))
        p = os.path.sep.join([BASE_PATH, split])
        imagePaths = list(paths.list_images(p))

        # Shuffle them and extract the labels from the paths
        random.shuffle(imagePaths)
        labels = [p.split(os.path.sep)[-2] for p in imagePaths]

        # If the label encoder is None, create it
        if le is None:
            le = LabelEncoder()
            le.fit(labels)

        # Open the output CSV file for writing
        csvPath = os.path.sep.join([BASE_CSV_PATH, "{}.csv".format(split)])
        csv = open(csvPath, "w")

        # loop over the images in batches
        for (b, i) in enumerate(range(0, len(imagePaths), BATCH_SIZE)):
            # extract the batch of images and labels, then initialize the
            print("[INFO] processing batch {}/{}".format(b + 1,
                                                         int(np.ceil(len(imagePaths) / float(BATCH_SIZE))))
                 )
            
            # Get a batch of images *paths* to extract features
            batchPaths = imagePaths[i:i + BATCH_SIZE]
            batchLabels = le.transform(labels[i:i + BATCH_SIZE])
            batchImages = []

            # loop over the images and labels in the current batch
            for imagePath in batchPaths:
                # Load image using the Keras helper utility
                # Resize to 224x224 pixels (to match VGG16 architecture)
                image = load_img(imagePath, target_size=(224, 224))
                image = img_to_array(image)

                # Preprocess: (1) expand the dimensions and (2) substract the mean RGB intensity
                image = np.expand_dims(image, axis=0)
                image = imagenet_utils.preprocess_input(image)

                # Add the image *array* to the batch
                batchImages.append(image)

            # Pass the images through the network and use the outputs as
            # our actual features, then reshape the features into a
            # flattened volume = vector
            batchImages = np.vstack(batchImages)                             # Stacked to match Keras format
            features = model.predict(batchImages, batch_size=BATCH_SIZE)     # Get features for BS images
            features = features.reshape((features.shape[0], 7 * 7 * 512))    # Reshape to create a flattened vector BS, FEAT

            # Loop over the class labels and extracted features
            for (label, vec) in zip(batchLabels, features):
                # construct a row that exists of the class label and extracted features
                vec = ",".join([str(v) for v in vec])       # Join features by comma ,
                csv.write("{},{}\n".format(label, vec))     # Write label and vector of features to csv separate by ,

        # close the CSV file
        csv.close()

    # Serialize the label encoder to disk
    f = open(LE_PATH, "wb")
    f.write(pickle.dumps(le))
    f.close()
    
# Extract features for specific Batch
def ExtractFeatures_batch_VGG(image_batch_paths):
    # load the VGG16 network and initialize the label encoder
    print("[INFO] loading VGG16 network...")
    model = VGG16(weights="imagenet", include_top=False)    # No top layer

    # Containers 
    batchImages = []
    
    # loop over the images and labels in the current batch
    for imagePath in image_batch_paths:
        # Load image using the Keras helper utility
        # Resize to 224x224 pixels (to match VGG16 architecture)
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)

        # Preprocess: (1) expand the dimensions and (2) substract the mean RGB intensity
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)

        # Add the image *array* to the batch
        batchImages.append(image)

    # Pass the images through the network and use the outputs as
    # our actual features, then reshape the features into a
    # flattened volume = vector
    batchImages = np.vstack(batchImages)                                   # Stacked to match Keras format
    features = model.predict(batchImages, batch_size=len(batchImages))     # Get features for BS images
    features = features.reshape((features.shape[0], 7 * 7 * 512))          # Reshape to create a flattened vector BS, FEAT
    
    # Return features
    return features

# Extract features for each set ResNet
def ExtractFeatures_ResNet(TRAIN='training', 
                           TEST='evaluation',
                           VAL='validation',
                           BATCH_SIZE=32,
                           BASE_PATH=os.path.join('TransferLearningKeras', 'dataset'),
                           BASE_CSV_PATH=os.path.join('TransferLearningKeras', 'output'),
                           LE_PATH=os.path.join('TransferLearningKeras', 'output'),
                          ):
    
    # load the ResNet50 network and initialize the label encoder
    print("[INFO] loading ResNet50 network...")
    model = ResNet50(weights="imagenet", include_top=False)
    le = None

    # loop over the data splits
    for split in (TRAIN, TEST, VAL):
        # grab all image paths in the current split
        print("[INFO] processing '{} split'...".format(split))
        p = os.path.sep.join([BASE_PATH, split])
        imagePaths = list(paths.list_images(p))

        # randomly shuffle the image paths and then extract the class
        # labels from the file paths
        random.shuffle(imagePaths)
        labels = [p.split(os.path.sep)[-2] for p in imagePaths]

        # if the label encoder is None, create it
        if le is None:
            le = LabelEncoder()
            le.fit(labels)

        # open the output CSV file for writing
        csvPath = os.path.sep.join([BASE_CSV_PATH,
            "{}.csv".format(split)])
        csv = open(csvPath, "w")

        # loop over the images in batches
        for (b, i) in enumerate(range(0, len(imagePaths), BATCH_SIZE)):
            # extract the batch of images and labels, then initialize the
            # list of actual images that will be passed through the network
            # for feature extraction
            print("[INFO] processing batch {}/{}".format(b + 1,
                int(np.ceil(len(imagePaths) / float(BATCH_SIZE)))))
            batchPaths = imagePaths[i:i + BATCH_SIZE]
            batchLabels = le.transform(labels[i:i + BATCH_SIZE])
            batchImages = []

            # loop over the images and labels in the current batch
            for imagePath in batchPaths:
                # load the input image using the Keras helper utility
                # while ensuring the image is resized to 224x224 pixels
                image = load_img(imagePath, target_size=(224, 224))
                image = img_to_array(image)

                # preprocess the image by (1) expanding the dimensions and
                # (2) subtracting the mean RGB pixel intensity from the
                # ImageNet dataset
                image = np.expand_dims(image, axis=0)
                image = imagenet_utils.preprocess_input(image)

                # add the image to the batch
                batchImages.append(image)

            # pass the images through the network and use the outputs as
            # our actual features, then reshape the features into a
            # flattened volume
            batchImages = np.vstack(batchImages)
            features = model.predict(batchImages, batch_size=BATCH_SIZE)
            features = features.reshape((features.shape[0], 7 * 7 * 2048))

            # loop over the class labels and extracted features
            for (label, vec) in zip(batchLabels, features):
                # construct a row that exists of the class label and
                # extracted features
                vec = ",".join([str(v) for v in vec])
                csv.write("{},{}\n".format(label, vec))

        # close the CSV file
        csv.close()

    # serialize the label encoder to disk
    f = open(LE_PATH, "wb")
    f.write(pickle.dumps(le))
    f.close()

# Extract features for specific Batch
def ExtractFeatures_batch_ResNet(image_batch_paths):
    # load the VGG16 network and initialize the label encoder
    print("[INFO] loading ResNet50 network...")
    model = ResNet50(weights="imagenet", include_top=False)    # No top layer

    # Containers 
    batchImages = []
    
    # loop over the images and labels in the current batch
    for imagePath in image_batch_paths:
        # Load image using the Keras helper utility
        # Resize to 224x224 pixels (to match VGG16 architecture)
        image = load_img(imagePath, target_size=(224, 224)) 
        
        image = img_to_array(image)

        # Preprocess: (1) expand the dimensions and (2) substract the mean RGB intensity
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)

        # Add the image *array* to the batch
        batchImages.append(image)

    # Pass the images through the network and use the outputs as
    # our actual features, then reshape the features into a
    # flattened volume = vector
    batchImages = np.vstack(batchImages)                                   # Stacked to match Keras format
    features = model.predict(batchImages, batch_size=len(batchImages))     # Get features for BS images
    features = features.reshape((features.shape[0], 7 * 7 * 2048))         # Reshape to create a flattened vector BS, FEAT
    
    # Return features
    return features
 
# Custom Data generator: to yield csv features
def csv_feature_generator(inputPath,
                          bs, 
                          numClasses, 
                          mode="train"):
    # open the input file for reading
    f = open(inputPath, "r")

    # loop indefinitely
    while True:
        # initialize our batch of data and labels
        data = []
        labels = []

        # keep looping until we reach our batch size
        while len(data) < bs:
            # attempt to read the next row of the CSV file
            row = f.readline()
        
            # check to see if the row is empty, indicating we have
            # reached the end of the file
            if row == "":
                # reset the file pointer to the beginning of the file
                # and re-read the row
                f.seek(0)
                row = f.readline()

                # if we are evaluating we should now break from our
                # loop to ensure we don't continue to fill up the
                # batch from samples at the beginning of the file
                if mode == "eval":
                    break

            # extract the class label and features from the row
            row = row.strip().split(",")
            label = row[0]
            label = to_categorical(label, num_classes=numClasses)
            features = np.array(row[1:], dtype="float")

            # update the data and label lists
            data.append(features)
            labels.append(label)

        # yield the batch to the calling function
        yield (np.array(data), np.array(labels))
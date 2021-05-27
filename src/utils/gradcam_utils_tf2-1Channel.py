# Importations
import sys

import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model

import tensorflow as tf
from tensorflow.python.framework import ops
tf.compat.v1.disable_eager_execution()

def build_model(model=None, 
                modelPath=None,
                model_constructor=None,
                dims=(38,31,1),
                pretrained='VGG16'):
    
    # Multiple returns
    if model is None and modelPath is None:
        if pretrained == 'VGG16':
            return VGG16(include_top=True, weights='imagenet')
        elif pretrained == 'ResNet50':
            return ResNet50(include_top=True, weights='imagenet')
    if model is not None:
        # Model from scratch
        model = model
        return model
        
    if modelPath is not None:
        return load_model(modelPath)

def load_image(path,
               targetSize=(32,38),
               preprocess=True):
    x = image.load_img(path, target_size=targetSize)
    if preprocess:
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
    return x

def deprocess_image(x):
    """Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    """
    x = x.copy()
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def normalize(x):
    return (x + 1e-10) / (K.sqrt(K.mean(K.square(x))) + 1e-10)

def build_guided_model(model):
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                   tf.cast(op.inputs[0] > 0., dtype)

    g = tf.compat.v1.get_default_graph()
    with g.gradient_override_map({'Relu': 'GuidedBackProp'}):
        new_model = build_model(model)
    return new_model

def guided_backprop(input_model,
                    images,
                    layer_name):
    input_imgs = input_model.input
    #print('input_imgs:', input_imgs)
    layer_output = input_model.get_layer(layer_name).output
    grads = K.gradients(layer_output, input_imgs)[0]
    backprop_fn = K.function([input_imgs, K.learning_phase()], [grads])
    grads_val = backprop_fn([images, 0])[0]
    return grads_val

def grad_cam(input_model,
             image,
             cls,
             layer_name,
             classes=[0],
             targetSize=(32,38)):
    #image = np.expand_dims(image, axis=0)
    loss = tf.gather_nd(input_model.output, np.dstack([range(1), classes])[0])
    layer_output = input_model.get_layer(layer_name).output
    grads = K.gradients(loss, layer_output)[0]
    gradient_fn = K.function([input_model.input, K.learning_phase()], [layer_output, grads])

    conv_output, grads_val = gradient_fn([image, 0])    
    weights = np.mean(grads_val, axis=(1, 2))
    cams = np.einsum('ijkl,il->ijk', conv_output, weights)
    
    # Process CAMs
    new_cams = np.empty((1, targetSize[0], targetSize[1]))
    #print("new_camsShape:", new_cams.shape)
    for i in range(new_cams.shape[0]):
        cam_i = cams[i] - cams[i].mean()
        cam_i = (cam_i + 1e-10) / (np.linalg.norm(cam_i, 2) + 1e-10)
        new_cams[i] = cv2.resize(cam_i, (targetSize[1], targetSize[0]), cv2.INTER_LINEAR)
        new_cams[i] = np.maximum(new_cams[i], 0)
        new_cams[i] = new_cams[i] / new_cams[i].max()
    
    return np.squeeze(new_cams, axis=0)

def grad_cam_batch(input_model,
                   images,
                   classes,
                   layer_name,
                   targetSize=(32,38)):
    loss = tf.gather_nd(input_model.output, np.dstack([range(images.shape[0]), classes])[0])
    layer_output = input_model.get_layer(layer_name).output
    grads = K.gradients(loss, layer_output)[0]
    gradient_fn = K.function([input_model.input, K.learning_phase()], [layer_output, grads])

    conv_output, grads_val = gradient_fn([images, 0])    
    weights = np.mean(grads_val, axis=(1, 2))
    cams = np.einsum('ijkl,il->ijk', conv_output, weights)
    
    # Process CAMs
    new_cams = np.empty((images.shape[0], targetSize[1], targetSize[0]))
    for i in range(new_cams.shape[0]):
        cam_i = cams[i] - cams[i].mean()
        cam_i = (cam_i + 1e-10) / (np.linalg.norm(cam_i, 2) + 1e-10)
        new_cams[i] = cv2.resize(cam_i, targetSize, cv2.INTER_LINEAR)
        new_cams[i] = np.maximum(new_cams[i], 0)
        new_cams[i] = new_cams[i] / new_cams[i].max()
    
    return new_cams

def compute_saliency(model,
                     guided_model,
                     img_path,
                     layer_name='block5_conv3',
                     cls=-1,
                     visualize=True,
                     save=True,
                     path=None,
                     top_n=2,
                     inputSize=(32,38),
                     channels=3,
                     size=(15, 10)):

    # Pre-Process image
    # Loop
    if channels == 1:
        image = cv2.imread(img_path, 0)
    else:
        image = cv2.imread(img_path)
    image = cv2.resize(image, (inputSize[1], inputSize[0]))
    
    if channels == 1:
        image = image.reshape((image.shape[0], image.shape[1], 1))
    
    if channels != 1:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # To array and process
    image = np.array(image)
    print("Shape image:", image.shape)
    preprocessed_input = image / 255.
    preprocessed_input = np.expand_dims(preprocessed_input, 0)
    print("Preprocessed inputs:", preprocessed_input.shape)
    # Gather predictions
    predictions = model.predict(preprocessed_input)
    
    # Get top n (5 by default)
    top = np.sort(predictions[0])[:top_n][::-1]
    classes = np.argsort(predictions[0])[-top_n:][::-1]
    print("top:", top)
    print("predictions:", predictions)
    print("classes:", classes)
    
    # Predictions
    print('Model prediction:')
    for c, p in zip(classes, np.array(top).flatten()):
        #print(c,p)
        label = 'fire' if c == 1 else 'no_fire'
        print('\t{:15s}\t({})\twith probability {:.3f}'.format(label, c, p))
    
    # If cls = -1, most likely classes
    if cls == -1:
        cls = np.argmax(predictions)
    class_name = "fire" if classes[0] == 1 else 'no_fire'
    print("Explanation for '{}'".format(class_name))
    #print("cls:", cls)
    
    # Calculate the 3 methods
    gradcam = generate_gradCAM(batch_size=1, 
                              layer=layer_name,
                              model=model,
                              processedimages= preprocessed_input, 
                              rawimages=preprocessed_input,
                              save=False,
                              showID=-1,
                              title='Test',)
    
    
    #gradcam = grad_cam(model, preprocessed_input / 255., cls, layer_name, classes=classes[cls], targetSize=inputSize)
    print("gradcam:", gradcam.shape)
    gb = guided_backprop(guided_model, preprocessed_input, layer_name)
    print("gb:", gb.shape)
    guided_gradcam = gb * gradcam[..., np.newaxis]
    print("guided_backprop:", guided_gradcam.shape)

    # Save outputs
    if save:
        jetcam = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
        jetcam = (np.float32(jetcam) + load_image(img_path, preprocess=False)) / 2
        if path is not None:
            gradcamPath = os.path.join(path, 'gradcam.png')
            guidedBackPath = os.path.join(path, 'guided_backprop.png')
            guidedcamPath = os.path.join(path, 'guided_gradcam.png')
        else:
            gradcamPath = 'gradcam.png'
            guidedBackPath = 'guided_backprop.png'
            guidedcamPath = 'guided_gradcam.png'
            
        cv2.imwrite(gradcamPath, np.uint8(jetcam))
        cv2.imwrite(guidedBackPath, deprocess_image(gb[0]))
        cv2.imwrite(guidedcamPath, deprocess_image(guided_gradcam[0]))
    
    # Visualize (show)
    if visualize:
        plt.figure(figsize=size)
        plt.subplot(131)
        plt.title('GradCAM')
        plt.axis('off')
        if len(image.shape) >= 3 and channels == 1:
            image = image.squeeze(-1)
        plt.imshow(image)
        plt.imshow(gradcam[0], cmap='jet', alpha=0.5)

        plt.subplot(132)
        plt.title('Guided Backprop')
        plt.axis('off')
        plt.imshow(np.flip(deprocess_image(gb[0]), -1))
        
        plt.subplot(133)
        plt.title('Guided GradCAM')
        plt.axis('off')
        plt.imshow(np.flip(deprocess_image(guided_gradcam[0]), -1))
        plt.show()
        
    return gradcam, gb, guided_gradcam

def generate_gradCAM(model, 
                    rawimages,
                    processedimages, 
                    layer, 
                    batch_size=32, 
                    save=False,
                    savefile=None,
                    title='',
                    showID=0):
    # Let's set classes for explanations as most probable class for each image.
    top = np.argmax(model.predict(processedimages), 1)
    gradcam = np.empty((processedimages.shape[:-1]))
    
    # Number of pictures
    N = processedimages.shape[0]

    # Batch loop
    for i in range((N + batch_size - 1) // batch_size):
        start = i * batch_size
        end = min((i+1) * batch_size, N)
        gradcam[start:end] = grad_cam_batch(model, 
                                            processedimages[start:end],
                                            top[start:end],
                                            layer,
                                            (processedimages.shape[2], processedimages.shape[1]))

    # Save file
    if save:
        outfile = 'gradcam.lzma' if savefile is None else savefile
        cdump(gradcam, outfile, compression='lzma')

    # Show
    if showID >= 0: 
        i = showID
        plt.title(title)
        plt.imshow(rawimages[i])
        plt.imshow(gradcam[i], alpha=0.3, cmap='jet')
        plt.show()
    
    # Return gradcam array
    return gradcam

def show_gradCAM(rawimage, 
                gradcam, 
                showID=0, 
                title=''):
    i = showID
    plt.title(title)
    plt.imshow(rawimage[i])
    plt.imshow(gradcam[i], alpha=0.5, cmap='jet')
    plt.show()
    
def generate_guidedbackprop(guided_model, 
                            processedimages, 
                            deprocess_object,
                            layer, 
                            batch_size=32, 
                            save=False,
                            savefile=None,
                            title='',
                            showID=0):
    # Container
    gbp = np.empty((processedimages.shape))
    N = processedimages.shape[0]

    # Batch loop
    for i in range((N + batch_size - 1) // batch_size):
        start = i * batch_size
        end = min((i+1) * batch_size, N)
        gbp[start:end] = guided_backprop(guided_model, 
                                         processedimages[start:end], 
                                         layer)

    # Save
    if save:
        outfile = 'guided_backprop.lzma' if savefile is None else savefile
        cdump(gradcam, outfile, compression='lzma')
    
    # Show
    if showID >= 0:
        i = showID
        plt.title(title)
        plt.imshow(np.flip(deprocess_object(gbp[i]), -1), cmap='jet')
        plt.show()
        
    # Return guided backprop
    return gbp

def show_guidedBP(gbp, 
                  deprocess_object,
                  title='', 
                  showID=0):
    i = showID
    plt.title(title)
    plt.imshow(np.flip(deprocess_object(gbp[i]), -1), cmap='jet')
    plt.show()
    
def generate_guidedgradCAM(gbp, 
                           gradcam,
                           showID=0,
                           save=False,
                           savefile=None,
                           title='',
                           deprocess_object=None):
    # Guided gradCam
    guided_gradcam = gbp * gradcam[..., np.newaxis]

    # Save
    if save:
        outfile = 'guided_gradcam.lzma' if savefile is None else savefile
        cdump(guided_gradcam, 'guided_gradcam.lzma', compression='lzma')
    
    # Predictions
    if showID >= 0:
        i = showID
        plt.title(title)
        plt.imshow(deprocess_object(guided_gradcam[i]), 
                   alpha=0.5, cmap='jet')
        plt.show()
        
    # Return guided gradcam
    return guided_gradcam

def show_guidedgradCAM(ggcam, 
                       title='', 
                       showID=0, 
                       deprocess_object=None):
    # Predictions
    i = showID
    plt.title(title)
    plt.imshow(deprocess_object(ggcam[i]), 
               alpha=0.5, cmap='jet')
    plt.show()
    
# Process rawimages and gcam and save gcam ones (returns array of images)
def gcam_processed(rawimages, 
                   gcam,
                   outGCAM=os.path.join('GCAM_output'),
                   show=False, 
                   size=(5,5), 
                   outsize=(128,128)):
    # Process a batch of rawimages
    if not os.path.exists(outGCAM):
        os.makedirs(outGCAM)

    # Size
    if show:
        plt.rcParams['figure.figsize'] = size

    # Save processed pictures
    for idx, image in enumerate(rawimages):
        fileName = os.path.join(outGCAM, str(idx) + '.png')
        im = plt.imshow(rawimages[idx])
        im2 = plt.imshow(gcam[idx], alpha=0.3, cmap='jet')
        plt.axis('off')
        plt.tight_layout()
        plt.axis("tight")  # gets rid of white border
        plt.axis("image")  # square up the image instead of filling the "figure" space
        plt.savefig(fileName, bbox_inches='tight', pad_inches=0.0)

    # Read back the gcam processed
    GCAM_imagesPaths = sorted(list(paths.list_images(outGCAM)))
    GCAM_images = []
    
    # Processe back
    for imagepath in GCAM_imagesPaths:
        image = cv2.imread(imagepath)
        image = cv2.resize(image, outsize)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        GCAM_images.append(image)    
        
    # Return
    return GCAM_images

# Process rawimages and gcam and save gcam ones (returns array of images)
def gprop_processed(gprop,
                    outGPROP=os.path.join('GPROP_output'),
                    deprocess_object=None,
                    show=False, 
                    size=(5,5), 
                    outsize=(128,128)):
    # Process a batch of rawimages
    if not os.path.exists(outGPROP):
        os.makedirs(outGPROP)

    # Save processed pictures
    for idx, image in enumerate(rawimages):
        plt.imsave(os.path.join(outGPROP, str(idx) + '.png'), 
                   np.flip(deprocess_object(gprop[idx]), -1), 
                   cmap='jet', format='png')
        
    # Show
    if show:
        plt.rcParams['figure.figsize'] = size
        plt.imshow(np.flip(deprocess_object(gprop[idx]), -1), 
                   cmap='jet',)

    # Read back the gcam processed
    GPROP_imagesPaths = sorted(list(paths.list_images(outGPROP)))
    GPROP_images = []
    
    # Processe back
    for imagepath in GPROP_imagesPaths:
        image = cv2.imread(imagepath)
        image = cv2.resize(image, outsize)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        GPROP_images.append(image)    
        
    # Return
    return GPROP_images

# Process rawimages and gcam and save gcam ones (returns array of images)
def guidedGCAM_processed(gprop,
                         gradcam,
                         outGGCAM=os.path.join('GGCAM_output'),
                         deprocess_object=None,
                         show=False, 
                         size=(5,5), 
                         alpha=0.5,
                         outsize=(128,128)):
    # Process a batch of rawimages
    if not os.path.exists(outGGCAM):
        os.makedirs(outGGCAM)
        
    # Guided gradCam
    guided_gradcam = gprop * gradcam[..., np.newaxis]

    # Save processed pictures
    for idx, image in enumerate(guided_gradcam):
        plt.imsave(os.path.join(outGGCAM, str(idx) + '.png'), 
                   np.flip(deprocess_object(guided_gradcam[idx]), -1), 
                   cmap='jet', format='png')
        
    # Show
    if show:
        plt.rcParams['figure.figsize'] = size
        plt.imshow(np.flip(deprocess_object(guided_gradcam[idx]), -1), 
                   cmap='jet', alpha=alpha)

    # Read back the gcam processed
    GGCAM_imagesPaths = sorted(list(paths.list_images(outGGCAM)))
    GGCAM_images = []
    
    # Processe back
    for imagepath in GGCAM_imagesPaths:
        image = cv2.imread(imagepath)
        image = cv2.resize(image, outsize)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        GGCAM_images.append(image)    
        
    # Return
    return GGCAM_images

# plot all maps in squares
def plot_filters(feature_maps, 
                 size=(20,20), 
                 cmap=None):
    square = np.sqrt(feature_maps.shape[-1]).astype(np.int)
    ix = 1
    for _ in range(square):
        for _ in range(square):
            # specify subplot and turn of axis
            ax = plt.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])

            # plot filter channel in grayscale
            plt.imshow(feature_maps[0, :, :, ix-1], cmap=cmap)
            ix += 1

    # show the figure
    plt.rcParams['figure.figsize'] = size[0], size[1]
    plt.show()   
    
# Get conv layers for feature maps
def get_featuremaps_model(model, layers_idxs):
    # Check model
    if model == 'VGG16':
        model = VGG16()
    if model == 'ResNet50':
        model = ResNet50()
    
    # Get Outputs
    outputs = [model.layers[i+1].output for i in ixs]
    
    # Generate new model
    model = Model(inputs=model.inputs, outputs=outputs)
    
    # New model with multiple outputs
    return model
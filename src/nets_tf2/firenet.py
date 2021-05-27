# Importations
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, SeparableConv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras import backend as K

# Main class: FireNet with Separable CONV layers to decrease the number of trainable parameters
class FireNet:
    @staticmethod
    def build_model(width, height, depth, classes):
        # Sequential model
        model = Sequential()
        
        # Shape: HxWxC
        inputShape = (height, width, depth)
        chanDim = -1

        # Check if channels are first to reshape inputs
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # Block 1: CONV 32, RELU, BN, MPOOL, DROPOUT
        model.add(SeparableConv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Block 2: (CONV 64, RELU, BN) x 2, MPOOL, DROPOUT
        model.add(SeparableConv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(SeparableConv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Block 3: (CONV 128, RELU, BN) x 3, MPOOL, DROPOUT
        model.add(SeparableConv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(SeparableConv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(SeparableConv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # FLATTEN values and connect to DENSE (256), REL, BN, DROPOUT
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # DENSE (2) and SOFTMAX
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # Return the network
        return model
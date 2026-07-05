import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, SeparableConv2D, MaxPooling2D, BatchNormalization,
    Input, Add, GlobalAveragePooling2D, Dense, Dropout, ReLU, SpatialDropout2D
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


def residual_block(x, filters, dropout_rate=0.2, weight_decay=1e-4):
    shortcut = x

    # First conv
    x = Conv2D(filters, (3, 3), padding='same',
            kernel_initializer="he_normal",
            kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = SpatialDropout2D(dropout_rate)(x)

    # Second conv
    x = Conv2D(filters, (3, 3), padding='same',
            kernel_initializer="he_normal",
            kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Match shortcut channels if needed
    if shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), padding='same',
                        kernel_initializer="he_normal",
                        kernel_regularizer=l2(weight_decay))(shortcut)

    x = Add()([shortcut, x])
    return x


def depthwise_block(x, filters, dropout_rate=0.2, weight_decay=1e-4):
    x = SeparableConv2D(filters, (3, 3), padding='same',
                        depthwise_initializer="he_normal",
                        pointwise_initializer="he_normal",
                        depthwise_regularizer=l2(weight_decay),
                        pointwise_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = SpatialDropout2D(dropout_rate)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    return x


def build_cropnet_hybrid(input_shape=(224, 224, 3), num_classes=38):
    inputs = Input(shape=input_shape)

    # Initial conv (using stride instead of pooling for efficiency)
    x = Conv2D(32, (3, 3), strides=(2, 2), padding='same',
            kernel_initializer="he_normal")(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Hybrid Blocks
    x = depthwise_block(x, 64)
    x = residual_block(x, 64)

    x = depthwise_block(x, 128)
    x = residual_block(x, 128)

    x = depthwise_block(x, 256)
    x = residual_block(x, 256)

    # Classifier
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)   # standard dropout for dense layers
    x = Dense(256, activation='relu', kernel_initializer="he_normal")(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs, name="CropNet_Hybrid")
    return model

import ssl, certifi
ssl._create_default_https_context = lambda *args, **kwargs: ssl.create_default_context(cafile=certifi.where())

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D, Concatenate, Input, Add
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50


def residual_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding="same", use_bias=False)(inputs)
    x = BatchNormalization()(x); x = Activation("relu")(x)
    x = Conv2D(num_filters, 3, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)

    s = Conv2D(num_filters, 1, padding="same", use_bias=False)(inputs)
    s = BatchNormalization()(s)

    x = Add()([x, s])
    x = Activation("relu")(x)
    return x


def dilated_conv(inputs, num_filters):
    x1 = Conv2D(num_filters, 3, padding="same", dilation_rate=3, use_bias=False)(inputs)
    x1 = BatchNormalization()(x1); x1 = Activation("relu")(x1)

    x2 = Conv2D(num_filters, 3, padding="same", dilation_rate=6, use_bias=False)(inputs)
    x2 = BatchNormalization()(x2); x2 = Activation("relu")(x2)      # <- BN/ReLU on x2

    x3 = Conv2D(num_filters, 3, padding="same", dilation_rate=9, use_bias=False)(inputs)
    x3 = BatchNormalization()(x3); x3 = Activation("relu")(x3)

    x = Concatenate()([x1, x2, x3])
    x = Conv2D(num_filters, 1, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x); x = Activation("relu")(x)
    return x


def decoder_block(x, skip, filters):
    x = UpSampling2D((2, 2), interpolation="bilinear")(x)
    if isinstance(skip, (list, tuple)):   # safety in case a list sneaks in
        skip = skip[0]
    x = Concatenate(axis=-1)([x, skip])
    x = residual_block(x, filters)
    return x


def build_model(inp_shape):
    inputs = Input(inp_shape, name="image")
    resnet50 = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)

    # Encoder taps (use stable handles, not hardcoded input layer names)
    s0 = resnet50.input                               # 1/1
    s1 = resnet50.get_layer("conv1_relu").output      # 1/2
    s2 = resnet50.get_layer("conv2_block3_out").output  # 1/4
    s3 = resnet50.get_layer("conv3_block4_out").output  # 1/8
    s4 = resnet50.get_layer("conv4_block6_out").output  # 1/16

    b1 = dilated_conv(s4, 1024)                       # bottleneck at 1/16

    # Decoder
    d1 = decoder_block(b1, s3, 512)   # 1/8
    d2 = decoder_block(d1, s2, 256)   # 1/4
    d3 = decoder_block(d2, s1, 128)   # 1/2
    d4 = decoder_block(d3, s0, 64)    # 1/1

    # Deep-supervision maps
    y1 = UpSampling2D((8, 8), interpolation="bilinear")(d1); y1 = Conv2D(1, 1, activation="sigmoid")(y1)
    y2 = UpSampling2D((4, 4), interpolation="bilinear")(d2); y2 = Conv2D(1, 1, activation="sigmoid")(y2)
    y3 = UpSampling2D((2, 2), interpolation="bilinear")(d3); y3 = Conv2D(1, 1, activation="sigmoid")(y3)
    y4 = Conv2D(1, 1, activation="sigmoid")(d4)

    # Fuse to a single-channel mask (so it matches (H,W,1) labels)
    # used to be 'y' not 'outputs' in line below
    outputs = Concatenate()([y1, y2, y3, y4])              # (H, W, 4)
    # outputs = Conv2D(1, 1, activation="sigmoid")(y)  # (H, W, 1)

    return Model(inputs, outputs, name="UNet")


if __name__ == "__main__":
    input_shape = (512, 512, 3)
    model = build_model(input_shape)
    model.summary()




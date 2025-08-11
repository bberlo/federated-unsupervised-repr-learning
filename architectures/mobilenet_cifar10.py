from tensorflow import keras


def mobilenet_cifar10(name):

	def conv_bn_rl(x, f, k=1, s=1, p='same'):
		x = keras.layers.Conv2D(f, k, strides=s, padding=p)(x)
		x = keras.layers.BatchNormalization()(x)
		x = keras.layers.ReLU()(x)
		return x

	def mobilenet_block(x, f, s=1):
		x = keras.layers.DepthwiseConv2D(3, s, padding='same')(x)
		x = keras.layers.BatchNormalization()(x)
		x = keras.layers.ReLU()(x)

		x = conv_bn_rl(x, f)
		return x

	i = keras.layers.Input((32, 32, 3))

	x = conv_bn_rl(i, 32, 3, 2)
	x = mobilenet_block(x, 64)

	x = mobilenet_block(x, 128, 2)
	x = mobilenet_block(x, 128)

	x = mobilenet_block(x, 256, 2)
	x = mobilenet_block(x, 256)

	x = mobilenet_block(x, 512, 2)

	x = keras.layers.AvgPool2D(2)(x)

	x = keras.layers.Flatten()(x)
	output = keras.layers.Dense(10, activation='softmax')(x)

	model = keras.models.Model(i, output, name=name)
	return model

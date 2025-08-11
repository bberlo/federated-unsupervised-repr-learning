from tensorflow import keras


class BackboneAE:
	def __init__(self, input_shape=(400, 3), l2_rate=0.0001, dp_rate=0.1):
		self.input_shape = input_shape
		self.l2_rate = l2_rate
		self.dp_rate = dp_rate

	def get_model(self):
		i_acc = keras.layers.Input(shape=self.input_shape, name="acc_input")
		i_gyr = keras.layers.Input(shape=self.input_shape, name="gyr_input")

		c_a_acc = keras.layers.Conv1D(32, kernel_size=24,
						strides=1,
						activation=None,
						padding="same", kernel_regularizer=keras.regularizers.l2(self.l2_rate), kernel_initializer='he_uniform',
						use_bias=False, name="c_a_acc")

		c_b_acc = keras.layers.Conv1D(64, kernel_size=16,
						strides=1,
						activation=None,
						padding="same", kernel_regularizer=keras.regularizers.l2(self.l2_rate), kernel_initializer='he_uniform',
						use_bias=False, name="c_b_acc")

		c_c_acc = keras.layers.Conv1D(96, kernel_size=8,
						strides=1,
						activation=None,
						padding="same", kernel_regularizer=keras.regularizers.l2(self.l2_rate), kernel_initializer='he_uniform',
						use_bias=False, name="c_c_acc")

		d_c_a_acc = keras.layers.Conv1D(96, kernel_size=8,
						strides=1,
						activation=None,
						padding="same", kernel_regularizer=keras.regularizers.l2(self.l2_rate), kernel_initializer='he_uniform',
						use_bias=False, name="d_c_a_acc")

		d_c_b_acc = keras.layers.Conv1D(64, kernel_size=16,
						strides=1,
						activation=None,
						padding="same", kernel_regularizer=keras.regularizers.l2(self.l2_rate), kernel_initializer='he_uniform',
						use_bias=False, name="d_c_b_acc")

		d_c_c_acc = keras.layers.Conv1D(32, kernel_size=24,
						strides=1,
						activation=None,
						padding="same", kernel_regularizer=keras.regularizers.l2(self.l2_rate), kernel_initializer='he_uniform',
						use_bias=False, name="d_c_c_acc")

		d_c_d_acc = keras.layers.Conv1D(3, kernel_size=4,
						strides=1,
						activation=None,
						padding="same", kernel_regularizer=keras.regularizers.l2(self.l2_rate), kernel_initializer='he_uniform',
						use_bias=False, name="d_c_d_acc")

		c_a_gyr = keras.layers.Conv1D(32, kernel_size=24,
						strides=1,
						activation=None,
						padding="same", kernel_regularizer=keras.regularizers.l2(self.l2_rate), kernel_initializer='he_uniform',
						use_bias=False, name="c_a_gyr")

		c_b_gyr = keras.layers.Conv1D(64, kernel_size=16,
						strides=1,
						activation=None,
						padding="same", kernel_regularizer=keras.regularizers.l2(self.l2_rate), kernel_initializer='he_uniform',
						use_bias=False, name="c_b_gyr")

		c_c_gyr = keras.layers.Conv1D(96, kernel_size=8,
						strides=1,
						activation=None,
						padding="same", kernel_regularizer=keras.regularizers.l2(self.l2_rate), kernel_initializer='he_uniform',
						use_bias=False, name="c_c_gyr")

		d_c_a_gyr = keras.layers.Conv1D(96, kernel_size=8,
						strides=1,
						activation=None,
						padding="same", kernel_regularizer=keras.regularizers.l2(self.l2_rate), kernel_initializer='he_uniform',
						use_bias=False, name="d_c_a_gyr")

		d_c_b_gyr = keras.layers.Conv1D(64, kernel_size=16,
						strides=1,
						activation=None,
						padding="same", kernel_regularizer=keras.regularizers.l2(self.l2_rate), kernel_initializer='he_uniform',
						use_bias=False, name="d_c_b_gyr")

		d_c_c_gyr = keras.layers.Conv1D(32, kernel_size=24,
						strides=1,
						activation=None,
						padding="same", kernel_regularizer=keras.regularizers.l2(self.l2_rate), kernel_initializer='he_uniform',
						use_bias=False, name="d_c_c_gyr")

		d_c_d_gyr = keras.layers.Conv1D(3, kernel_size=4,
						strides=1,
						activation=None,
						padding="same", kernel_regularizer=keras.regularizers.l2(self.l2_rate), kernel_initializer='he_uniform',
						use_bias=False, name="d_c_d_gyr")

		c_a_ct = keras.layers.Conv1D(128, kernel_size=4,
						strides=1,
						activation=None,
						padding="same", kernel_regularizer=keras.regularizers.l2(self.l2_rate), kernel_initializer='he_uniform',
						use_bias=False, name="c_a_ct")

		# -----------------------------------Encoder----------------------------------------- #

		ac = c_a_acc(i_acc)

		ac = keras.layers.Activation("relu")(ac)
		ac = keras.layers.MaxPool1D(4, 2, padding='same')(ac)

		ac = c_b_acc(ac)

		ac = keras.layers.Activation("relu")(ac)
		ac = keras.layers.MaxPool1D(4, 2, padding='same')(ac)

		ac = c_c_acc(ac)

		ac = keras.layers.Activation("relu")(ac)
		ac = keras.layers.Dropout(self.dp_rate)(ac)

		gy = c_a_gyr(i_gyr)

		gy = keras.layers.Activation("relu")(gy)
		gy = keras.layers.MaxPool1D(4, 2, padding='same')(gy)

		gy = c_b_gyr(gy)

		gy = keras.layers.Activation("relu")(gy)
		gy = keras.layers.MaxPool1D(4, 2, padding='same')(gy)

		gy = c_c_gyr(gy)

		gy = keras.layers.Activation("relu")(gy)
		gy = keras.layers.Dropout(self.dp_rate)(gy)

		ct = keras.layers.Concatenate()([ac, gy])
		ct = c_a_ct(ct)
		# ct = keras.layers.BatchNormalization(momentum=0.99, scale=False)(ct)
		e = keras.layers.Activation("relu")(ct)

		# ------------------------------Decoder---------------------------------------------- #

		d_ac = d_c_a_acc(e)

		d_ac = keras.layers.Activation("relu")(d_ac)
		d_ac = keras.layers.UpSampling1D(2)(d_ac)

		d_ac = d_c_b_acc(d_ac)

		d_ac = keras.layers.Activation("relu")(d_ac)
		d_ac = keras.layers.UpSampling1D(2)(d_ac)

		d_ac = d_c_c_acc(d_ac)

		d_ac = keras.layers.Activation("relu")(d_ac)

		d_ac = d_c_d_acc(d_ac)

		d_ac = keras.layers.Activation("relu")(d_ac)

		d_gy = d_c_a_gyr(e)

		d_gy = keras.layers.Activation("relu")(d_gy)
		d_gy = keras.layers.UpSampling1D(2)(d_gy)

		d_gy = d_c_b_gyr(d_gy)

		d_gy = keras.layers.Activation("relu")(d_gy)
		d_gy = keras.layers.UpSampling1D(2)(d_gy)

		d_gy = d_c_c_gyr(d_gy)

		d_gy = keras.layers.Activation("relu")(d_gy)

		d_gy = d_c_d_gyr(d_gy)

		d_gy = keras.layers.Activation("relu")(d_gy)

		gmp_e = keras.layers.GlobalMaxPooling1D()(e)
		return keras.models.Model([i_acc, i_gyr], [gmp_e, d_ac, d_gy])

from keras import backend as K


class Config:

	def __init__(self):

		self.verbose = True

		self.network = 'vgg'

		# setting for data augmentation
		self.use_horizontal_flips = False
		self.use_vertical_flips = False
		self.rot_90 = False

		# anchor box scales
		self.anchor_box_scales = [32, 64, 96, 128, 196]

		# anchor box ratios
		#self.anchor_box_ratios = [[1, 1], [0.5, 1], [1, 0.5]]
		self.anchor_box_ratios = [[1, 1], [0.5, 1], [1, 0.5]]

		# size to resize the smallest side of the image
		self.im_size = 512

		# image channel-wise mean to subtract
		self.img_channel_mean = [103.939, 116.779, 123.68]
		self.img_scaling_factor = 2.0

		# number of ROIs at once default 4
		self.num_rois = 8

		# stride at the RPN (this depends on the network configuration) default 16
		self.rpn_stride = 32

		# default False
		self.balanced_classes = False

		# scaling the stdev
		#self.std_scaling = 4.0
		self.std_scaling = 2.0

		self.classifier_regr_std = [8.0, 8.0, 4.0, 4.0]

		# overlaps for RPN
		self.rpn_min_overlap = 0.3
		self.rpn_max_overlap = 0.7

		# overlaps for classifier ROIs
		self.classifier_min_overlap = 0.1
		self.classifier_max_overlap = 0.5

		# placeholder for the class mapping, automatically generated by the parser
		self.class_mapping = None

		#location of pretrained weights for the base network 
		# weight files can be found at:
		# https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels_notop.h5
		# https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5

		self.model_path = 'model_frcnn.hdf5'

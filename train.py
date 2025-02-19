import os
import numpy as np
import torch
from PIL import Image

import torchvision
from torchvision.models import MobileNet_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import FasterRCNN, FasterRCNN_ResNet50_FPN_Weights, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.rpn import AnchorGenerator

from references.engine import train_one_epoch, evaluate
import references.utils
import references.transforms as T

# Define Global Variables
# load a model pre-trained on COCO

# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

# replace the classifier with a new one, that has
num_classes = 5  # 4 class (breech face) + background

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features 
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=num_classes)

# num_classes which is user-defined
# load a pre-trained model for classification and return
# only the features
backbone = torchvision.models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).features

# FasterRCNN needs to know the number of
# output channels in a backbone. For mobilenet_v2, it's 1280
# so we need to add it here
backbone.out_channels = 1280

# let's make the RPN generate 5 x 3 anchors per spatial
# location, with 5 different sizes and 3 different aspect	
# ratios. We have a Tuple[Tuple[int]] because each feature
# map could potentially have different sizes and
# aspect ratios
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))

# let's define what are the feature maps that we will
# use to perform the region of interest cropping, as well as
# the size of the crop after rescaling.
# if your backbone returns a Tensor, featmap_names is expected to
# be [0]. More generally, the backbone should return an
# OrderedDict[Tensor], and in featmap_names you can choose which
# feature maps to use.
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                output_size=7,
                                                sampling_ratio=2)

# # put the pieces together inside a FasterRCNN model
model = FasterRCNN(backbone,
                   num_classes=5,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)

# number of cycles or epochs for training
num_epochs = 3


# Cartridge dataset used for training and testing
class CartridgeDataset(torch.utils.data.Dataset):
	def __init__(self, root, transforms):
		self.root = root
		self.transforms = transforms
		# Dataset consists of six classes
		#	1) Aperture shear
		#	2) Firing pin drag
		#	3) Firing pin impression
		#	4) Breech face

		# Collecting data on files
		ApertureShearImage = [os.path.join(root, "Aperture_shear/image/" + file) for file in list(os.listdir(os.path.join(root, "Aperture_shear/image")))]
		ApertureShearMask = [os.path.join(root, "Aperture_shear/mask/" + file) for file in list(os.listdir(os.path.join(root, "Aperture_shear/mask")))]
		FiringpindragImage = [os.path.join(root, "Firing_pin_drag/image/" + file) for file in list(os.listdir(os.path.join(root, "Firing_pin_drag/image")))]
		FiringpindragMask = [os.path.join(root, "Firing_pin_drag/mask/" + file) for file in list(os.listdir(os.path.join(root, "Firing_pin_drag/mask")))]
		FiringpinimpressionImage = [os.path.join(root, "Firing_pin_impression/image/" + file) for file in list(os.listdir(os.path.join(root, "Firing_pin_impression/image")))]
		FiringpinimpressionMask = [os.path.join(root, "Firing_pin_impression/mask/" + file) for file in list(os.listdir(os.path.join(root, "Firing_pin_impression/mask")))]
		BreechfaceImage = [os.path.join(root, "Breech_face/image/" + file) for file in list(os.listdir(os.path.join(root, "Breech_face/image")))]
		BreechfaceMask = [os.path.join(root, "Breech_face/mask/" + file) for file in list(os.listdir(os.path.join(root, "Breech_face/mask")))]
		self.images = sorted(ApertureShearImage 
			+ FiringpindragImage 
			+ FiringpinimpressionImage 
			+ BreechfaceImage)
		self.masks = sorted(ApertureShearMask
			+ FiringpindragMask
			+ FiringpinimpressionMask
			+ BreechfaceMask)

	def __getitem__(self, idx):
		# load images and masks
		image_path = self.images[idx]
		mask_path = self.masks[idx]
		image = Image.open(image_path).convert("RGB")
		# note that we haven't converted the mask to RGB
		# because each color corresponds to a different instance
		# with 0 being background
		mask = Image.open(mask_path)
		# convert the PIL Image into a numpy array
		mask = np.array(mask)
		# instances are encoded as different colors
		obj_ids = np.unique(mask)
		# first id is the background, so remove it
		obj_ids = obj_ids[1:]

		# split the color-encoded mask into a set
		# of binary masks
		masks = mask == obj_ids[:, None, None]

		label_dict = {'Aperture_shear' : 1,
						'Firing_pin_drag' : 2,
						'Firing_pin_impression' : 3,
						'Breech_face' : 4}

		# get bounding box coordinates for each mask
		num_objs = len(obj_ids)
		boxes = []
		labels = []
		for i in range(len(obj_ids)):
			path = os.path.normpath(mask_path)
			ans = path.split(os.sep)[1]
			labels.append(label_dict[ans])
			pos = np.where(masks[i])
			xmin = np.min(pos[1])
			xmax = np.max(pos[1])
			ymin = np.min(pos[0])
			ymax = np.max(pos[0])
			boxes.append([xmin, ymin, xmax, ymax])

		# convert everything into a torch.Tensor
		boxes = torch.as_tensor(boxes, dtype=torch.float32)
		# there is only one classes
		labels = torch.as_tensor(labels, dtype=torch.int64)
		masks = torch.as_tensor(masks, dtype=torch.uint8)

		image_id = torch.tensor([idx])
		area = (boxes[:,3] - boxes[:, 1]) *(boxes)
		# suppose all instances are not crowd
		iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

		target = {}
		target["boxes"] = boxes
		target["labels"] = labels
		target["masks"] = masks
		target["image_id"] = image_id
		target["area"] = area
		target["iscrowd"] = iscrowd

		if self.transforms is not None:
			image, target = self.transforms(image, target)

		return image, target

	def __len__(self):
		return len(self.images)


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    # model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    print("no.of in features:", in_features)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    print("no.of f mask", in_features_mask)

    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def train():
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	# device = torch.device('cpu')
	print("device name:", device)


	# dataset and defined transformations
	dataset = CartridgeDataset('data', get_transform(train=True))
	dataset_test = CartridgeDataset('data', get_transform(train=False))

	# print("label: ", dataset[1][1]['labels'])

	# split the dataset in train and test set
	# keeping it the same for now
	indices = torch.randperm(len(dataset)).tolist()
	dataset = torch.utils.data.Subset(dataset, indices)
	dataset_test = torch.utils.data.Subset(dataset_test, indices)

	# define training and validation data loaders
	data_loader = torch.utils.data.DataLoader(
		dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=references.utils.collate_fn)
	data_loader_test = torch.utils.data.DataLoader(
		dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=references.utils.collate_fn)

	# get the model
	model = get_model_instance_segmentation(num_classes)

	# move model to the right device
	model.to(device)

	# construct an optimizer
	params = [p for p in model.parameters() if p.requires_grad]
	optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
	# learning rate scheduler
	lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

	# training loop
	for epoch in range(num_epochs):
		# train for one epoch, printing every 10 iterations
		train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
		# update the learning rate
		lr_scheduler.step()
		# evaluate on the test dataset
		evaluate(model, data_loader_test, device=device)

	print("That's it!")
	return model

if __name__ == '__main__':
	finalModel = train()
	# Saving Model for future inference
	torch.save(finalModel.state_dict(), "CartridgeMaskRCNN.pth")
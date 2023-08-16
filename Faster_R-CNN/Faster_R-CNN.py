from skimage import io
from skimage.transform import resize
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from utils import *
from model import *
from metrics import *
import matplotlib.pyplot as plt
import random
import os

## Class for loading Images and their corresponding annotations

class ObjectDetectionDataset(Dataset):
    '''
    A Pytorch Dataset class to load the images and their corresponding annotations.
    Returns:
    ------------
    images: torch.Tensor of size (B, C, H, W)
    gt bboxes: torch.Tensor of size (B, max_objects, 4)
    gt classes: torch.Tensor of size (B, max_objects)
    '''
    def __init__(self, annotation_path_all, img_dir, img_size, name2idx):
        self.annotation_path_all = annotation_path_all
        self.img_dir = img_dir
        self.img_size = img_size
        self.name2idx = name2idx
        self.img_data_all, self.get_bboxes_all, self.get_classes_all = self.get_data()

    def __len__(self):
        return self.img_data_all.size(dim=0)

    def __getitem__(self, idx):
        return self.img_data_all[idx], self.get_bboxes_all[idx], self.get_classes_all[idx]

    def get_data(self):
        img_data_all = []
        get_idxs_all = []

        get_boxes_all, get_classes_all, img_paths = parse_annotation(self.annotation_path_all, self.img_dir, self.img_size)

        for i, img_path in enumerate(img_paths):

            # skip if the image path is not valid
            if (not img_path) or (not os.path.exists(img_path)):
                continue

            # read and resize image
            img = io.imread(img_path)
            img = resize(img, self.img_size)

            # convert image to torch tensor and reshape it so channels come first
            img_tensor = torch.from_numpy(img).permute(2, 0, 1)

            # encode class names as integers
            get_classes = get_classes_all[i]
            get_idx = torch.Tensor([self.name2idx[name] for name in get_classes])

            img_data_all.append(img_tensor)
            get_idxs_all.append(get_idx)

        # pad bounding boxes and classes so they are of the same size
        get_bboxes_pad = pad_sequence(get_boxes_all, batch_first=True, padding_value=-1)
        get_classes_pad = pad_sequence(get_idxs_all, batch_first=True, padding_value=-1)

        # stack all images
        img_data_stacked = torch.stack(img_data_all, dim=0)

        return img_data_stacked.to(dtype=torch.float32), get_bboxes_pad, get_classes_pad

# image setting
img_width = 320
img_height = 240
image_dir = os.path.join("data", "Dogs")
name2idx = {'Airedale':0, 'miniature_poodle':1, 'affenpinscher':2, 'schipperke':3, 'Australian_terrier':4, 'Welsh_springer_spaniel':5,
'curly_coated_retriever':6, 'Staffordshire_bullterrier':7, 'Norwich_terrier':8, 'Tibetan_terrier':9, 'English_setter':10,
'Norfolk_terrier':11, 'Pembroke':12, 'Tibetan_mastiff':13, 'Border_terrier':14, 'Great_Dane':15, 'Scotch_terrier':16,
'flat_coated_retriever':17, 'Saluki':18, 'Irish_setter':19, 'Blenheim_spaniel':20, 'Irish_terrier':21, 'bloodhound':22,
'redbone':23, 'West_Highland_white_terrier':24, 'Brabancon_griffo':25, 'dhole':26, 'kelpie':27, 'Doberman':28,
'Ibizan_hound':29, 'vizsla':30, 'cairn':31, 'German_shepherd':32, 'African_hunting_dog':33, 'Dandie_Dinmont':34,
'Sealyham_terrier':35, 'German_short_haired_pointer':36, 'Bernese_mountain_dog':37, 'Saint_Bernard':38,
'Leonberg':39, 'Bedlington_terrier':40, 'Newfoundland':41, 'Lhasa':42, 'Chesapeake_Bay_retriever':43,
'Lakeland_terrier':44, 'Walker_hound':45, 'American_Staffordshire_terrier':46, 'otterhound':47, 'Sussex_spaniel':48,
'Norwegian_elkhound':49, 'bluetick':50, 'dingo':51, 'Irish_water_spaniel':52, 'Fila Braziliero':53, 'standard_schnauzer':54,
'Mexican_hairless':55, 'EntleBucher':56, 'Afghan_hound':57, 'kuvasz':58, 'English_foxhound':59, 'keeshond':60,
'Irish_wolfhound':61, 'Scottish_deerhound':62, 'Rottweiler':63, 'black_and_tan_coonhound':64, 'Great_Pyrenees':65,
'boxer':66, 'wire_haired_fox_terrier':67, 'borzoi':68, 'groenendael':69, 'collie':70, 'Gordon_setter':71, 'Kerry_blue_terrier':72,
'briard':73, 'Rhodesian_ridgeback':74, 'Boston_bull':75, 'bull_mastiff':76, 'silky_terrier':77, 'Brittany_spaniel':78,
'Eskimo_dog':79, 'giant_schnauzer':80, 'malinois':81, 'Bouvier_des_Flandres':82, 'whippet':83, 'Appenzeller':84,
'Chinese_Crested_Dog':85, 'soft_coated_wheaten_terrier':86, 'Weimaraner':87, 'clumber':88, 'Greater_Swiss_Mountain_dog':89,
'toy_terrier':90, 'Italian_greyhound':91, 'basset':92, 'basenji':93, 'Australian_Shepherd':94, 'Maltese_dog':95,
'Japanese_spaniel':96, 'Cane_Carso':97, 'Japanese_Spitzes':98, 'Old_English_sheepdog':99, 'Black_sable':100,
'Shetland_sheepdog':101, 'English_springer':102, 'beagle':103, 'cocker_spaniel':104, 'standard_poodle':105,
'komondor':106, 'chow':107, 'Yorkshire_terrier':108, 'Shih_Tzu':109, 'Chihuahua':110, 'Pekinese':111, 'miniature_pinscher':112,
'pug':113, 'papillon':114, 'Shiba_Dog':115, 'French_bulldog':116, 'Siberian_husky':117, 'malamute':118, 'Pomeranian':119,
'Samoyed':120, 'miniature_schnauzer':121, 'Border_collie':122, 'Cardigan':123, 'toy_poodle':124, 'Bichon_Frise':125,
'chinese_rural_dog':126, 'Labrador_retriever':127, 'golden_retriever':128, 'teddy':129, 'pad': -1,
}
idx2name = {v:k for k, v in name2idx.items()}

## Create Dataset and Dataloader for training

# annotation paths for training
annotation_path_all_train = []
path = 'data/Annotations'
with open('data/train_10_200.txt') as f:
    file_line = f.readline().rstrip('\n')
    while file_line:
        filename = os.path.join(path, file_line)+'.xml'
        annotation_path_all_train.append(filename)
        file_line = f.readline().rstrip('\n')  # read next line

# Create Dataset and Dataloader for training
dogs_dataset = ObjectDetectionDataset(annotation_path_all_train, image_dir, (img_height, img_width), name2idx)
# the size of training data set is 10*200, so choose batch_size=20
batch_size_train = 20
dogs_dataloader = DataLoader(dogs_dataset, batch_size=batch_size_train, drop_last=True)

## Display Images and Bounding Boxes of two training samples

# Grab the first batch for displaying and future use
for img_batch, get_bboxes_batch, get_classes_batch in dogs_dataloader:
    img_data_1st = img_batch
    get_bboxes_1st = get_bboxes_batch
    get_classes_1st = get_classes_batch
    break

# get class names
idxs = random.sample(range(0, batch_size_train), 2)
get_class_1 = get_classes_1st[idxs[0]]
get_class_1 = [idx2name[idx.item()] for idx in get_class_1]
get_class_2 = get_classes_1st[idxs[1]]
get_class_2 = [idx2name[idx.item()] for idx in get_class_2]

# Display Images and Bounding Boxes
nrows, ncols = (1, 2)
fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8))
fig, axes = display_img([img_data_1st[idxs[0]], img_data_1st[idxs[1]]], fig, axes)
fig, _ = display_bbox(get_bboxes_1st[idxs[0]], fig, axes[0], classes=get_class_1)
fig, _ = display_bbox(get_bboxes_1st[idxs[1]], fig, axes[1], classes=get_class_2)
plt.show()

## Convolutional Backbone Network
# We will use the first 4 layers of resnet50 as our convolutional backbone
model = torchvision.models.resnet50(pretrained=True)
req_layers = list(model.children())[:8]
backbone = nn.Sequential(*req_layers)
# unfreeze all the parameters
for param in backbone.named_parameters():
    param[1].requires_grad = True
# run the image through the backbone
out = backbone(img_data_1st)
out_c, out_h, out_w = out.size(dim=1), out.size(dim=2), out.size(dim=3)

# How much the image has been down-scaled
width_scale_factor = img_width // out_w
height_scale_factor = img_height // out_h
print(width_scale_factor, height_scale_factor)

# Building the model
img_size = (img_height, img_width)
out_size = (out_h, out_w)
n_classes = len(name2idx) - 1 # exclude pad idx
roi_size = (2, 2)
detector = TwoStageDetector(img_size, out_size, out_c, n_classes, roi_size)

## Model training
def training_loop(model, learning_rate, train_dataloader, n_epochs):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train() # Set the module in training mode
    loss_list = []
    for i in tqdm(range(n_epochs)):
        total_loss = 0
        # train the model by batch size iteratively
        for img_batch, get_bboxes_batch, get_classes_batch in train_dataloader:
            # forward pass
            loss = model(img_batch, get_bboxes_batch, get_classes_batch)
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        loss_list.append(total_loss)
    return loss_list

learning_rate = 1e-3
n_epochs = 100 # set as large as 1000 if your if your computational resources allow
loss_list = training_loop(detector, learning_rate, dogs_dataloader, n_epochs)

# Plot the loss curve
plt.plot(loss_list)
plt.show()

# save model
torch.save(detector, "model.pt")

# load model
detector = torch.load('model.pt')

## Model performance

# annotation path for testing
annotation_path_all_test = []
path = 'data/Annotations'
with open('data/test_10_40.txt') as f:
    file_line = f.readline().rstrip('\n')
    while file_line:
        filename = os.path.join(path, file_line)+'.xml'
        annotation_path_all_test.append(filename)
        file_line = f.readline().rstrip('\n')  # read next line

# Create Dataset and Dataloaders for testing
dogs_dataset_test = ObjectDetectionDataset(annotation_path_all_test, image_dir, (img_height, img_width), name2idx)
# the size of testing set is 10 * 40, so choose batch_size=20
batch_size_test = 20
dogs_dataloader_test = DataLoader(dogs_dataset_test, batch_size=batch_size_test, drop_last=True)

# Get the labels of all testing samples
classes_all_test = []
for img_batch, get_bboxes_batch, get_classes_batch in dogs_dataloader_test:
    classes_all_test.append(get_classes_batch.tolist())

# One sample one label
classes_all_test_1 = []
for i in range(len(classes_all_test)):
    for j in range(len(classes_all_test[i])):
        classes_all_test_1.append(classes_all_test[i][j][0])

# Get the predicted labels of all testing samples
classes_pred_all_test = []
detector.eval() # Set the module in evaluation mode
for img_batch, get_bboxes_batch, get_classes_batch in dogs_dataloader_test:
    proposals_final, conf_scores_final, classes_final = detector.inference(img_batch, conf_thresh=0.8, nms_thresh=0.05)
    classes_pred_all_test.append(classes_final)

# One sample one label
classes_pred_all_test_1 = []
for i in range(len(classes_pred_all_test)):
    for j in range(len(classes_pred_all_test[i])):
        classes_pred_all_test_1.append(classes_pred_all_test[i][j].tolist())
classes_pred_all_test_2 = []
for i in range(len(classes_pred_all_test_1)):
    classes_pred_all_test_2.append(classes_pred_all_test_1[i][0])

# Accuracy
average_accuracy = accuracy(classes_all_test_1, classes_pred_all_test_2)
print("Accuracy:", average_accuracy)

# Precision
precision = macro_precision(classes_all_test_1, classes_pred_all_test_2)
print("Macro-averaged Precision score:", precision)

# Recall
recall = macro_recall(classes_all_test_1, classes_pred_all_test_2)
print("Macro-averaged recall score:", recall)

# F1 score
f1 = macro_f1(classes_all_test_1, classes_pred_all_test_2)
print("Macro-averaged f1 score:", f1)

# AUC score
auc = roc_auc_score_multiclass(classes_all_test_1, classes_pred_all_test_2)
print("AUC score:", {idx2name.get(int(key)) : value for key, value in auc.items()})

# Confusion Matrix
classes_all_test_2 = []
classes_pred_all_test_3 = []
# random select 10 dog breeds
class_list = random.sample(range(115, 117), 2)
# extract the 10 breeds from the testing results
for class1, class2 in zip(classes_all_test_1, classes_pred_all_test_2):
    if class1 in class_list:
        classes_all_test_2.append(class1)
        classes_pred_all_test_3.append(class2)

plt.figure(figsize = (18,18))
sns.heatmap(metrics.confusion_matrix(classes_all_test_2, classes_pred_all_test_3), annot = True,
            xticklabels = [idx2name[idx.item()] for idx in np.unique(classes_all_test_2)],
            yticklabels = [idx2name[idx.item()] for idx in np.unique(classes_pred_all_test_3)], cmap = 'summer')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

## Display Images and Bounding Boxes of two testing samples

# annotation path for testing
annotation_path_all_test = []
path = 'data/Annotations'
with open('data/test_2.txt') as f:
    file_line = f.readline().rstrip('\n')
    while file_line:
        filename = os.path.join(path, file_line)+'.xml'
        annotation_path_all_test.append(filename)
        file_line = f.readline().rstrip('\n')  # read next line

# Create Dataset and Dataloaders for testing
dogs_dataset_test = ObjectDetectionDataset(annotation_path_all_test, image_dir, (img_height, img_width), name2idx)
batch_size_test = 2
dogs_dataloader_test = DataLoader(dogs_dataset_test, batch_size=batch_size_test, drop_last=True)

# Grab the first batch for displaying
for img_batch, get_bboxes_batch, get_classes_batch in dogs_dataloader_test:
    break

# Make inference
detector.eval()
proposals_final, conf_scores_final, classes_final = detector.inference(img_batch, conf_thresh=0.8, nms_thresh=0.05)

# project proposals to the image space
prop_proj_1 = project_bboxes(proposals_final[0], width_scale_factor, height_scale_factor, mode='a2p')
prop_proj_2 = project_bboxes(proposals_final[1], width_scale_factor, height_scale_factor, mode='a2p')
# get classes
classes_pred_1 = [idx2name[cls] for cls in classes_final[0].tolist()]
classes_pred_2 = [idx2name[cls] for cls in classes_final[1].tolist()]
# Display Images and Bounding Boxes
nrows, ncols = (1, 2)
fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8))
fig, axes = display_img(img_batch[:2], fig, axes)
fig1, _ = display_bbox(prop_proj_1, fig, axes[0], classes=classes_pred_1)
fig2, _ = display_bbox(prop_proj_2, fig, axes[1], classes=classes_pred_2)
plt.show()
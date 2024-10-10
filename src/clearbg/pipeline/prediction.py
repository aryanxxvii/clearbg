import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from clearbg.config.configuration import ConfigurationManager
from clearbg.constants import *
from clearbg.model.u2net import U2NET
import numpy as np
import clearbg.utils.utils as data_transforms
import matplotlib.pyplot as plt
from skimage import io
import torch.nn.functional as F
import os
from torchvision.transforms.functional import normalize


model = U2NET(3, 1)

config = ConfigurationManager()
model_eval_config = config.get_evaluation_config()
model_path = os.path.join(PROJECT_ROOT, model_eval_config.model_path)

model.load_state_dict(torch.load(model_path, map_location="cpu"))

model.eval()

def preprocess(image):
    label_3 = np.zeros(image.shape)
    label = np.zeros(label_3.shape[0:2])

    if 3 == len(label_3.shape):
        label = label_3[:, :, 0]
    elif 2 == len(label_3.shape):
        label = label_3

    if 3 == len(image.shape) and 2 == len(label.shape):
        label = label[:, :, np.newaxis]
    elif 2 == len(image.shape) and 2 == len(label.shape):
        image = image[:, :, np.newaxis]
        label = label[:, :, np.newaxis]

    transform = transforms.Compose([data_transforms.RescaleT(320), data_transforms.ToTensorLab(flag=0)])
    sample = transform({"imidx": np.array([0]), "image": image, "label": label})

    return sample


image_path =  "./tough.jpg" # Update with your image path
image = Image.open(image_path).convert("RGB")

img = preprocess(np.array(image))

input_size=[1024,1024]
im_path = 'tough.jpg'
result_path = "."

with torch.no_grad():
    im = io.imread(im_path)
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    im_shp=im.shape[0:2]
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2,0,1)
    im_tensor = F.upsample(torch.unsqueeze(im_tensor,0), input_size, mode="bilinear").type(torch.uint8)
    image = torch.divide(im_tensor,255.0)
    image = normalize(image,[0.5,0.5,0.5],[1.0,1.0,1.0])

    if torch.cuda.is_available():
        image=image.cuda()
    result=model(image)
    result=torch.squeeze(F.upsample(result[0][0],im_shp,mode='bilinear'),0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result-mi)/(ma-mi)
    result = result.numpy()

    output_mask = result[0]
    output_mask = (output_mask - output_mask.min()) / (output_mask.max() - output_mask.min()) * 255
    output_mask = output_mask.astype(np.uint8)
    output_image = Image.fromarray(output_mask)
    plt.imshow(output_image, cmap='gray')
    plt.axis('off')  # Hide axis
    plt.show()
import os, shutil, cv2,datetime, torch, torchvision.transforms, json, torchvision, copy, time, open_clip, sys,PIL.Image
import matplotlib.pyplot as plt
os.chdir('/home/klimenko/seg_materials/')
print(os.getcwd())
import pandas as pd
import numpy as np



import argparse

# Create the parser
parser = argparse.ArgumentParser(description='Your script description here.')
parser.add_argument('--cuda', type=int, help='CUDA device number')
args = parser.parse_args()
if args.cuda is not None:
    print(f'CUDA device selected: {args.cuda}')
else:
    print('No CUDA device selected.')
    
    
    
device = "cuda:"+str(args.cuda)
#from seg_utils import process_image, show_anns_reg,convert_to_classes
from groundingdino_utils_windows import GROUNDING_DINO_OUTPUT
import matplotlib.colors as mcolors
from PIL import Image
import time

import os,tqdm, shutil, cv2,datetime, time, torch, torchvision.transforms, json, torchvision, copy, time, open_clip, sys,PIL.Image
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image,ImageFile
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
from torch.utils.data import random_split
import torch.nn as nn
import pandas as pd
from collections import Counter
from mit_semseg.config import cfg
from mit_semseg.dataset import TestDataset
from mit_semseg.models import ModelBuilder, SegmentationModule
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


#########################
# DEFINE SAM AND SETTINGS
#########################

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)


mask_generator_2 = SamAutomaticMaskGenerator(
    model=sam,  # Your SAM model
    points_per_side=64,  # Increase points_per_side for finer segmentation
    pred_iou_thresh=0.75,  # Adjust prediction IOU threshold for stricter segmentation
    stability_score_thresh=0.75,  # Increase stability score threshold for more stable segmentation
    crop_n_layers=0,  # Keep the original image resolution
    crop_n_points_downscale_factor=0,  # Keep the original resolution for more details
)


def show_anns_reg(anns,color_list):
    img_sam = np.zeros([anns[0]['segmentation'].shape[0], anns[0]['segmentation'].shape[1],3])
    img_result = np.zeros([anns[0]['segmentation'].shape[0], anns[0]['segmentation'].shape[1],3])
    for index, ann in enumerate(anns):
        m = ann['segmentation']
        mask_expanded = np.repeat(np.expand_dims(m, axis=-1), 3, axis=-1)
        color = np.random.random(3)
        sub_img = mask_expanded * color
        sub_img_result = mask_expanded * color_list[index] 
        img_sam = img_sam+sub_img   
        img_result = img_result+sub_img_result
    return img_sam, img_result


####################################
# Hex2RGB conversion for LIBHSI
###################################

def hex_to_rgb(hex_color):
    """Convert hexadecimal color to RGB tuple."""
    # Remove '#' if present
    hex_color = hex_color.lstrip('#')
    # Convert hex to RGB
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b)



####################################
# ADE20K Segmentation to Extract Facade
###################################

net_encoder = ModelBuilder.build_encoder(arch='resnet50dilated',fc_dim=2048,weights='ckpt/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth')
net_decoder = ModelBuilder.build_decoder(arch='ppm_deepsup',fc_dim=2048,num_class=150,weights='ckpt/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth',use_softmax=True)
crit = torch.nn.NLLLoss(ignore_index=-1)
segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
segmentation_module.eval()
device = torch.device("cuda:1")
segmentation_module.to(device)
pil_to_tensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  ])


def segment_image(path, resolution_factor):
    pil_image = PIL.Image.open(path).convert('RGB')
    width, height = pil_image.size
    new_dim = (width // resolution_factor, height // resolution_factor)
    resized_pil_image = pil_image.resize(new_dim)
    img_data = pil_to_tensor(resized_pil_image)
    singleton_batch = {'img_data': img_data[None].to(device)}
    output_size = img_data.shape[1:]
    with torch.no_grad():
        scores = segmentation_module(singleton_batch, segSize=output_size)
    _, pred = torch.max(scores, dim=1)
    pred = pred.cpu()[0].numpy()
    ADEmask = np.where(np.logical_or(pred == 1, pred == 25), 1, 0) # removed ADE20K segmentation
    
    ADE_FRACTION = np.sum(ADEmask)/(ADEmask.shape[0]*ADEmask.shape[1])
    print('building frac:_',ADE_FRACTION)
    if ADE_FRACTION < 0.2:
        raise ValueError("not enough building there")
    
    empty_mask = 1 - ADEmask
    ADEmask = np.repeat(ADEmask[:, :, np.newaxis], 3, axis=2)
    
    
    
    
    result = resized_pil_image*ADEmask # removed ADE20K segmentation
    return result.astype(np.uint8), empty_mask.astype(bool)




# data = [['classname_full', 'classname_short', 'color'],
#         ["glass surface", 'glass', 'blue'],
#         ["concrete surface", 'concrete', 'yellow'],
#         ["brick wall", 'brick', 'red'],
#         ["siding wall", 'siding', 'green'],
#         ["roof tiles shingles", 'roof', 'orange'],
#         ["stone", 'stone', 'cyan'],
#         ["plaster facade surface", 'plaster', 'white'],
#         ["metal surface", 'metal', 'purple'],]

# Convert the list of lists to a DataFrame
#label_df = pd.DataFrame(data[1:], columns=data[0])



def add_strong_blur_to_mask(mask, max_blur_size=40, blur_strength=25):
    mask_uint8 = (mask * 255).astype(np.uint8)
    blur_size = min(max_blur_size, mask.shape[0] // 2 - 1)
    blur_size = min(blur_size * 2 + 1, mask.shape[0] // 2 - 1)
    blurred_mask = cv2.GaussianBlur(mask_uint8, (blur_size, blur_size), blur_strength)
    blurred_mask = blurred_mask / 255.0
    return np.minimum((blurred_mask+mask)**0.5, 1)


model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model_clipseg = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

def CLIPSEG_PREDICT(image, mask,label_df):
    image = Image.fromarray(image)
    
    prompts = list(label_df['classname_full'])
    labels_short = list(label_df['classname_short'])
    labels_long = list(label_df['classname_full'])
    
    preds_list = []
    scores_list =[]
    inputs = processor(text=prompts, images=[image] * len(prompts), padding="max_length", return_tensors="pt")
    with torch.no_grad():
      outputs = model_clipseg(**inputs)
    preds = outputs.logits.unsqueeze(1)
    for tensor in preds:
#         shifted_tensor = tensor - tensor.min()
#         sum_shifted_tensor = shifted_tensor.sum()
#         normalized_tensor = (shifted_tensor / sum_shifted_tensor)

        normalized_tensor=tensor
        preds_list.append(normalized_tensor[0])
        resized_mask = torch.tensor(zoom(mask, (352 / mask.shape[0], 352 / mask.shape[1]), order=1))
        scores_list.append(torch.sum((normalized_tensor[0])*resized_mask).item())

    #print(scores_list)
    index = scores_list.index(max(scores_list))
    answer = labels_short[index]
    print(labels_long[index])
    return answer, index


def clip_classify(image, label_df):
    pil_image = Image.fromarray(image)
    labels_long = list(label_df['classname_full'])
    image = preprocess(pil_image).unsqueeze(0)

    labels = list(label_df['classname_full'])
    labels_short = list(label_df['classname_short'])
    text = tokenizer(labels)
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    

    index = torch.argmax(text_probs).item()
    answer = labels_short[index]
   
    predicted_label_index = torch.argmax(text_probs)
    predicted_label = labels[predicted_label_index]
    confidence_percentage = text_probs[0, predicted_label_index].item() * 100
    #print(f"Label: {predicted_label} | {confidence_percentage:.2f}%" )
    #display
    print(labels_long[predicted_label_index])
    return answer, pil_image

def produce_list_of_elements(masks,pred,pure_image, label_df, zoom_factor):
    out_list = []
    out_list_fragments=[]
    out_list_masks=[]
    color_list=[]
    color_list_ac=[]
    for seg in masks:
        #print("|", end="")
        segmask = seg['segmentation']#add_strong_blur_to_mask(seg['segmentation'])
        #seg['segmentation']#add_strong_blur_to_mask(
        segmask_blur = seg['segmentation']#add_strong_blur_to_mask(seg['segmentation'],max_blur_size=40, blur_strength=25)
        segmask = zoom(segmask, (zoom_factor,zoom_factor), order=0)
        instance =pure_image#
        instance_clip = (pred*np.repeat(segmask_blur[:, :, np.newaxis], 3, axis=2)).astype(np.uint8)#pure_image
        indices = np.argwhere(segmask)

        # Get bounding box coordinates
        min_row, min_col = np.min(indices, axis=0)
        max_row, max_col = np.max(indices, axis=0)

        # Crop the region of interest
        factor = 1.3 #1.5
        instance = instance[max(0, (min_row + max_row) // 2 - int(factor * (max_row - min_row + 1) // 2)):min(instance.shape[0] - 1, (min_row + max_row) // 2 + int(factor * (max_row - min_row + 1) // 2)) + 1,
                            max(0, (min_col + max_col) // 2 - int(factor * (max_col - min_col + 1) // 2)):min(instance.shape[1] - 1, (min_col + max_col) // 2 + int(factor * (max_col - min_col + 1) // 2)) + 1, :]
#         instance_clip = instance_clip[max(0, (min_row + max_row) // 2 - int(factor * (max_row - min_row + 1) // 2)):min(instance.shape[0] - 1, (min_row + max_row) // 2 + int(factor * (max_row - min_row + 1) // 2)) + 1,
#                             max(0, (min_col + max_col) // 2 - int(factor * (max_col - min_col + 1) // 2)):min(instance.shape[1] - 1, (min_col + max_col) // 2 + int(factor * (max_col - min_col + 1) // 2)) + 1, :]
        instance_mask = segmask[max(0, (min_row + max_row) // 2 - int(factor * (max_row - min_row + 1) // 2)):min(segmask.shape[0] - 1, (min_row + max_row) // 2 + int(factor * (max_row - min_row + 1) // 2)) + 1,
                            max(0, (min_col + max_col) // 2 - int(factor * (max_col - min_col + 1) // 2)):min(segmask.shape[1] - 1, (min_col + max_col) // 2 + int(factor * (max_col - min_col + 1) // 2)) + 1]

#         out_list.append(instance)
#         out_list_fragments.append(instance_clip)
#         out_list_masks.append(instance_mask)
        print(instance.shape, end=" ")
        #a, i = clip_classify(instance_clip, label_df)
        #acs, aci = CLIPSEG_PREDICT(instance, instance_mask,label_df)
        acs, aci = clip_classify(instance_clip, label_df)
        a,i =acs, aci
        color = label_df.loc[label_df['classname_short'] == a, 'color'].values[0]
        color_ac = label_df.loc[label_df['classname_short'] == acs, 'color'].values[0]
        rgb = list((np.asarray(mcolors.to_rgb(color))*255))
        rgb_ac = list((np.asarray(mcolors.to_rgb(color_ac))*255))
        
        color_list.append(rgb)
        color_list_ac.append(rgb_ac)
#         plt.imshow(instance)
        
    return out_list,out_list_fragments,out_list_masks, color_list, color_list_ac





def process_image(PATH, label_df, zoom_factor, downscale_factor):

    start_time = time.time()



    print(PATH, '_________________')
#     zoom_factor = 10
#     downscale_factor = 1
    pred, empty_mask = segment_image(PATH, int(zoom_factor/downscale_factor))
    
    pure_image = cv2.imread(PATH)
    #pure_image = zoom(pure_image, (1,1,1), order=0)
    pure_image_for_masking = cv2.resize(pure_image, (pred.shape[1], pred.shape[0]))
    print(pure_image_for_masking.shape)
    cv2.imwrite('pure_image_for_masking.jpg', pure_image_for_masking)
    pure_image = cv2.cvtColor(pure_image, cv2.COLOR_RGB2BGR)
    masks = mask_generator_2.generate(pure_image_for_masking)
    size_threshold = pure_image.shape[0]*pure_image.shape[1]/100/(zoom_factor*zoom_factor)#200 75
    size_upper = pure_image.shape[0]*pure_image.shape[1]/(zoom_factor*zoom_factor)*0.95
    masks = [mask for mask in masks if mask['area'] >= size_threshold and mask['area'] <= size_upper]
    
    
    masks_pre = []
    for mask1 in masks:
        print('-', end="")
        intersection = np.sum(np.logical_and(mask1['segmentation'], empty_mask))
        if intersection < np.sum(mask1['segmentation'])/4:
            masks_pre.append(mask1)
    
    print(len(masks), len(masks_pre))
    print('')
    masks_non_nested = []
    
    
    

    for mask1 in masks_pre:
           
            print("|", end="")
            overlap = False
            for mask2 in masks:
                if mask1['area'] > mask2['area']:
                    #overlap_mask = mask1['segmentation']*1-mask2['segmentation']*1
                    overlap_mask = np.logical_and(mask1['segmentation'], np.logical_not(mask2['segmentation']))
                    #overlap_mask = (overlap_mask == 1).astype(int)
                    overlap_mask_area = np.sum(overlap_mask)
                    mask1['segmentation'] = overlap_mask
                    mask1['area'] = overlap_mask_area

            if mask1['area']>size_threshold:
                masks_non_nested.append(mask1) 
    print('')
    print(len(masks_pre), len(masks_non_nested))
    del masks_pre
    del masks
    out_list,out_list_fragments,out_list_masks, color_list, color_list_ac =produce_list_of_elements(masks_non_nested,pred,pure_image, label_df, zoom_factor)
    end_time = time.time()
    print("Processed in ", end_time - start_time, "s")
    return pred, pure_image, masks_non_nested, out_list,out_list_fragments,out_list_masks, color_list, color_list_ac




def convert_to_classes(result,image_array, label_df_DINO,label_df):
    class_array = np.zeros_like(result[:,:,0])
    for i in range(len(label_df_DINO)):
        mask = np.all(image_array == np.asarray(label_df_DINO['color'][i]), axis=-1)
        class_array[mask]=label_df_DINO['class_number'][i]
        print('CONVERT ',np.sum(class_array))

    for i in range(len(label_df)):
     
        #mask = np.all(result == np.asarray(hex_to_rgb(label_df['color'][i])), axis=-1)
        mask = np.all(result == np.asarray(mcolors.to_rgb(label_df['color'][i]))*255, axis=-1)
        class_array[mask]=label_df['class_number'][i]
   
        
    return class_array



data = [['classname_full', 'classname_short', 'color','class_number'],
        ['glazing', 'glazing','blue',10],
        ["exposed concrete", 'concrete', 'yellow',11],
        ["brick", 'brick', 'red',12],
        ["timber siding", 'siding', 'green',13], #timber siding facade
        ["siding", 'siding', 'green',13],
        ["masonry", 'stone', 'red',12],
        ["marble", 'stone', 'red',12],
#         ["smooth painted facade surface", 'stucco', 'white',16],
        ["stucco", 'stucco', 'white',16], #facade element
        ["plaster", 'stucco', 'white',16],#facade
        ["roof tiles", 'metal', 'purple',17],
        ["metal panels", 'metal', 'purple',17],
       ["metal panels cladding", 'metal', 'purple',17],]
label_df = pd.DataFrame(data[1:], columns=data[0])

data_DINO = [['classname', 'color','class_number'],
                ["window",  (0,0,255),10],
                ["door",(210,105,30),32],]
label_df_DINO = pd.DataFrame(data_DINO[1:], columns=data_DINO[0])



# DIR = '/home/klimenko/seg_materials/MAPS/images/ams/AMS_RGB/'
# DIR_ACTUAL = '/home/klimenko/seg_materials/MAPS/images/ams/AMS_RGB/'
# OUT_DIR = '/home/klimenko/seg_materials/MAPS/images/ams/OUTPUTS/'
DIR = '/home/klimenko/seg_materials/FINAL_DATASETS/AMSTERDAM/RGB/'
DIR_ACTUAL = DIR
OUT_DIR = '/home/klimenko/seg_materials/FINAL_DATASETS/AMSTERDAM/NEW_OUTPUTS5/'




ooo=0
for FILENAME in os.listdir(DIR):
    ooo = ooo+1
    print(ooo)
    
    PATH = DIR_ACTUAL + FILENAME
    print(PATH)
    if not PATH.endswith('.DS_Store'):# and FILENAME.startswith('180')
        if FILENAME  not in os.listdir(OUT_DIR+'COMBINED/') and FILENAME  not in os.listdir(OUT_DIR+'RGB/'):
            print(PATH)
            try:
                print(PATH)
                image = Image.open(PATH)
                image = image.convert("RGB")
                image2 = cv2.cvtColor(np.asarray(image).astype('uint8'), cv2.COLOR_BGR2RGB)
                cv2.imwrite(OUT_DIR+'RGB/'+FILENAME[:-4]+'.jpg', np.asarray(image2))
                
                pred, pure_image, masks, out_list,out_list_fragments,out_list_masks, color_list, color_list_ac = process_image(PATH, label_df,1,1)   
                start_time = time.time()
                image = image.resize((image.width // 1, image.height // 1))
                try:
                    image_array = GROUNDING_DINO_OUTPUT(image)
                    image_array = cv2.resize(image_array, (pred.shape[1],pred.shape[0]))
                except:
                    print('GDINO DID NOT WORK')
                    image_array = np.zeros_like(pred)
                end_time = time.time()
                elapsed_time = end_time - start_time
                print("Elapsed time:", elapsed_time, "seconds")

                
                pred_all = cv2.resize(pure_image, (pred.shape[1], pred.shape[0]))
                sam,result = show_anns_reg(masks,color_list)
                mask = np.all(pred == [0, 0, 0], axis=-1)
                image_array_mask = np.clip(image_array,0,1)
                image_array_mask = np.sum(image_array_mask, axis=2)
                image_array_mask = np.clip(image_array_mask,0,1)
                image_array_mask=1-np.repeat(image_array_mask[:, :, np.newaxis], 3, axis=2)

                sam_ac,result_ac = show_anns_reg(masks,color_list_ac)
                sam[mask]=0
                result[mask]=0
                result_ac[mask]=0
                sam_ac[image_array_mask]=0
                result = result*image_array_mask
                result_ac=result_ac*image_array_mask

                result_display = result.astype(float) + image_array.astype(float)/255
                result_ac_display = result_ac.astype(float)+ image_array.astype(float)#/255
                result_display = np.clip(result_display,0,255)
                result_ac_display = np.clip(result_ac_display,0,255)
                
                
                classes_result_ac = convert_to_classes(result_ac_display, image_array, label_df_DINO,label_df)
                np.save(OUT_DIR+'CLIPSEG/'+FILENAME[:-4]+'.npy', classes_result_ac)
                pure_image = cv2.cvtColor(pure_image, cv2.COLOR_BGR2RGB)
                result_save = cv2.cvtColor(result_display.astype('uint8'), cv2.COLOR_BGR2RGB)
                image_array_save = cv2.cvtColor(image_array.astype('uint8'), cv2.COLOR_BGR2RGB)
                cv2.imwrite(OUT_DIR+'COMBINED/'+FILENAME[:-4]+'.jpg', pure_image*0.65+result_save*0.35+image_array_save)
                cv2.imwrite(OUT_DIR+'SAM/'+FILENAME[:-4]+'.jpg', pure_image*0.55+sam*255*0.35)
                cv2.imwrite(OUT_DIR+'PRED/'+FILENAME[:-4]+'.jpg', (1-mask)*255)


            except Exception as e:
                print('Error:', e)
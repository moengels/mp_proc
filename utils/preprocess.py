import math
import time
import yaml
from skimage import util, exposure, feature, filters, morphology, measure, io, transform
from scipy import stats, ndimage
from tkinter import filedialog
from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
import os
import tifffile
from tqdm import tqdm

#%% functions
#adaption of chat gpt suggestion for iterative rotation, without hough transform, just projective sum and maxing out edges 
# stack: z,x,y 
def upright_image(stack, log = False):
    angles = np.linspace(-3, 3, 30)
    max_upright_pixels = 0
    best_angle = 0
    minp = np.min(stack, axis=0)
    min_dim = np.argmin(minp.shape)
    for angle in angles:
        # Rotate the image
        rotated_img = transform.rotate(minp, angle, resize=False, mode="wrap", preserve_range=True)
        # Add speckle noise
        noisy_img = util.random_noise(rotated_img, mode='speckle', mean=0.2)
        # Gaussian smoothing
        smoothed_img = filters.gaussian(rotated_img, sigma=10, preserve_range=True)
        # Canny edge detection
        edges = feature.canny(smoothed_img, sigma=1)

        lines = np.ones_like(smoothed_img)*edges
        img_projection = np.sum(lines, axis=min_dim)
        img_projection = img_projection[2: -3]
        
        upright_pixels = np.max(img_projection)
        print(upright_pixels)
        if upright_pixels > max_upright_pixels:
            max_upright_pixels = upright_pixels
            best_angle = angle
    if log: 
        # Print the best angle and the number of lines detected
        print(f"Best Angle: {best_angle} degrees")
        print(f"Max Line Count: {max_upright_pixels}")

        # Plot the results
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1), plt.imshow(minp, cmap="gray"), plt.title("Original Image")
        plt.subplot(1, 2, 2), plt.imshow(transform.rotate(minp, best_angle, resize=True, mode="wrap"), cmap="gray"), plt.title("Best Detected Angle")
        plt.show()

    return transform.rotate(minp, best_angle, resize=False, mode="wrap", preserve_range=True)


def most_common_value(img):
    # Flatten image to 1D array
    flat_img = img.flatten()
    
    # Calculate histogram of image
    hist, bins = np.histogram(flat_img, bins=range(int(np.min(flat_img)), int(np.max(flat_img))+2))
    
    # Find index of most frequent value
    idx = np.argmax(hist)
    
    # Return most frequent value
    return bins[idx]



def segmentPlanes_brightfield(stack):
    N = stack.shape
    plane = stack[0,:,:]
    plane = filters.gaussian(plane, sigma = 3, preserve_range=True)
    #th = filters.threshold_triangle(plane)
    th =  most_common_value(plane)/10
    binary = plane > th
    mask = np.logical_and(np.ones(plane.shape), binary > 0).astype(int)

    morph = morphology.dilation(mask) 
    #segm = segmentation.clear_border(morph)
    label_image = measure.label(morph)

    bounding_box = []
    rotation_info = []
    for region in measure.regionprops(label_image):
        if region.area_bbox >= (N[1]*N[2])/2: #split in the middle if detected area too big
            minr, minc, maxr, maxc = region.bbox
            bounding_box.append([minr, minc, maxr, int((minc+(maxc-minc)/2))])
            bounding_box.append([minr, int((minc+(maxc-minc)/2)), maxr,maxc])
            rotation_info.append([(region.centroid[0], region.centroid[1]-(maxc-minc)/4), math.degrees(region.orientation)])
            rotation_info.append([(region.centroid[0] + region.centroid[1]+(maxc-minc)/4), math.degrees(region.orientation)])
        elif region.area_bbox >= (N[1]*N[2])/6: 
            bounding_box.append(region.bbox)
            rotation_info.append([region.centroid, math.degrees(region.orientation)])
    
    subimage = []
    for bb, rot in zip(bounding_box, rotation_info):
        subimage.append(stack[:,bb[0]:bb[2],bb[1]:bb[3]])
        if rot[1] != 0:
            subimage[-1] = upright_image(subimage[-1], *rot)

    return subimage, bounding_box



def segmentPlanes_fluorescence(stack, sigma_gauss = 5, plane = 0, equal_flag = False):
    #fig, axs = plt.subplots(1, 4)
    #image = stack[plane,:,:].squeeze()
    image = np.max(stack, axis=0)
    if equal_flag:
        image = exposure.adjust_log(image, 10)
    
    # smooth image to not segment single beads (maybe nieed even bigger kernel, but out of focus plane shoudk account for that
    smoothed = filters.gaussian(image, sigma = sigma_gauss, preserve_range=True)
    #axs[0].imshow(smoothed)
    # detect separating edge
    edges = feature.canny(smoothed, sigma=30)
    edges = morphology.binary_dilation(morphology.binary_dilation(edges)) # dilate twice to be sure edges connect and separate the subimages
    #axs[1].imshow(edges)
    # set seeds for flood fill (separation is assumed to be along the vertical axis
    seed1 = (int(edges.shape[0]/2), int(edges.shape[1]/3))
    seed2 = (int(edges.shape[0]/2), int(2*edges.shape[1]/3))

    subimage1, bbox1, c1, a1 = cropPlane(stack, edges, seed1)
    if a1 != 0:
        #subimage1 = upright_image(subimage1, c1, a1)
        subimage1 = upright_image(subimage1)
    #axs[2].imshow(subimage1[0,:,:])
    subimage2, bbox2, c2, a2 = cropPlane(stack, edges, seed2)
    if a2 != 0:
        #subimage2 = upright_image(subimage2, c2, a2)
        subimage2 = upright_image(subimage2)
    #axs[3].imshow(subimage2[0,:,:])

    return [subimage1, subimage2], [bbox1, bbox2]
    
    
def cropPlane(stack, edges, seed):
    # flood fill to find left panel 
    subimage = morphology.flood_fill(edges, seed, 2, connectivity=1)
    #convert to label to extract bounding box
    #subimage = measure.label(subimage)
    #plt.imshow(subimage)
    subimage = subimage.astype(np.uint8)
    roi = measure.regionprops(subimage)
    bb = roi[0].bbox
    centroid = roi[0].centroid
    angle = math.degrees(roi[0].orientation)
    # mask original image to remove background
    mask = subimage == 1
    bin_stack = stack * np.stack([mask]*stack.shape[0], axis=0)
    
    #crop along bounding box with background set to 0 
    stack_left = bin_stack[:,bb[0]:bb[2],bb[1]:bb[3]]
    
    return stack_left, bb, centroid, angle, 


def crop_square_rois_4D(stack):
    N = stack.shape

    if N[2] == N[3]:
        return stack
    else:
        max_dim = 2+np.argmax(N[2:])
        min_dim = 2+np.argmin(N[2:])

        ratio = N[max_dim]/N[min_dim]
        
        n_subimages = int(np.ceil(ratio))
        start_indices_float = np.linspace(0,N[max_dim]-N[min_dim], n_subimages)
        start_indices = [int(index) for index in start_indices_float]
        output = []
        for roi in start_indices:
            if max_dim == 2:
                output.append(stack[:,:,roi:roi+N[3],:])
            else:
                output.append(stack[:,:,:,roi:roi+N[2]])
        
        return output 

def crop_square_rois(stack):
    N = stack.shape

    if N[1] == N[2]:
        return stack

    max_dim = 1+np.argmax(N[1:])
    min_dim = 1+np.argmin(N[1:])

    ratio = N[max_dim]/N[min_dim]
    
    n_subimages = int(np.ceil(ratio))
    start_indices_float = np.linspace(0,N[max_dim]-N[min_dim], n_subimages)
    start_indices = [int(index) for index in start_indices_float]
    output = []
    for roi in start_indices:
        if max_dim == 1:
            output.append(stack[:,roi:roi+N[2],:])
        else:
            output.append(stack[:,:,roi:roi+N[1]])
    
    return output
            

def write_images_rois(rois, output_path, fileID, filetype = "tif"):
    makeFolder(output_path)
    for idx, img in enumerate(rois):
        filename = f"{fileID}_roi{idx:02d}.{filetype}" 
        tifffile.imwrite(os.path.join(output_path, filename), img)#,
        #resolution=(1./0.108, 1./0.108),
        #metadata={'axes': 'ZXY'})

def write_logs(path, info):
    makeFolder(path)
    with open(os.path.join(path, f'log_{time.strftime("%Y%m%d_%H%M")}.txt'), "w") as f:
        for key, value in info.items():
            f.write(str(key) + ": " + str(value) + "\n")


def splitStackatChannel(stack):
    if len(stack.shape) == 4:
        return [np.squeeze(stack[:,0,:,:]), np.squeeze(stack[:,1,:,:])]
    else:
        return [np.squeeze(stack[0::2,:,:]), np.squeeze(stack[1::2,:,:])]

def load_filelist_from_directory(path=None, ):
    try: 
        os.path.exists(path)
    except:
        root = Tk()
        root.withdraw()
        path = filedialog.askdirectory()
            
    outpath = os.path.join(path, "rois")
    if not os.path.exists(outpath):
        os.mkdir(outpath)
        print("created folder : ", outpath)

    outpath = os.path.join(path, "planes")
    if not os.path.exists(outpath):
        os.mkdir(outpath)
        print("created folder : ", outpath)

    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and (f.endswith("tiff") or f.endswith("tif"))], path



def load_spacing_data(path=None):
    file = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and (f.endswith("yml") or f.endswith("yaml"))]
    with open(os.path.join(path, file[0])) as f:
        data = yaml.load(f, Loader=yaml.loader.SafeLoader)
    return data



# based on skewness of histogram 
def isFluorescenceImage(image):
    histogram, bins = np.histogram(image.flatten(), bins=256)
    hist_skewness = stats.skew(histogram)

    if hist_skewness <= 0:
        return True
    else: 
        return False

# registration leaves exagerated frame at the edge of the FOV and non-illuminated area
# since the border is anyway not very useful when processing phase data, we crop ~ 5% of the image on each side  
# pxl is cropped pixel from the repsective boundary: (left, right, top, bottom)
# for single plane (3D, t,x,y) 
def remove_aperture_frame(stack, pxl = (20,20,20,20)):
    _, c, r = stack.shape
    return stack[:,int(pxl[0]):c-int(pxl[1]), int(pxl[2]):r-int(pxl[3])]


# for whole stack
def crop_registration_masks(stack, pxl =(20,20,20,20)):
    z,t,x,y = stack.shape
    return stack[:,:,int(pxl[0]):x-int(pxl[1]), int(pxl[2]):y-int(pxl[3])]


def write_registered_stack(data, output_path, fileID, filetype = "tif"):
    makeFolder(output_path)
    filename = f"registered_{fileID}.{filetype}" 
    image_labels = [f'{i}' for i in range(data.shape[0] * data.shape[1])]
    tifffile.imwrite(os.path.join(output_path, filename),
                     np.transpose(data, (1,0,2,3)),
                     imagej=True,
                     resolution=(1./0.108, 1./0.108),
                     metadata={
                         'spacing': 620,
                         'unit': 'nm',
                         'axes': 'TZYX',
                         'Labels': image_labels})
    

def write_registered_stack_rois(rois, output_path, fileID, filetype = "tif"):
    makeFolder(output_path)
    for idx, img in enumerate(rois):
        fileID_roi= f"{fileID}_roi{idx:02d}" 
        write_registered_stack(img, output_path, fileID_roi, filetype)

def makeFolder(newpath):
    if not os.path.exists(newpath):
        os.makedirs(newpath)
        print("created folder: ", newpath)

def writeSinglePlaneOutput(images, basepath, filename):
    for idx, i in tqdm(enumerate(images)):
        plane = np.squeeze(i)
        rois = crop_square_rois(plane)
        fileID = filename.split('.')[0] + f"_p{idx:02d}"
        roipath = os.path.join(basepath, 'rois')
        makeFolder(roipath)
        write_images_rois(rois, roipath, fileID, filetype = "tif")

        subimage_path = os.path.join(basepath, 'planes')
        makeFolder(subimage_path)
        fileID += ".tif"
        tifffile.imwrite(os.path.join(subimage_path, fileID), plane)


def writeStackOutput(images, basepath, filename):
    stack_path = os.path.join(basepath, "stack")
    makeFolder(stack_path)
    fileID = filename.split('.')[0]
    ROIS = crop_square_rois_4D(images)
    write_registered_stack_rois(ROIS, stack_path, fileID, filetype = "tif")
    write_registered_stack(images, stack_path, fileID, filetype = "tif")



# function that splits the stack at a vertical line (axis = 1) at pixel 
# output is kept consistent for other bounding box based imag eplane segmentations  
# bounding box bb: [h0,v0,h1,v1] bb[0]:bb[2],bb[1]:bb[3]]
def split_stack_vertical(stack, pixel=None):
    if pixel is None:
        pixel = int(stack.shape[2]/2)
    subimage1 = stack[:,:,:pixel]
    subimage2 = stack[:,:,pixel:]

    bbox1 = [0,0,pixel,stack.shape[2]]
    bbox2 = [pixel,0,stack.shape[1],stack.shape[2]]

    return [subimage1, subimage2], [bbox1, bbox2]
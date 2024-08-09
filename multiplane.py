import skimage as skim 
import scipy as scp
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import tifffile
import yaml 
from tkinter import filedialog
from tkinter import *
import json

from utils.qpmain import qpmain
from utils.phase_structure import phase_structure
import utils.metadata as meta 



def get_fileID(filename):
    img_id = os.path.splitext(filename)[0]
    img_id = img_id.replace('_metadata', '')
    img_id = img_id.replace('.ome', '')
    return img_id 

def makeFolder(newpath):
    if not os.path.exists(newpath):
        os.makedirs(newpath)
        print("created folder: ", newpath)


class MultiplaneProcess:

    P = {}
    P['NA_ill']=0.27
    P['dz']=620 #nm
    P['dt']=30 #ms, default
    P['dF_batch']=1000 #frames, framebatch_size default
    P['do_phase']=False #bool whether to calculate phase from brightfield
    P['do_preproc']=True #bool whether to do preprocessing (FOV detection, cal estimation etc)
    P['ncams']=2 # remove pixels 
    P['dpixel']=7 # remove pixels from frame to remove registration artifacts
    P['order_default']= [2,3,0,1] # default order of planes after cropping
    P['flip_cam'] = [False, True] # bool, whether to flip the camera data (assuming there are 2 cameras)

    file_extensions = [".tif", ".tiff"]
    log = False


    def __init__(self):

        self.filenames = []
        self.metadata_files = {}
        self.output_path = None
        self.cal_path = None
        self.path = None
        self.meta = {}
        self.cal = {}
        self.is_bead = False
        #self.path = self.select_data_directory()




    def select_data_directory(self, path = None):
        if path is not None: 
            self.path = path   

        if self.path is None or not os.path.exists(self.path):
            root = Tk()
            root.withdraw()
            self.path = filedialog.askdirectory(title='select data directory')
        #else: 
        #    self.path = path       

        return self.path
    

    def set_logging(self, log):
        self.log = log
    

    def load_calibration(self):
        if not os.path.exists(os.path.join(self.path, 'calib.json')):
            # ask for user input for calibraiton
            root = Tk()
            root.withdraw()
            filepath = filedialog.askopenfile(title='Select calibration data file', filetypes =['*.json'])

        else:
            filepath = os.path.join(self.path, 'calib.json')

        try: 
            self.cal = json.load(filepath) 
        except:
            print("Something went wrong with loading the calibration file, skipping the process") 

        self.check_calibration() 

        return self.cal
        
    def check_calibration(self):
        # check if calibration file contains all relevant information 
        if type(self.cal) is not dict: 
            print(f"Calibration not a dictionary but {type(self.cal)} check inputs")
            return 
        
        if not bool(self.cal): 
            print(f"Calibration file is empty")
        
        validate_keys = ['fovs', 'brightness', 'dz', 'transform', 'order', 'deg']
        missing_keys = []        
        for k in validate_keys: 
            if k not in self.cal.keys():
                missing_keys.append(k)

        if not missing_keys: 
            print(f"No fields missing in calibration, proceeding")
        else:  
            print('Keys missing: ')
            print(', '.join(map(str, missing_keys))) 
            print("Initiating calibration: ")
            self.calibrate()


    def load_data(self):
        return skim.io.imread(os.path.join(self.path, self.filenames[0]))
    
    def create_cal_path(self):
        self.cal_path = os.path.join(self.path, 'cal_data')
        makeFolder(self.cal_path)
                                

    def calibrate(self, is_bead = False): 
        # do calibration on path data (if no bead data) 
        # if not is_bead: is only approximation but better than nothing
        self.is_bead = is_bead
        # load first dataset (4gb chunk)
        
        # check whether filelist has been filled 
        if not self.filenames:
            self.get_files_with_metadata()

        image = skim.io.imread(os.path.join(self.path, self.filenames[0]))
        print(f"Read image {self.filenames[0]}; size {image.shape}; type {image.dtype}")
        N_img = image.shape   


        fovs, self.cal['fovs'], self.cal['deg'] = self.adaptiveThreshold(image) 

        #file_convert = fovs.astype(np.uint16)

        if is_bead: 
            # figure out plane order otherwise take default order
            self.cal['dz'], self.cal['order'] = self.estimate_interplane_distance(fovs)
            #self.cal['order'] = self.get_plane_order(fovs)
        else: 
            self.cal['order'] = self.P['order_default'] 
            self.cal['dz'] = self.P['dz']
            

        print(f"Using order {self.cal['order']}")
        fps = np.ones(N_img[0])*(N_img[1]/2)
        fps = fps.astype(np.uint16)
        #fovs = fovs[self.cal['order'][::-1],:,:,:]

        if 'brightness' not in self.cal.keys():
            self.cal['brightness'] = self.estimate_brightess_from_stack(fovs[self.cal['order'][::-1],:,:,:]) 

        if 'transform' not in self.cal.keys():
            self.cal['transform'] = self.get_average_transform_via_xcorr(fovs[self.cal['order'][::-1],:,:,:], fps)

        print("Registration of data...")
        registered_subimages = self.transform_stack(fovs[self.cal['order'][::-1],:,:,:], self.cal['transform'])

        registered_subimages = np.clip(registered_subimages, 0, 2**16-1).astype(np.uint16)
        if len(registered_subimages.shape) == 4:
            axes = 'ZTYX'
        else:
            # axes = 'ZCTYX'
            axes = 'CTZYX'

  

        if self.log:
            self.create_cal_path()
            tifffile.imwrite(os.path.join(calpath, self.filenames[0]), registered_subimages, 
                        metadata={
                            'axes': axes,
                            'TimeIncrement': self.P['dt'],
                            'ZSpacing': self.P['dz']
                        }
                    ) 

        self.write_calibration()

        return self.cal 

    def get_plane_order(self, stack):
        return stack
    
    def write_calibration(self):
        #makeFolder(path)
        with open(os.path.join(self.path,'cal'), 'w') as yaml_file:
            #yaml.dump(self.cal, yaml_file, default_flow_style=False)
            json.dump(self.cal, yaml_file, cls=NumpyEncoder)


    def get_files_with_metadata(self):
        for file in os.listdir(self.path):
            # check only text files
            for ext in self.file_extensions:
                if file.endswith(ext):
                    self.filenames.append(file)
            if file.endswith('_metadata.txt'): 
                #check if metadata file is present and if so if it has an associated image file
                imagefile_specifier = get_fileID(file)
                if f'{imagefile_specifier}.ome.tif' in os.listdir(self.path):
                    self.meta[f'{imagefile_specifier}'] = {'file': file}

    def parse_metafile(self, filename):
        info = meta.openMetadata(os.path.join(self.path,self.meta[filename]['file']))
        header = meta.getHeader(info)
        return header 
    


    def get_metadata(self):
        for k in self.meta.keys():
            self.meta[k] = self.parse_metafile(k)
        return self.meta
    
    
    def estimate_brightess_from_stack(self, stack):
    # stack: z, t, y, x
        z, _,_,_ = stack.shape
        average_brightness = np.empty(shape=(z))
        for i in range(z):
            average_brightness[i] = np.mean(stack[i,::20,:,:].squeeze())
        brightness_factors = [b/np.mean(average_brightness) for b in average_brightness]
        b = {i: k for i,k in enumerate(brightness_factors)}
        return b 
    
    def upright_images(self, stack, P=None, log=False):
        datatype = stack.dtype
        angles = np.linspace(-3, 3, 31)
        max_upright_pixels = 0
        best_angle = 0
        minp = np.min(stack, axis=0)
        min_dim = np.argmin(minp.shape)
        if P is None:
            print("Determine skew angle...")
            for angle in tqdm(angles):
                # Rotate the image
                rotated_img = skim.transform.rotate(minp, angle, resize=False, mode="wrap", preserve_range=True)
                # Gaussian smoothing
                smoothed_img = skim.filters.gaussian(rotated_img, sigma=10, preserve_range=True)
                # Canny edge detection
                edges = skim.feature.canny(smoothed_img, sigma=1)

                lines = np.ones_like(smoothed_img) * edges
                img_projection = np.sum(lines, axis=min_dim)
                img_projection = img_projection[2: -3]

                upright_pixels = np.max(img_projection)
                if upright_pixels > max_upright_pixels:
                    max_upright_pixels = upright_pixels
                    best_angle = angle
            if self.log: 
                print(f"Best Angle: {best_angle} degrees")
                print(f"Max Line Count: {max_upright_pixels}")

                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1), plt.imshow(minp, cmap="gray"), plt.title("Original Image")
                plt.subplot(1, 2, 1), plt.imshow(minp, cmap="gray"), plt.title("Original Image")
                plt.subplot(1, 2, 2), plt.imshow(skim.transform.rotate(minp, best_angle, resize=True, mode="wrap"), cmap="gray"), plt.title("Best Detected Angle")
                plt.show()
        else:
            best_angle = P

        print("Rotating by skew angle...")
        limit = [-0.1, 0.1]
        if limit[0] < best_angle < limit[1]:
            rotated_stack = stack
        else:
            rotated_stack = self.rotate_stack(stack, best_angle, datatype)

        return rotated_stack, best_angle

    def rotate_stack(self, stack, best_angle, datatype):
        return scp.ndimage.rotate(stack, best_angle, axes=(1, 2), reshape=False, output=datatype, mode="wrap")

    def adaptiveThreshold(self, stack, n_planes=4, z_axis=0, camera_axis=1, size_estimate=None):
        flip = self.P['flip_cam'] 
        dim = stack.shape[z_axis]
        remaining_axis = np.linspace(0, len(stack.shape)-1, len(stack.shape)).astype(np.int32)
        remaining_axis = np.delete(remaining_axis, [z_axis, camera_axis])
        Nx, Ny = remaining_axis[0], remaining_axis[1]
        planes_per_cam = int(n_planes/stack.shape[camera_axis])
        if size_estimate is None: 
            size_estimate = stack.shape[Nx]*stack.shape[Ny]/(planes_per_cam)*1.2

        mip = np.median(stack, axis=z_axis)
        fov_props = {}
        angle_props = {}
        max_height, max_width = 0, 0
        for cam in range(stack.shape[camera_axis]):
            fov_props[cam] = {}
            stack[:,cam,:,:], deskew_angle = self.upright_images(stack[:,cam,:,:])
            angle_props[cam] = deskew_angle
            mip = np.max(stack[:,cam,:,:], axis=z_axis) 
            mip = skim.filters.gaussian(mip, sigma=10, preserve_range=True)
            th = np.quantile(mip.ravel(), 0.2)

            props = self.threshold_segment(mip, th)

            fov_size = np.mean([x.area_bbox for x in props])
            lower_fovsize, upper_fovsize = size_estimate*0.8, size_estimate*1.15
            print(f"Adaptive thresholding cam {cam}..")
            while len(props) != planes_per_cam and not lower_fovsize < fov_size < upper_fovsize:
                if len(props) > planes_per_cam or fov_size < lower_fovsize:
                    th += 10
                elif len(props) < planes_per_cam or fov_size > upper_fovsize:
                    th -= 4    
                props = self.threshold_segment(mip, th)
                fov_size = np.mean([x.area_bbox for x in props])

            for idx, p in enumerate(props):
                fov_props[cam][idx] = list(p.bbox)
                if p.bbox[2]-p.bbox[0] > max_width:
                    max_width = p.bbox[2]-p.bbox[0]
                if p.bbox[3]-p.bbox[1] > max_height:
                    max_height = p.bbox[3]-p.bbox[1]
        print("Cropping final fovs..")
        image_crops = np.empty(shape=(n_planes, dim, max_width, max_height))

        for cam_idx, cam_props in fov_props.items():
            for planes_idx, planes_bbox in cam_props.items():
                planes_bbox = self.adjust_bbox(mip.shape, planes_bbox, (max_width, max_height))
                fov_idx = int(cam_idx*planes_per_cam+planes_idx)
                image_crops[fov_idx,:,:,:] = np.expand_dims(self.crop_bbox(stack[:,cam_idx,:,:].squeeze(), planes_bbox), axis=0)

                if flip[cam_idx]:
                    image_crops[fov_idx,:,:,:] = np.flip(np.squeeze(image_crops[fov_idx,:,:,:]), axis=1) # axis change?

        return image_crops.astype(np.uint16), fov_props, angle_props

    def crop_with_parameters(self, stack, P, n_planes=4, z_axis=0, camera_axis=0):
        flip = self.P['flip_cam'] 
        dim = stack.shape[z_axis]
        remaining_axis = np.linspace(0, len(stack.shape)-1, len(stack.shape), dtype=int)
        remaining_axis = np.delete(remaining_axis, [z_axis, camera_axis])
        Nx, Ny = remaining_axis[0], remaining_axis[1]
        planes_per_cam = int(n_planes/stack.shape[camera_axis])

        fov_props = P["fovs"]
        f0 = fov_props[0][0]
        max_width, max_height = f0[2]-f0[0], f0[3]-f0[1]
        for cam in range(stack.shape[camera_axis]):
            stack[:,cam,:,:], _ = self.upright_images(stack[:,cam,:,:], P["deg"][cam])

        image_crops = np.empty(shape=(n_planes, dim, max_width, max_height))

        for cam_idx, cam_props in fov_props.items():
            for planes_idx, planes_bbox in cam_props.items():
                fov_idx = int(cam_idx*planes_per_cam+planes_idx)
                image_crops[fov_idx,:,:,:] = np.expand_dims(self.crop_bbox(stack[:,cam_idx,:,:].squeeze(), P["fovs"][cam_idx][planes_idx]), axis=0)

                if flip[cam_idx]:
                    image_crops[fov_idx,:,:,:] = np.flip(np.squeeze(image_crops[fov_idx,:,:,:]), axis=0)

        return image_crops

    def threshold_segment(self, mip, th):
        binary = mip > th
        mask = np.logical_and(np.ones(mip.shape), binary > 0).astype(int)

        morph = skim.morphology.dilation(mask)
        #segm = skim.segmentation.clear_border(morph)
        label_image = skim.measure.label(morph)

        return skim.measure.regionprops(label_image)

    def adjust_bbox(self, shape, bbox, bbox_size):
        for i in range(len(shape)):
            while bbox[i+2]-bbox[i] < bbox_size[i]:
                diff = int(bbox_size[i] - (bbox[i+2]-bbox[i]))
                bbox[i] = int(np.max([np.floor(bbox[i]-diff/2), 0]))
                bbox[i+2] = int(np.min([np.floor(bbox[i+2]+diff/2), shape[i]]))
        return bbox

    def crop_bbox(self, stack, bb):
        return stack[:,bb[0]:bb[2], bb[1]:bb[3]]

    def save_yaml(self, stack, path):
        for i in range(stack.shape[1]):
            params = {}
            filename = os.path.join(path, f'Plane_{i}')
            params["Plane"] = i
            params["directory"] = os.path.join(filename, "")
            self.metadata_files[i] = filename
            with open(filename + ".yaml", "w") as file:
                yaml.dump(params, file)

    def save_stack(self, stack):
        for i in range(stack.shape[0]):
            tifffile.imwrite(self.metadata_files[i] + ".tiff", stack[i])

    def get_average_transform_via_xcorr(self, stack, fp):
        # stack: z, t, y, x 
        # fp: focal planes (int), shape: (z,1)
        z, t,_,_ = stack.shape
        transforms = np.empty(shape=(z-1, 2))
        eval_points = min(t, 5)
        eval_points_delta  = np.linspace(max(-int(t/2), -20), min(int(t/2), 20), num=min(eval_points, 7), dtype=int)
        upsample = 100
        for p in range(z-1): 
            #eval_plane = int(np.round(np.mean([fp[p], fp[p+1]])))
            #iterate points in bead stack
            eval_points_shift = np.empty(shape=(eval_points, 2))
            for z_point in range(eval_points):
            # pixel level precision first
                shift, _,_ = skim.registration.phase_cross_correlation(stack[0,fp[0]+eval_points_delta[z_point],:,:], stack[p+1,fp[p+1]+eval_points_delta[z_point],:,:], upsample_factor=upsample) # evaluate at focal plane
                eval_points_shift[z_point] = shift/upsample
            transforms[p] = np.mean(eval_points_shift, axis=0)
        return transforms


    def transform_stack(self, stack, transform):
        # stack: z, t, y, x 
        # transform:  (z,2) (xy shift vector)
        z, _, _, _ = stack.shape
        outer = tqdm(total=z-1, desc='Image plane', position=1)
        for ip in range(z-1):
            stack[ip+1,:,:,:] = self.shift_via_fft(stack[ip+1,:,:,:].squeeze(), transform[ip])
            outer.update(1)
        return stack

    def shift_via_fft(self, stack, transform):
        # stack: t, y, x
        # transform: (x,y)
        t, _, _ = stack.shape
        inner = tqdm(total=t, desc='timepoint', position=0)
        for img in range(t):
            transformed_img = scp.ndimage.fourier_shift(input=np.fft.fftn(stack[img,:,:]), shift=transform)
            stack[img,:,:] = np.fft.ifftn(transformed_img)
            inner.update(1)
        return stack

    def estimate_interplane_distance(self, stack):
        from multiplane_calibration import MultiplaneCalibration
        cal = MultiplaneCalibration()
        cal.set_zstep(self.P['dz'])
        res = cal.estimate_interplane_distance(stack)
        self.create_cal_path()
        self.write_figure(cal.figs['dz'], self.cal_path, "interplane_distance", '.svg')
        self.write_figure(cal.figs['dz'], self.cal_path, "interplane_distance", '.png')

        return cal.dz['dz'], cal.order


    def write_figure(self, f, outpath, fname, filetype):
        # f: figure handle
        # path: output path
        # fname: filename

        output_name = os.path.join(outpath, fname+filetype)
        makeFolder(outpath)
        plt.show(f)
        plt.savefig(output_name, dpi = 600, bbox_inches="tight", pad_inches=0.1, transparent=True)    



    def execute(self):
        for self.root, _, self.filenames in os.walk(self.path):
            self.filenames = [os.path.join(self.root, file) for file in self.filenames if file.endswith(tuple(self.file_extensions))]
        self.filenames.sort()
        print("Data Directory:", self.path)


        self.check_calibration()


        run=True
        filecounter=0
        idx=0 # batchindex, counter
        f=0 #framecounter
        ncams = self.P['ncams']
        

        file_specifier = os.path.splitext(self.filenames[filecounter])[0]
        tif = tifffile.TiffFile(os.path.join(self.path, self.filenames[filecounter]))
        width, height = tif.pages._keyframe.keyframe.imagewidth, tif.pages._keyframe.keyframe.imagelength

        while run:
            # file loading via tifffile plugin, reading metadata for parsing
            try:
                #image = tifffile.imread(os.path.join(path, filenames[0]), key=range(f, f+framebatch, framebatch))
                image=np.zeros((int(np.ceil(self.P['dF_batch'] / ncams)), ncams, width, height))
                kk=0
                while kk<self.P['dF_batch']:
                    if f is len(tif.pages):
                        filecounter+=1 
                        tif=tifffile.TiffFile(os.path.join(self.path, self.filenames[filecounter]))
                        f=0
                    
                    image[int(np.floor(kk/ncams)),int(kk%ncams), :,:] = np.transpose(tif.pages[int(f)].asarray(), [1,0])
                    kk+=1 # current frame counter
                    f+=1 # global frame counter
                idx+=1 # savefile counter

            except:
                run=False
                continue


            print(f'\n Processing {self.filenames[0]} file {idx}')


            if self.log:
                fig, axs= plt.subplots(image.shape[1], image.shape[0], figsize=(9, 27),
                                subplot_kw={'xticks': [], 'yticks': []})
                #for ax in axs.flat: 
                for row in range(image.shape[0]):
                    for col in range(image.shape[1]):
                        axs[col, row].imshow(image[row, col,...])


        #for file in tqdm(self.filenames):
        #    if not os.path.isfile(file):
        #        continue


'''
            stack = tifffile.imread(file)
            # for single plane data handling
            if len(stack.shape) == 4:
                stack = stack[:, 0, :, :]

            P = None
            if P is None:
                image_crops, fov_props, angle_props = self.adaptiveThreshold(stack, 4, 0, 1)
                P = {"fovs": fov_props, "deg": angle_props}
            else:
                image_crops = self.crop_with_parameters(stack, P, 4, 0, 1)

            self.save_yaml(image_crops, self.output_path)
            self.save_stack(image_crops)
            
'''             

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
import numpy as np
import ipywidgets as ipyw
import matplotlib.pyplot as plt
from IPython.display import display


class ImageSliceViewer3D:
    """ 
    ImageSliceViewer3D is for viewing volumetric image slices in jupyter or
    ipython notebooks. 
    
    User can interactively change the slice plane selection for the image and 
    the slice plane being viewed. 

    Argumentss:
    Volume = 3D input image
    figsize = default(8,8), to set the size of the figure
    cmap = default('plasma'), string for the matplotlib colormap. You can find 
    more matplotlib colormaps on the following link:
    https://matplotlib.org/users/colormaps.html
    
    code grabbed from https://github.com/mohakpatel/ImageSliceViewer3D
    
    """
    
    def __init__(self, volume, figsize=(8,8), cmap='plasma'):
        self.volume = volume
        self.figsize = figsize
        self.cmap = cmap
        self.v = [np.min(volume), np.max(volume)]
        
        # Call to select slice plane
        ipyw.interact(self.view_selection, view=ipyw.RadioButtons(
            options=['x-y','y-z', 'z-x'], value='x-y', 
            description='Slice plane selection:', disabled=False,
            style={'description_width': 'initial'}))

    
    def view_selection(self, view):
        # Transpose the volume to orient according to the slice plane selection
        orient = {"y-z":[1,2,0], "z-x":[2,0,1], "x-y": [0,1,2]}
        self.vol = np.transpose(self.volume, orient[view])
        maxZ = self.vol.shape[2] - 1
        
        # Call to view a slice within the selected slice plane
        ipyw.interact(self.plot_slice, 
            z=ipyw.IntSlider(min=0, max=maxZ, step=1, continuous_update=False, 
            description='Image Slice:'))
        
    def plot_slice(self, z):
        # Plot slice for the given plane and slice
        self.fig = plt.figure(figsize=self.figsize)
        plt.imshow(self.vol[:,:,z], cmap=plt.get_cmap(self.cmap), 
            vmin=self.v[0], vmax=self.v[1])


class dualImageSliceViewer3D:
    
    def __init__(self, volume1, volume2, figsize=(20,20), cmap='plasma'):
        self.volume1 = volume1
        self.volume2 = volume2
        self.figsize = figsize
        self.cmap = cmap
        self.v1 = [np.min(volume1), np.max(volume1)]
        self.v2 = [np.min(volume2), np.max(volume2)]
        
        # Call to select slice plane
        ipyw.interact(self.view_selection2, view=ipyw.RadioButtons(
            options=['x-y','y-z', 'z-x'], value='x-y', 
            description='Slice plane selection:', disabled=False,
            style={'description_width': 'initial'}))

    def view_selection2(self, view):
        # Transpose the volume to orient according to the slice plane selection
        orient = {"y-z":[1,2,0], "z-x":[2,0,1], "x-y": [0,1,2]}
        self.vol1 = np.transpose(self.volume1, orient[view])
        maxZ1 = self.vol1.shape[2] - 1
        self.vol2 = np.transpose(self.volume2, orient[view])
        
        # Call to view a slice within the selected slice plane
        ipyw.interact(self.plot_slice2, 
            z=ipyw.IntSlider(min=0, max=maxZ1, step=1, continuous_update=False, 
            description='Image Slice:'))
    
    def plot_slice2(self, z):
        # Plot slice for the given plane and slice
        self.fig = plt.figure(figsize=self.figsize)
        self.ax = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)
        self.ax.imshow(self.vol1[:,:,z], cmap=plt.get_cmap(self.cmap), 
            vmin=self.v1[0], vmax=self.v1[1])
        self.ax2.imshow(self.vol2[:,:,z], cmap=plt.get_cmap(self.cmap), 
            vmin=self.v2[0], vmax=self.v2[1])





class ThresholdImageSliceViewer3D:
    
    def __init__(self, volume, operator = '<', figsize=(20,20), cmap='plasma'):
        self.volume1 = volume
        self.th =  np.min(volume) + (np.max(volume)-np.min(volume))/2
        self.operator = operator
        if self.operator == '<':
            self.volume2 = volume < self.th
        else: 
            self.volume2 = volume > self.th

        self.figsize = figsize
        self.cmap = cmap
        self.z = 0  
        self.v1 = [np.min(volume), np.max(volume)]
        #self.v2 = [np.min(self.volume2), np.max(self.volume2)]
        self.v2 = [0, 1]

        
        # Call to select slice plane
        ipyw.interact(self.view_selection2, view=ipyw.RadioButtons(
            options=['x-y','y-z', 'z-x'], value='x-y', 
            description='Slice plane selection:', disabled=False,
            style={'description_width': 'initial'}))

        

    def show_images(self, slice_no, threshold):
        # Plot slice for the given plane and slice
        self.fig = plt.figure(figsize=self.figsize)
        self.ax = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)
        self.z = slice_no
        self.th = threshold

        if self.operator == '<':
            self.vol2 = self.vol1 < self.th
        else: 
            self.vol2 = self.vol1 > self.th

        self.ax.imshow(self.vol1[:,:,slice_no], cmap=plt.get_cmap(self.cmap), 
            vmin=self.v1[0], vmax=self.v1[1])
        self.ax2.imshow(self.vol2[:,:,slice_no], cmap=plt.get_cmap(self.cmap), 
            vmin=self.v2[0], vmax=self.v2[1])
        plt.show()


    def view_selection2(self, view):
        # Transpose the volume to orient according to the slice plane selection
        orient = {"y-z":[1,2,0], "z-x":[2,0,1], "x-y": [0,1,2]}
        self.vol1 = np.transpose(self.volume1, orient[view])
        maxZ1 = self.vol1.shape[2] - 1
        self.vol2 = np.transpose(self.volume2, orient[view])

        slice_slider =ipyw.IntSlider(min=0, max=maxZ1, step=1, value = self.z, continuous_update=False, description='Image Slice:')
        threshold_slider = ipyw.IntSlider(min=0, max=np.max(self.volume1), step=1, value = self.th, continuous_update=False,  description='Threshold:')
        
        ipyw.interact(self.show_images, slice_no=slice_slider, threshold=threshold_slider)
    
    def get_threshold(self):
        if self.operator == '<':
            out = self.volume1 < self.th
        else: 
            out = self.volume1 > self.th
        
        return self.th, out



class MultiImageSliceViewer3D:
    """ 
    ImageSliceViewer3D is for viewing volumetric image slices in jupyter or
    ipython notebooks. 
    
    User can interactively change the slice plane selection for the image and 
    the slice plane being viewed. 

    Argumentss:
    Volumes = List of 3D input images
    figsize = default(8,8), to set the size of the figure
    cmap = default('plasma'), string for the matplotlib colormap. You can find 
    more matplotlib colormaps on the following link:
    https://matplotlib.org/users/colormaps.html
    
    code grabbed from https://github.com/mohakpatel/ImageSliceViewer3D
    
    """
    
    def __init__(self, volumes, figsize=(20,20), cmap='plasma'):
        self.volumes = volumes
        self.figsize = figsize
        self.cmap = cmap
        self.v = [np.min(volumes[0]), np.max(volumes[0])]
        
        # Call to select slice plane
        ipyw.interact(self.view_selection, view=ipyw.RadioButtons(
            options=['x-y','y-z', 'z-x'], value='x-y', 
            description='Slice plane selection:', disabled=False,
            style={'description_width': 'initial'}))

    
    def view_selection(self, view):
        # Transpose each volume to orient according to the slice plane selection
        orient = {"y-z":[1,2,0], "z-x":[2,0,1], "x-y": [0,1,2]}
        self.vols = [np.transpose(vol, orient[view]) for vol in self.volumes]
        maxZ = self.vols[0].shape[2] - 1
        
        # Call to view a slice within the selected slice plane
        ipyw.interact(self.plot_slices, 
            z=ipyw.IntSlider(min=0, max=maxZ, step=1, continuous_update=False, 
            description='Image Slice:'))
        
    def plot_slices(self, z):
        # Plot slice for the given plane and slice for all volumes in the same figure
        self.fig = plt.figure(figsize=self.figsize)
        ncols = 6
        n = int(np.ceil(len(self.vols)/ncols))
        for i, vol in enumerate(self.vols):
            plt.subplot(n, ncols, i+1)
            plt.imshow(vol[:,:,z], cmap=plt.get_cmap(self.cmap), 
                vmin=self.v[0], vmax=self.v[1])




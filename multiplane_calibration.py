import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
import skimage.filters as skfilt
from skimage import feature, io
from tqdm import tqdm 
from scipy.optimize import curve_fit
from matplotlib.colors import LinearSegmentedColormap
import warnings
import os
warnings.filterwarnings('ignore')


class MultiplaneCalibration:
    # calibration on diffraction limited bead data
    # fovs of subimages already selected, transformation not yet calculated
    # input: stack, (plane, z, x, y) of bead data

    def __init__(self):
        self.dz = {}
        self.pos_sr = {}
        self.bead_sr = {}
        self.order = []
        self.beads = {}
        self.log = False
        self.tracks={}
        self.figs = {}
        #processing parameters
        self.pp = {'gauss_sigma': 2, # sigma for DoG gaussian kernel
                   'roi' : 6, # roi radius around peak, to delete locs near edges 
                   'frame_min' : 15, # min amount of consecutive frames to consider it a bead trace
                   'd_max' : 5, # maximum distance of locs in consecutive frames to be considered belonging to the same trace 
                   'zstep': 10}  # default stage zstep size for coregistration nd PSF fitting


    def setlog(self, log=bool):
        self.log = log
        return self


    def estimate_interplane_distance(self, stack):
        print('Estimating interplane distance..')
        planes = stack.shape[0]
        self.pp['planes'] = planes
        self.pp['stack_height'] = stack.shape[1]
        pos_candidate_all = {}
        beadID_candidate_proj = {}
        outer = tqdm(total=planes, desc='Finding peak candidates', position=0)
        for p in range(planes):
            outer.update(1)
            pos_candidate_all[p] = self.find_candidate_positions(stack[p,...])
            beadID_candidate_proj[p] = self.find_candidate_positions_in_projection(stack[p,...])

        outer = tqdm(total=planes, desc='SR-localising peaks', position=0)
        for p in range(planes):
            self.pos_sr[p]= self.localise_candidates(stack[p,...], pos_candidate_all[p])
            outer.update(1)
        
        # Identifying beads in z
        outer = tqdm(total=planes, desc='Tracking beads in z', position=0)
        for p in range(planes):
            outer.update(1)
            self.beads[p] = self.track_locs_in_z(self.pos_sr[p], beadID_candidate_proj[p])
            self.beads[p] = self.clean_up_tracks(self.beads[p])

        outer = tqdm(total=planes, desc='Convert datastructure', position=0)
        for p in range(planes):
            outer.update(1)
            self.tracks[p] = self.convert_dict_to_array(self.beads[p])
            
        print('Determining relative z-distances and order')
        self.dz, self.order = self.get_dz()
        return self.dz
        #return self.tracks


    def set_zstep(self, zstep):
        self.pp['zstep'] = zstep
        
    def get_figures(self):
        return self.figs


    def convert_dict_to_array(self, dict):
        vals = dict.values()
        return np.array(list(vals))

    def find_candidate_positions_in_projection(self, stack):
        assert len(stack.shape)==3, "stack has wrong dimensions"
        z_pos, Nx, Ny = stack.shape
        mip = np.max(stack, axis=0)
        sigma = self.pp['gauss_sigma']

        mip_filt = skfilt.difference_of_gaussians(mip, low_sigma=sigma)

        
        # find local peaks, use situational threshold
        # clean up locs from the edges 
        
        th = np.std(mip_filt)*2 # minval local max
        locs = feature.peak_local_max(mip_filt, threshold_abs = th)

        # consolidate by removing locs from the borders
        markForDeletion = []
        for i in range(locs.shape[0]-1):
            if (locs[i][1] <= (self.pp['roi']+1)) \
            or (locs[i][1] >= Nx-(self.pp['roi']+1)) \
            or (locs[i][0] <= (self.pp['roi']+1)) \
            or (locs[i][0] >= Ny-(self.pp['roi']+1)):
                markForDeletion = np.append(markForDeletion,i)

        #And delete these indeces from the array
        markForDeletion = np.int_(markForDeletion)
        locs = np.delete(locs,markForDeletion,axis=0)
        # append z position in the stack to the loc
        locs = np.append(locs, np.zeros((locs.shape[0],1), dtype=np.uint16), axis=1)

        return locs

    def find_candidate_positions(self, stack):
        assert len(stack.shape)==3, "stack has wrong dimnensions"
        z_pos, Nx, Ny = stack.shape
        # stack: z,x,y beadstack in a single subimage
        inner = tqdm(total=z_pos, desc='Filtering', position=0)
        filt = np.empty_like(stack, dtype=np.float32)
        # apply dog filter to amplify spatial derivatives
        sigma = self.pp['gauss_sigma']
        for z in range(z_pos):
            filt[z,:,:] = skfilt.difference_of_gaussians(stack[z,:,:], low_sigma=sigma)
            inner.update(1)
        
        # find local peaks, use situational threshold
        # clean up locs from the edges 
        #loc_peaks = np.empty((1,3)) # container of final locs
        inner = tqdm(total=z_pos, desc='Peak finding', position=0)
        for z in range(z_pos):
            th = np.std(filt[z,:,:])*2 # minval local max
            locs = feature.peak_local_max(filt[z,:,:], threshold_abs = th)

            # consolidate by removing locs  from the borders
            markForDeletion = []
            for i in range(locs.shape[0]-1):
                if (locs[i][0] <= (self.pp['roi']+1)) \
                or (locs[i][0] >= Nx-(self.pp['roi']+1)) \
                or (locs[i][1] <= (self.pp['roi']+1)) \
                or (locs[i][1] >= Ny-(self.pp['roi']+1)):
                    markForDeletion = np.append(markForDeletion,i)

            #And delete these indeces from the array
            markForDeletion = np.int_(markForDeletion)
            locs = np.delete(locs,markForDeletion,axis=0)
            # append z position in the stack to the loc
            locs = np.append(locs, np.ones((locs.shape[0],1), dtype=np.uint16)*z, axis=1)

            inner.update(1)

            if z == 0: 
                loc_peaks = locs
            else:
                loc_peaks = np.append(loc_peaks, locs, axis=0)

        return loc_peaks
         


    # localise position accurately with phasor method
    def localise_candidates(self, stack, pos_candidate):
        assert len(stack.shape)==3, "stack has wrong dimnensions"
        rr = self.pp['roi']
        z_pos, Nx, Ny = stack.shape
        pos_sr = pos_candidate.copy().astype(np.float32)
        pos_sr = np.append(pos_sr, np.zeros((pos_sr.shape[0], 1), dtype=np.float32), axis=1)

        if self.log:
            f = plt.figure()
            #Calculate the number of subplots that will be drawn - based on how many peaks are found
            subplotsize = int(np.ceil(np.sqrt(pos_candidate.shape[0])))
        
        skip_counter = 0    

        for peak in range(pos_candidate.shape[0]):
            l = pos_candidate[peak]
            roi = stack[l[2], int(l[0]-rr):int(l[0]+rr), int(l[1]-rr):int(l[1]+rr)]
            
            if roi.shape[0]!=roi.shape[1] or roi.shape[0]==0:
                skip_counter+=1
                #print(f"Skipping peak {peak}, irregular shape: {roi.shape}")
                continue

            if self.log:
                #Show the ROIs in a subplot image
                plt.subplot(subplotsize,subplotsize,peak+1)
                plt.imshow(roi.copy(),cmap='gray')


            # sr position in roi crop 
            roi_pos = list(self.phasor_localise(roi))

            # update to global coordinates
            sr_pos = [l[0]-rr+roi_pos[0], l[1]-rr+roi_pos[1]]
            #sr_pos[0] = l[0]-rr+roi_pos[0]
            #sr_pos[1] = l[1]-rr+roi_pos[1]

            phot = absolute_intensity(roi, roi_pos)
            #phot = photometry_intensity(roi)

            pos_sr[peak][0] = sr_pos[0] # ypos
            pos_sr[peak][1] = sr_pos[1] # xpos
            pos_sr[peak][3] = phot # brightness


        print(f"Skipped {skip_counter} / {pos_candidate.shape[0]} ({skip_counter*100/pos_candidate.shape[0] :.2f}%) peaks in fitting due to irregular shape.")

        return pos_sr


    def track_locs_in_z_all(self, pos_sr):
        # assuming there is only one or less corresponding loc per frame in z
        beads = {}
        bidx = 0 # bead_idx 
        run = True
        z_loop = 0
        max_frames = self.pp['planes']
        #pos_sr = self.pos_sr.copy()
        while run:
            
            # search for new bead position in current z_loop 
            loc = next((l for l in pos_sr if l[2] == z_loop), None)
            if loc is None:
                if z_loop != max_frames:
                    z_loop += 1
                else:     
                    run = False
                    break

            else:
                # if unaccounted localisation exist, assign to a new bead and iterate through the stack to find its corresponding z-positions
                beads[bidx] = [loc]
                pos_sr = delete_loc(pos_sr, loc) 
                #np.delete(pos_sr, loc, axis=0)
                


                for z in range(z_loop+1, max_frames):
                    next_frame = pos_sr[pos_sr[:,2]==z]
                    next_point = self.find_closest_neighbour_in_next_frame(beads[bidx][-1][:2], next_frame)
                    if next_point is None:
                        #beads[bidx].append(np.zeros(4))
                        beads[bidx].append(['None', 'None', z, 'None'])
                    else:
                        beads[bidx].append(next_point)
                        # remove loc from original loc array 
                        pos_sr = delete_loc(pos_sr, next_point)

                bidx += 1 # update bead idx after loop through stack for new assigment 
        
        return beads
    

    def track_locs_in_z(self, pos_sr, bead_pos):
        # assuming there is only one or less corresponding loc per frame in z
        tracks = {}
        max_frames = self.pp['stack_height']
        #pos_sr = self.pos_sr.copy()
        for p, _ in enumerate(bead_pos): 
            tracks[p] = []
            for z in range(max_frames-1):
                next_frame = pos_sr[pos_sr[:,2]==z]
                if next_frame.any():
                    if z == 0 or not tracks[p]:
                        next_point = self.find_closest_neighbour_in_next_frame(bead_pos[p][:2], next_frame)
                    else:  
                        next_point = self.find_closest_neighbour_in_next_frame(tracks[p][-1][:2], next_frame)
                else: 
                    
                    next_point = None
                #if next_point is None:
                    #beads[bidx].append(np.zeros(4))
                    #tracks[p].append([bead_pos[p][0], bead_pos[p][1], z, 'None'])
                #else:
                if next_point is not None:
                    tracks[p].append(next_point)
                    # remove loc from original loc array 
                    pos_sr = delete_loc(pos_sr, next_point)    

            tracks[p] = np.array(tracks[p])
        return tracks


    def clean_up_tracks(self, tracks):
        # remove tracks are not long enough, defined by pp parameter
        # interpolate values that havent been detected from localisation or tracking step  

        # remove all traces that are below length threshold 
        cleaned_tracks = {}
        for k in tracks.keys():
            if len(tracks[k]) >= self.pp['frame_min']:
                # interpolate missing z values
                cleaned_tracks[k] = self.expand_and_interpolate(tracks[k], self.pp['stack_height'])
                #cleaned_tracks[k] = tracks[k]
        return cleaned_tracks

        
       


    def expand_and_interpolate(self, arr, target_length):
        n, cols = arr.shape
        assert cols == 4, "Input array must have shape (n, 4)"
        
        if n >= target_length:
            return arr
        
        # Create a full z array from min to max + 1 with step 1
        min_z = 0
        max_z = self.pp['stack_height']
        full_z = np.arange(min_z, max_z)
        
        # Create a new array with the full set of z values
        new_arr = np.zeros((target_length, 4))
        
        # Set the z positions
        new_arr[:,3] = full_z

        arr_keys = np.delete(range(cols), 2)

        # Interpolate x, y, and the value at the z position
        for i in arr_keys:
            new_arr[:, i] = np.interp(full_z, arr[:, 2], arr[:, i])
        
        
        return new_arr



    def find_earliest_neighbour_in_next_frame(self, p, arr):
        # go through every element, calculate distance to current point
        # if point is within distnace, stop iteration 
        # return the index in the list 
        # p: (x1,y1) of target point
        # arr: np.array of neighbouring candidates
        
        for i, point in enumerate(arr):
            if self.get_distance(p, point) <= self.pp['d_max']:
                return point
            else:
                return None  # if no point is within  distance
            


    def find_closest_neighbour_in_next_frame(self, p, arr):
        # return the index in the list 
        # p: (x1,y1) of target point
        # arr: np.array of neighbouring candidates
        
        distances = np.sum((arr[:,:2] - np.array(p))**2, axis=1)
        
        # Find the index of the minimum distance
        closest_index = np.argmin(distances)

        if distances[closest_index] <= self.pp['d_max']**2:
            return arr[closest_index]
        else: 
            return None


    def get_distance(self, a, b):
        # get eucledian distance betwwen point a & b
        # a : (x1, y1)
        # b: (x2, y2)
        d = ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5

        return d


    def get_dz(self):
        x = range(self.pp['stack_height'])
        res = []
        fits = []
        data = []
        for t in self.tracks.keys():
            y=np.mean(self.tracks[t][:,:,3], axis=0)
            # Fit the Gaussian to the data
            data.append(y)
            params = self.fit_gaussian(x, y)
            res.append(params)
            fits.append(gaussian(x, *params))



        res = np.array(res)
        data = np.array(data)
        fits = np.array(fits)
        new_order = np.argsort(res[:,1])

        f = plot2LinesVerticalMarkers(data, fits, res[:,1], "grayvalue [1/units]", self.pp['zstep'])
        self.figs['dz'] = f


        #tracks = self.tracks
        #reorder tracks and get dz
        #tracks = np.sort(tracks, order=new_order)
        res = res[new_order, :] 
        
        self.order = new_order
        self.dz['dz'] = [res[i+1,1]-res[i,1] for i in range(res.shape[0]-1)]
        self.dz['labels'] = [f'{i+1}-{i}' for i in range(res.shape[0]-1)]

        return self.dz, self.order
    

    def phasor_localise(self, roi):
        #Perform 2D Fourier transform over the complete ROI
        roi_f = np.fft.fft2(roi)
        xangle = np.arctan(roi_f[0,1].imag/roi_f[0,1].real) - np.pi
        #Correct in case it's positive
        if xangle > 0:
            xangle -= 2*np.pi
        #Calculate position based on the ROI radius
        xpos = abs(xangle)/(2*np.pi/(self.pp['roi']*2+1))+0.5

        #Do the same for the Y angle and position
        yangle = np.arctan(roi_f[1,0].imag/roi_f[1,0].real) - np.pi
        if yangle > 0:
            yangle -= 2*np.pi
        ypos = abs(yangle)/(2*np.pi/(self.pp['roi']*2+1))+0.5

        return (xpos,ypos)
    
    def fit_gaussian(self, x, y):
        # Initial guesses for the parameters
        amp_init = np.max(y) - np.min(y)
        mean_init = np.sum(x * y) / np.sum(y)
        stddev_init = np.sqrt(np.sum(y * (x - mean_init) ** 2) / np.sum(y))
        offset_init = np.min(y)
        
        # Use curve_fit to fit the Gaussian function to the data
        popt, pcov = curve_fit(gaussian, x, y, p0=[amp_init, mean_init, stddev_init, offset_init])

        return popt


def plot2LinesVerticalMarkers(data1, data2, xMarker, yval, fileSpecifier = "zcalib_", ZDIST = 1):
    
    cdict = {'violet': "#960792ff",
             'orange': "#ff3b3bff", 
             'yellow': "#b28900ff", 
             'blue': "#009bb2ff"}
    clist = list(cdict.values())
    
    cmap = LinearSegmentedColormap.from_list("prism_colors", clist)
    
    fig, ax = plt.subplots()
    n_planes = len(data1[0])
    print(f"n_planes: {n_planes}")
    #cmap = cm.get_cmap('Pastel1', 8)
    z = np.linspace(0,ZDIST*n_planes, n_planes)
    idx = 0
    lgd = []
    LINES = []
    

    maxVal, minVal = np.max(data1), np.min(data1)
    
    for line, line2 in zip(data1, data2):

        LINES += ax.plot(z, line, color=clist[idx], alpha = 0.8, linewidth = 1)
        lgd.append("Channel {} - rel. z: {:.0f}nm; A: {:.1f}".format(idx, xMarker[idx], np.max(line2)))
        plt.plot(z, line2, color=clist[idx], linewidth = 1, linestyle='--', alpha = 1)
        plt.vlines(xMarker[idx], minVal, maxVal, color=clist[idx])
        #lgd.append(lgdlist[idx]+"_fit")
        idx = idx+1
        

    plt.ylim((minVal, maxVal))
    plt.xlim((0,np.max(z)))
    plt.xlabel("z 1/nm")
    plt.ylabel(yval)
    leg = Legend(ax, LINES, lgd, loc='lower right', bbox_to_anchor=(0.5, 1.), frameon=False)
    ax.add_artist(leg)
    
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)

    return ax
    
#    output_name = os.path.join(outpath, fileSpecifier+".svg")
#    plt.savefig(output_name, dpi = 600, bbox_inches="tight", pad_inches=0.1, transparent=True)    
    

def absolute_intensity(ROI, xy):
    w = [xy[0]-np.floor(xy[0]), xy[1]-np.floor(xy[1])]

    c = np.array([[np.floor(xy[0]), np.floor(xy[0])+2], 
            [np.floor(xy[1]), np.floor(xy[1])+2]]).astype(np.uint8)# point coordinates 
    
    # check whether coords are at the border and shift if necessary
    if c[0][1] >= ROI.shape[0]-1:
        c[0] = c[0]-1

    if c[1][1] >= ROI.shape[1]-1:
        c[1] = c[1]-1
    val = approx_interp2(ROI[c[0][0]:c[0][1], c[1][0]:c[1][1]], w)

    #val = np.max(ROI)

    return val



def approx_interp2(val,w):
    #val: 2,2 neighbouring values
    #w: weights for interpolation 
    return (val[0,0]*w[0]+ val[1,0]*(1-w[0]))*w[1] + (val[0,1]*w[0]+ val[1,1]*(1-w[0]))*(1-w[1])

#################################################################
#The intensity-measure we will use is a photometry-based method, described in:
#S. Preus, L.L. Hildebrandt, and V. Birkedal, Biophys. J. 111, 1278 (2016).
#Also see Figure S11 in doi.org/10.1063/1.5005899 (Martens et al., 2017)

def photometry_intensity(ROI):
  #First we create emtpy signal and background maps with the same shape as
  #the ROI.
  SignalMap = np.zeros(ROI.shape)
  BackgroundMap = np.zeros(ROI.shape)
  #Next we determine the ROI radius from the data
  ROIradius = (ROI.shape[0]-1)/2

  #Now we attribute every pixel in the signal and background maps to be
  #belonging either to signal or background based on the distance to the
  #center
  #For this, we loop over the x and y positions
  for xx in range(0,ROI.shape[0]):
    for yy in range(0,ROI.shape[1]):
      #Now we calculate Pythagoras' distance from this pixel to the center
      distToCenter = np.sqrt((xx-ROI.shape[0]/2+.5)**2 + (yy-ROI.shape[1]/2+.5)**2)
      #And we populate either SignalMap or BackgroundMap based on this distance
      if distToCenter <= (ROIradius): #This is signal for sure
        SignalMap[xx,yy] = 1
      elif distToCenter > (ROIradius-0.5): #This is background
        BackgroundMap[xx,yy] = 1

  #Now we take the 56th percentile of the data in the background map.
  #This is a valid measure for the expected background intensity

  #First we use the BackgroundMap as a mask for the intensity data, and use
  #that to get a list of Background Intensities.
  BackgroundIntensityList = np.ma.masked_array(ROI, mask=BackgroundMap).flatten()
  #And then we take the 56th percentile (or the value closest to it)
  if len(BackgroundIntensityList) > 0:
    BackgroundIntensity = np.percentile(BackgroundIntensityList,56)
  else:
    BackgroundIntensity = 0

  #To then assess the intensity, we simply sum the entire ROI in the SignalMap
  #and subtract the BackgroundIntensity for every pixel
  SignalIntensity = sum((ROI*SignalMap).flatten())
  SignalIntensity -= BackgroundIntensity*sum(SignalMap.flatten())

  #And we let the function return the SignalIntensity
  return max(0,SignalIntensity)



def delete_loc(matrix, loc_to_delete):
    # Find the index of the row to delete
    index = np.where((matrix == loc_to_delete).all(axis=1))[0][0]
    # Delete the row using np.delete
    matrix = np.delete(matrix, index, axis=0)
    return matrix


def gaussian(x, amp, mean, stddev, offset):
    return amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2)) + offset

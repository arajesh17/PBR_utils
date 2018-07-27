"Code to fill the Foramen Magnum of the C1 area with the intensities of local CSF voxels"

import numpy as np
from scipy import ndimage
import nibabel as nib
from copy import deepcopy
from glob import glob
from subprocess import check_call
import argparse
import os

def _get_output(mseid):
    num = int(mseid[3:])
    if num > 4500: return '/data/henry11/PBR/subjects/'
    else: return '/data/henry7/PBR/subjects/'

def i_img(i, shape, img_closing, img_data, aff, sub_name):
    # i is an integer that specifies number of iterations for binary dilation
    print('\n Running {!s} interations of dilation around the cord to get CSF intensity values'.format(i))
    i_zero = np.zeros(shape)
    kernel = np.ones((3,3,1))
    i_zero[ndimage.morphology.binary_dilation(img_closing, structure = kernel, iterations = i)] = 1
    i_sub = np.subtract(i_zero, img_closing)
    i_sub_data = img_data[np.where(i_sub == 1)]
    i_sub_mean = np.mean(i_sub_data)
    print('Mean intensity of voxels {!s} iterations around cord is {!s}'.format(i, i_sub_mean))
    i_out = deepcopy(img_data)
    i_out[np.where(img_closing == 1)] = i_sub_mean
    i_nii = nib.Nifti1Image(i_out, affine=aff)
    i_out_fname = '{}_i{!s}_filled.nii.gz'.format(sub_name.split('.')[0], i)
    nib.save(i_nii, i_out_fname)
    print('Saved {!s} iteration at {}'.format(i, i_out_fname))

def generate_seg(args):

    mseid = args.mse
    
    if args.ext:
        file_ext = args.ext
    else:
        file_ext = '_10caudA.roi'

    if args.workingdirectory:
        working_dir = args.workingdirectory
    else:
        working_dir =  os.path.join(_get_output(mseid), mseid, 'alignment', 'baseline_mni', 'no_chop')
    
    if not glob(os.path.join(working_dir, '*{}'.format(file_ext))):
        raise ValueError('the file {} does not exist'.format((os.path.join(_get_output(mseid), mseid, 
                         'alignment', 'baseline_MNI', 'no_chop', '*{}'.format(file_ext)))))

    volume = glob(os.path.join(working_dir, '*{}'.format(file_ext)))[0]
    
    if args.nochop:
        img_name = volume.split('nochop')[0] + 'nochop.nii.gz'
    else:
        img_name = volume.split(file_ext)[0]+'.nii.gz'
    output = volume.split('.')[0] + '.nii.gz'
    script = '/data/henry2/arajesh/henry10_clone/SC_registration/scripts/jim_roi_to_nii.py'
    cmd = ['python', script, '-img', img_name, '-roi', volume, '-output', output]
    print('\n','Running command: \n {} \n'.format(cmd))
    check_call(cmd)

    sub_name = output
    data = nib.load(sub_name).get_data()
    aff = nib.load(img_name).affine
    img = nib.load(img_name).get_data()

    out_data = deepcopy(data)

    distance = lambda a, b: np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    coords = np.where(data == 1)
    z_coords = np.unique(coords[2])
    for z in z_coords:

        com = ndimage.center_of_mass(data[:,:,z])
#         print('com', com, 'for z', z)
        x_array = coords[0][np.where(coords[2] == z)]
        y_array = coords[1][np.where(coords[2] == z)]
        points = np.array(list(zip(x_array, y_array)))

        radius_array = []
        for idx in range(len(points)):
            radius_array.append(distance(points[idx], com))
        radius = np.mean(radius_array)
#         print('radius is', radius)
        for x in range(np.min(x_array), np.max(x_array)+1):
            for y in range(np.min(y_array), np.max(y_array)+1):
                #print('x', x, 'y', y, 'dist', distance((x,y), com), 'radius ref', radius)
                if distance((x,y), com) <= radius:
                    out_data[x,y,z] = 1
        binary_closing = np.zeros(out_data.shape)
        binary_closing[ndimage.morphology.binary_closing(out_data, structure=np.ones((2,2,2)))] = 1
        
    if args.iter:
        i_img(i, binary_closing.shape, binary_closing, img, aff, sub_name)
    else:
        i_img(1, binary_closing.shape, binary_closing, img, aff, sub_name) 
    

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('mseid', help= 'the mseid that you want to run the foramen magnum intensity correction on')
    parser.add_argument('-x', '--ext', help= 'the extension of the filename that your .roi files are saved under the default is _10caudA.roi')
    parser.add_argument('-nc', '--nochop', action='store_true', help= 'Enter 0 if you have saved your files not in the no_chop directory, this will allow the script to find the correct .roi files, default is 1')
    parser.add_argument('-i', '--iter', help='Number of iterations of dilation you want to do around the spinal cord to get the CSF intensity, defaul is 1 iteration')
    parser.add_argument('-w', '--workingdirectory', help='directory in PBR where you stored the .roi files, default is "/mseid/alignment/baseline_mni/no_chop"')
    args = parser.parse_args()
    generate_seg(args)

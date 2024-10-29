import os
import re
from typing import List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm

import skimage
import scipy.io
from skimage.transform import resize
import glob

from tifffile import tifffile
import tqdm

# The spindle position data.
matlab_file_paths = {
    'Y445D': list(glob.glob("./data/Individual workspaces Y445D/Y445D_cell*.mat")),
    'WildType': list(glob.glob("./data/Individual workspaces WT/WT*.mat")),
    'ase1-7A': list(glob.glob("./data/Individual workspaces ase1-7A/ase1_7A*.mat")),
    'ase1-7A; Y445D': list(glob.glob("./data/Individual workspaces ase1-7A; Y445D/ase1_7A_Y445D*.mat")),
    'cin8-3A; Y445D': list(glob.glob("./data/Individual workspaces cin8-3A; Y445D/cin8_3A_Y445D*.mat")),
    'cin8-3A': list(glob.glob("./data/Individual workspaces cin8-3A/cin8_3A*.mat")),
    'ase1-7A; cin8-3A': list(glob.glob('./data/Individual workspaces ase1-7A; cin8-3A/ase1_7A_cin8_3A*.mat')),
    'ase1 WT': list(glob.glob('/Volumes/Crucial X8/Shannon Ase1/WT/Individual workspaces/Ase1_WT*.mat')),
    'ase1 YD': list(glob.glob('/Volumes/Crucial X8/Shannon Ase1/Y445D/Individual workspaces/Ase1_Y445D*.mat')),
    'kip1 WT': list(glob.glob('/Volumes/Crucial X8/Shannon Kip1/WT/Individual workspaces/Kip1_WT*.mat')),
    'kip1 YD': list(glob.glob('/Volumes/Crucial X8/Shannon Kip1/Y445D/Individual workspaces/Kip1_Y445D*.mat')),
    'Y445F': list(glob.glob("./data/Individual workspaces Y445F/Y445F_cell*.mat")),
    'Clb2del': list(glob.glob("/Volumes/Sean SSD/clb2del/Individual workspaces/clb2del_cell*.mat")),
    'Clb2del-Y445D': list(glob.glob("/Volumes/Sean SSD/clb2del; Y445D/Individual workspaces/clb2del_Y445D_cell*.mat"))
}

# The corresponding image file paths.
image_file_paths = {
    'WildType': list(glob.glob("/Volumes/Sean SSD/Shannon Dynamic Kymograph Data/Cropped Data/Cropped data WT/cell*.tif")),
    'ase1-7A': list(glob.glob("/Volumes/Sean SSD/Shannon Dynamic Kymograph Data/Cropped Data/Cropped data ase1-7A/cell*.tif")),
    'ase1-7A; cin8-3A': list(glob.glob("/Volumes/Sean SSD/Shannon Dynamic Kymograph Data/Cropped Data/Cropped data ase1-7A; cin8-3A/cell*.tif")),
    'cin8-3A': list(glob.glob("/Volumes/Sean SSD/Shannon Dynamic Kymograph Data/Cropped Data/Cropped data cin8-3A/cell*.tif")),
    'ase1-7A; Y445D': list(glob.glob("/Volumes/Sean SSD/Shannon Dynamic Kymograph Data/Cropped Data/Cropped data ase1-7A; Y445D/cell*.tif")),
    'Y445D': list(glob.glob("/Volumes/Sean SSD/Shannon Dynamic Kymograph Data/Cropped Data/Cropped data Y445D/cell*.tif")),
    'cin8-3A; Y445D': list(glob.glob("/Volumes/Sean SSD/Shannon Dynamic Kymograph Data/Cropped Data/Cropped data cin8-3A; Y445D/cell*.tif")),
    'ase1 WT': list(glob.glob("/Volumes/Sean SSD/Shannon Dynamic Kymograph Data/Shannon Ase1/WT/Cropped data/cell*.tif")),
    'ase1 YD': list(glob.glob("/Volumes/Sean SSD/Shannon Ase1/Y445D/Cropped data/cell*.tif")),
    'kip1 WT': list(glob.glob("/Volumes/Sean SSD/Shannon Dynamic Kymograph Data/Shannon Kip1/WT/Cropped data/cell*.tif")),
    'kip1 YD': list(glob.glob('/Volumes/Sean SSD/Shannon Dynamic Kymograph Data/Shannon Kip1/Y445D/Cropped data/cell*.tif')),
    'Y445F': list(glob.glob('/Volumes/Sean SSD/Shannon Dynamic Kymograph Data/Cropped data Y445F/cell*.tif')),
    'Clb2del': list(glob.glob("/Volumes/Sean SSD/clb2del/Cropped data/cell*.tif")),
    'Clb2del-Y445D': list(glob.glob("/Volumes/Sean SSD/clb2del; Y445D/Cropped data/cell*.tif"))
}

# The default voxel sizes to fall back on if the metadata is not correct.
default_voxel_sizes = {
    'ase1-7A; Y445D': np.array([0.0725000, 0.0725000, 0.2]),
}


OUTPUT_SHAPE = (256, 256)
SAVE_INDIVIDUAL_KYMOGRAPHS = True
CMAP = mpl.colormaps['gray']


def get_values_along_path(x1, x2, image, kernel=np.ones((1, 1, 1))):
    """
    Return a list of values along the path of an image placing the kernel down on each pixel.
    :param x1: The starting point.
    :param x2: The ending point.
    :param image:
    :param kernel:
    :return:
    """
    # Get a list of the indices that we are working with.
    points = np.array(skimage.draw.line_nd(x1, x2)).transpose()
    result = np.zeros(points.shape[0])
    kernel_size = kernel.shape[0]
    kernel_r = int(np.floor(kernel_size / 2))
    # image = np.pad(
    #     image,
    #     ((kernel_r, kernel_r), (kernel_r, kernel_r), (kernel_r, kernel_r)),
    #     'constant', constant_values=((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    # )
    convolved_image = scipy.signal.convolve(image, kernel)
    for i in range(len(result)):
        # Apply the kernel and save the result.
        point = points[i, :]
        res = convolved_image[point[0], point[1], point[2]]
        # sub_image = image[
        #                 points[i, 0]-kernel_r:points[i, 0]+kernel_r+1,
        #                 points[i, 1]-kernel_r:points[i, 1]+kernel_r+1,
        #                 points[i, 2]-kernel_r:points[i, 2]+kernel_r+1
        #             ]
        # res = np.sum(sub_image * kernel)
        result[i] = res
    return result


def load_spindle_positions(data_path) -> np.ndarray:  # TODO: Test implementation.
    """
    Load and return the spindle positions as an array with shape (2, F, 3). (F is the number of frames.)
    :param data_path: The path to the matlab file.
    :return: (The coordinates, time index)
    """
    mat = scipy.io.loadmat(data_path)
    # spindle_length_mat = mat["Spindle_length"]

    op_pos_mat = mat['sp1']
    op_pos = op_pos_mat[:, :3]  # The first three cols are the positions.
    np_pos_mat = mat['sp2']
    np_pos = np_pos_mat[:, :3]  # The first three cols are the positions.

    time_indices = op_pos_mat[:, 4].astype(int)
    max_time_index = np.max(time_indices)
    n_timepoints = op_pos.shape[0]
    positions = np.zeros((2, max_time_index, 3))
    positions[:, :, :] = np.nan
    for i in range(len(time_indices)):
        t = time_indices[i]-1
        positions[0, t, :] = op_pos[i]
        positions[1, t, :] = np_pos[i]

    return positions, time_indices


def read_tiff(path, channel_order) -> np.ndarray:
    """
    Read a tif/tiff file and return the resulting image file as a numpy array.

    Results will not be normalized as they are "scaled to raw" in the SIM processing so they can be compared to
    each other.

    The indexing order for this array is [CHANNEL, FRAME, X, Y, Z].

    :param path: The path to the .tif file.
    :return: The `numpy.ndarray` containing the data. The index order is `[F, C, X, Y, Z]`.
    """
    image = tifffile.imread(path)
    with tifffile.TiffFile(path) as tif:
        axes_order = tif.series[0].axes
        axes_reorder = [axes_order.index(i) for i in channel_order]
        # t = axes_order.index('T')
        # c = axes_order.index('C')
        # x = axes_order.index('X')
        # y = axes_order.index('Y')
        # z = axes_order.index('Z')


        # image = np.transpose(image, (t, c, x, y, z))  # Reorder the image to be indexed correctly.
        image = np.transpose(image, tuple(axes_reorder))  # Reorder the image to be indexed correctly.

    return image


def read_voxel_size(path):
    """
    Implemented based on information found in https://pypi.org/project/tifffile
    Found on: https://forum.image.sc/t/reading-pixel-size-from-image-file-with-python/74798/2
    """

    def _xy_voxel_size(tags, key):
        assert key in ['XResolution', 'YResolution']
        if key in tags:
            num_pixels, units = tags[key].value
            return units / num_pixels
        # return default
        return None

    with tifffile.TiffFile(path) as tiff:
        image_metadata = tiff.imagej_metadata
        if image_metadata is not None:
            z = image_metadata.get('spacing', 1.)
        else:
            # default voxel size
            z = None

        tags = tiff.pages[0].tags
        # parse X, Y resolution
        y = _xy_voxel_size(tags, 'YResolution')
        x = _xy_voxel_size(tags, 'XResolution')
        # return voxel size

        voxel_size = np.array([x, y, z])
        return voxel_size


def load_image_data(strain_name, cell_number) -> (np.ndarray, np.ndarray):
    """
    Load and return the green/cin8 image data for the given strain and cell.
    :param strain_name:
    :param cell_number:
    :return: Tuple of the image and the voxel size.
    """
    # Subset the files to ones with this cell.
    possible_paths = image_file_paths[strain_name]
    path_hits = [p for p in possible_paths if len(re.findall(f'cell{cell_number}_', p)) > 0]

    # Load the data based on whether there are one or two matching files.
    if len(path_hits) == 1:  # There is a single file. TODO: Test this.
        CIN8_CHANNEL_THREE_CHANNEL = 2
        CIN8_CHANNEL_TWO_CHANNEL = 1

        img_data = read_tiff(path_hits[0], ['C', 'T', 'X', 'Y', 'Z'])
        num_channels = img_data.shape[0]
        voxel_size = read_voxel_size(path_hits[0])

        if num_channels == 2:
            return img_data[CIN8_CHANNEL_TWO_CHANNEL, :, :, :, :], voxel_size
        elif num_channels == 3:
            return img_data[CIN8_CHANNEL_THREE_CHANNEL, :, :, :, :], voxel_size
        else:
            raise Exception("Unexpected channel number.")

    elif len(path_hits) == 2:  # There is a file for the red and the green channel.
        file_one_path, file_two_path = path_hits
        red_path = file_one_path if len(re.findall(r'_red.tif', file_one_path)) == 1 else file_two_path
        green_path = file_two_path if len(re.findall(r'_red.tif', file_one_path)) == 1 else file_one_path
        voxel_size = read_voxel_size(green_path)

        _red_img = read_tiff(red_path, ['T', 'X', 'Y', 'Z'])
        green_img = read_tiff(green_path, ['T', 'X', 'Y', 'Z'])

        return green_img, voxel_size

    else:
        raise IOError(f'Issue loading file path for cell {cell_number} for strain {strain}')


def stretch_array(arr, new_len) -> np.ndarray:
    """
    Return a new array of a specified length that is a linear interpolation of the input.
    :param arr:
    :param new_len:
    :return:
    """
    x = np.linspace(0.0, 1.0, new_len)
    xp = np.copy(arr) if type(arr) == np.ndarry else np.array(arr)
    fp = np.linspace(0.0, 1.0, len(xp))
    new_arr = np.interp(x, xp, fp)
    return new_arr


def generate_single_dynamic_kymograph(pole_coordinates, image, voxel_size, time_indices, save_path=None) -> np.ndarray:
    """
    Generage a kymograph between the two poles for a given image. It will
    :param pole_coordinates:
    :param image:
    :param voxel_size:
    :param time_indices:
    :return:
    """
    time_indices = time_indices.astype(int)
    kymograph = [None for _ in range(int(np.max(time_indices)))]
    n_frames = pole_coordinates.shape[1]

    for t in time_indices:
        values = get_values_along_path(
            (pole_coordinates[0, t-1, :] / voxel_size),
            (pole_coordinates[1, t-1, :] / voxel_size),
            image[t-1, :, :, :],
            kernel=np.ones((3, 3, 3))
        )
        kymograph[t-1] = values

    if save_path is not None:  # Save the individual kymograph.
        # The output grid (for the un-stretched output.
        max_len = np.max([len(k) for k in kymograph if k is not None])
        unstretched_kymo = np.zeros((len(kymograph), max_len))
        for i, k in enumerate(kymograph):
            if k is None:
                kymo = np.zeros(max_len)
                kymo[:] = np.nan
                unstretched_kymo[i] = kymo
            else:
                kymo = np.zeros(max_len)
                kymo[:] = np.nan
                start_idx = (max_len - len(k)) // 2
                stop_idx = start_idx + len(k)
                kymo[start_idx:stop_idx] = np.array(k)
                unstretched_kymo[i] = kymo

        plt.imsave(
            save_path,
            np.transpose(resize(unstretched_kymo, OUTPUT_SHAPE, order=0)),
            dpi=500,
            cmap=CMAP
        )

    # Convert the kymograph into a square numpy matrix.
    max_len = np.max([len(k) for k in kymograph if k is not None])
    square_kymo = np.zeros((len(kymograph), max_len))
    for i, k in enumerate(kymograph):
        if k is None:
            kymo = np.zeros(max_len)
            kymo[:] = np.nan
            square_kymo[i] = kymo
        else:
            kymo = np.zeros(max_len)
            kymo[:] = resize(np.array(k), (max_len,))  # Rescale the pixel values to be the same length. (max_len)
            square_kymo[i] = kymo

    return square_kymo


def avg_kymographs(kymographs: List[np.ndarray], output_shape=OUTPUT_SHAPE):
    """
    Take a list of kymographs and return a single kymograph that is the average of the list.
    :param kymographs: The list of kymographs.
    :param output_shape: The shape of the output.
    :return: A numpy.ndarray that is the average of all of the inputs.
    """
    resized_kymographs = []
    for kymo in kymographs:
        resized_kymograph = resize(kymo, output_shape, order=0)  # resize(kymo, (max_time_points, max_len))
        resized_kymographs.append(resized_kymograph)

    # Convert to a numpy array and then take the average at each position
    # (over the axis that represents the kymograph index, not the positional/temporal axes.)
    resized_kymographs = np.array(resized_kymographs)
    avg_kymograph = np.nanmean(resized_kymographs, axis=0)

    return avg_kymograph


if __name__ == '__main__':
    os.makedirs('./dynamic_kymographs', exist_ok=True)
    for strain in matlab_file_paths.keys():
        data_paths = matlab_file_paths[strain]
        kymographs = []

        # Generate a heatmap for each track file.
        for path in tqdm.tqdm(data_paths, desc=f"Analyzing {strain}"):
            # Extract the cell number from the path.
            cell_number = int(re.findall(r'cell(\d+)', path)[0])

            image_data, voxel_size = load_image_data(strain, cell_number)
            if strain in default_voxel_sizes.keys():
                voxel_size = default_voxel_sizes[strain]
            pole_positions, time_indices = load_spindle_positions(path)

            # Build the kymograph for this cell.
            if SAVE_INDIVIDUAL_KYMOGRAPHS:
                os.makedirs(f'./dynamic_kymographs/individual kymos {strain}', exist_ok=True)
                kymograph = generate_single_dynamic_kymograph(
                    pole_positions,
                    image_data,
                    voxel_size,
                    time_indices,
                    save_path=f'./dynamic_kymographs/individual kymos {strain}/{strain}_cell{cell_number}_dynamic_kymograph.svg'
                )
                kymographs.append(kymograph)
            else:
                kymograph = generate_single_dynamic_kymograph(pole_positions, image_data, voxel_size, time_indices)
                kymographs.append(kymograph)


        # Combine the kymographs into a single kymograph.
        final_kymograph = avg_kymographs(kymographs)

        # Normalize the kymograph to (0, 1) for visualization.
        final_kymograph = final_kymograph - np.min(final_kymograph)
        final_kymograph = final_kymograph / np.max(final_kymograph)

        # Save the heatmap as a png file.
        plt.imsave(
            f'./dynamic_kymographs/{strain}_dynamic_kymograph.svg',
            np.transpose(final_kymograph),
            dpi=500,
            cmap=CMAP
        )

        # print(f"Finished generating kymograph for strain: {strain}")

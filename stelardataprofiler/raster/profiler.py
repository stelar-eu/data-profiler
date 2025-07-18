import os
from datetime import datetime
from pathlib import Path
import pandas as pd
import dateutil.parser
from typing import Union, Tuple, List
import rasterio as rio
from rasterio.warp import transform_bounds
from rasterio.transform import from_origin
from rasterio.io import MemoryFile
from pyproj import CRS
from shapely.geometry import box
import uuid
from ..utils import write_to_json


# -------------- RASTER --------------#
# ----------- SINGLE IMAGE -----------#
def profile_single_raster(my_file_path: str) -> dict:
    """
    This method performs profiling and generates a profiling dictionary for an image file that exists in the given path.

    :param my_file_path: the path to an image file.
    :type my_file_path: str
    :return: A dict which contains the results of the profiler for the image.
    :rtype: dict

    """
    if os.path.isdir(my_file_path):
        print('The input is not a file!')
        return dict()

    filename = get_filename(my_file_path)
    profile_dict = {
        'analysis': {
            'title': 'Profiling Report',
            'date_start': '',
            'date_end': '',
            'duration': '',
            'filenames': [filename]
        },
        'table': {
            'profiler_type': 'Raster',
            'byte_size': 0,
            'n_of_imgs': 1,
            'avg_width': 0.0,
            'avg_height': 0.0,
        },
        'variables': []
    }

    # Start time
    now = datetime.now()
    start_string = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    profile_dict['analysis']['date_start'] = start_string

    # File size
    profile_dict['table']['byte_size'] = os.path.getsize(my_file_path)

    # Create image dictionary
    img_dict = {
        'name': '',
        'type': 'Raster',
        'crs': '',
        'spatial_coverage': '',
        'spatial_resolution': {
            'pixel_size_x': 0,
            'pixel_size_y': 0
        },
        'no_data_value': '',
        'format': ''
    }

    # Read image
    img = rio.open(my_file_path)

    # find image name
    name = Path(my_file_path).stem
    img_dict['name'] = name

    # find general image data
    img_dict.update(img.meta)

    # making transform JSON-serializable
    img_dict['transform'] = list(img_dict['transform'])

    profile_dict['table']['avg_width'] = img_dict['width']
    profile_dict['table']['avg_height'] = img_dict['height']

    # change nodata and driver keys
    img_dict['no_data_value'] = img_dict['nodata']
    del img_dict['nodata']

    img_dict['format'] = img_dict['driver']
    del img_dict['driver']

    # find tags
    img_dict['tags'] = []

    for k, v in img.tags().items():
        tag_dict = {
            'key': k,
            'value': v
        }

        img_dict['tags'].append(tag_dict)

    # change crs format
    if img.crs is not None:
        crs_list = CRS.from_string(str(img_dict['crs']))
        img_dict['crs'] = 'EPSG:' + str(crs_list.to_epsg())
    else:
        img_dict['crs'] = 'EPSG:4326'

    # calculate spatial resolution
    pixelSizeX, pixelSizeY = img.res
    img_dict['spatial_resolution']['pixel_size_x'] = pixelSizeX
    img_dict['spatial_resolution']['pixel_size_y'] = pixelSizeY

    # calculate spatial coverage
    # Bounding box (in the original CRS)
    bounds = img.bounds

    xmin, ymin, xmax, ymax = transform_bounds(CRS.from_string(img_dict['crs']), CRS.from_epsg(4326), *bounds)

    geom = box(xmin, ymin, xmax, ymax)
    img_dict['spatial_coverage'] = geom.wkt

    img_dict['bands'] = []
    # statistics for each band
    for band in range(1, img.count + 1):
        band_data = img.read(band).reshape(1, img.meta['width'] * img.meta['height'])[0].T

        # find band name
        if list(img.descriptions):
            band_name = img.descriptions[band - 1]
            if band_name is None:
                band_name = 'undefined'
        else:
            band_name = 'undefined'

        # find band statistics
        s = pd.Series(band_data)
        stats = s.describe(percentiles=[.10, .25, .75, .90])

        band_dict = {
            'uuid': str(uuid.uuid4()),
            'name': band_name,
            'count': stats[0],
            'min': stats[3],
            'max': stats[9],
            'average': stats[1],
            'stddev': stats[2],
            'median': stats[6],
            'kurtosis': s.kurtosis(),
            'skewness': s.skew(),
            'variance': s.var(),
            'percentile10': stats[4],
            'percentile25': stats[5],
            'percentile75': stats[7],
            'percentile90': stats[8],
        }

        img_dict['bands'].append(band_dict)

    profile_dict['variables'].append(img_dict)

    # End time
    now = datetime.now()
    end_string = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    profile_dict['analysis']['date_end'] = end_string

    # Time Difference
    profile_dict['analysis']['duration'] = str(
        dateutil.parser.parse(profile_dict['analysis']['date_end']) - dateutil.parser.parse(
            profile_dict['analysis']['date_start']))

    return profile_dict


# ----------- MULTIPLE IMAGES -----------#
# noinspection PyTypedDict
def profile_multiple_rasters(my_file_paths: List[str]) -> dict:
    """
    This method performs profiling and generates a profiling dictionary for the image files that exist in the given folder path.

    :param my_folder_path: list of paths to image files.
    :type my_folder_path: List[str]
    :return: A dict which contains the results of the profiler for the images.
    :rtype: dict

    """

    profile_dict = {
        'analysis': {
            'title': 'Profiling Report',
            'date_start': '',
            'date_end': '',
            'duration': '',
            'filenames': []
        },
        'table': {
            'profiler_type': 'Raster',
            'byte_size': 0,
            'n_of_imgs': 0,
            'avg_width': 0,
            'avg_height': 0,
            'combined_band_stats': []
        },
        'variables': []
    }

    # in dictionary if same band name in more than one images
    band_images = dict()

    # Start time
    now = datetime.now()
    start_string = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    profile_dict['analysis']['date_start'] = start_string

    for image in my_file_paths:
        filename = get_filename(image)

        profile_dict['analysis']['filenames'].append(filename)

        # Files size
        profile_dict['table']['byte_size'] += os.path.getsize(image)

        # Increase the number of images
        profile_dict['table']['n_of_imgs'] += 1

        # Create image dictionary
        img_dict = {
            'name': '',
            'type': 'Raster',
            'crs': '',
            'spatial_coverage': '',
            'spatial_resolution': {
                'pixel_size_x': 0,
                'pixel_size_y': 0
            },
            'no_data_value': '',
            'format': ''
        }

        # Read image
        img = rio.open(image)

        # find image name
        name = Path(image).stem
        img_dict['name'] = name

        # find general image data
        img_dict.update(img.meta)

        # making transform JSON-serializable
        img_dict['transform'] = list(img_dict['transform'])

        profile_dict['table']['avg_width'] += img_dict['width']
        profile_dict['table']['avg_height'] += img_dict['height']

        # change nodata and driver keys
        img_dict['no_data_value'] = img_dict['nodata']
        del img_dict['nodata']

        img_dict['format'] = img_dict['driver']
        del img_dict['driver']

        # find tags
        img_dict['tags'] = []

        for k, v in img.tags().items():
            tag_dict = {
                'key': k,
                'value': v
            }

            img_dict['tags'].append(tag_dict)

        # change crs format
        if img.crs is not None:
            crs_list = CRS.from_string(str(img_dict['crs']))
            img_dict['crs'] = 'EPSG:' + str(crs_list.to_epsg())
        else:
            img_dict['crs'] = 'EPSG:4326'

        # calculate spatial resolution
        pixelSizeX, pixelSizeY = img.res
        img_dict['spatial_resolution']['pixel_size_x'] = pixelSizeX
        img_dict['spatial_resolution']['pixel_size_y'] = pixelSizeY

        # calculate spatial coverage
        # Bounding box (in the original CRS)
        bounds = img.bounds

        xmin, ymin, xmax, ymax = transform_bounds(CRS.from_string(img_dict['crs']), CRS.from_epsg(4326), *bounds)

        geom = box(xmin, ymin, xmax, ymax)
        img_dict['spatial_coverage'] = geom.wkt

        img_dict['bands'] = []
        # statistics for each band
        for band in range(1, img.count + 1):
            band_data = img.read(band).reshape(1, img.meta['width'] * img.meta['height'])[0].T

            # find band name
            band_name = 'undefined'
            if list(img.descriptions):
                band_name = img.descriptions[band - 1]
                if band_name is None:
                    band_name = 'undefined'
            else:
                band_name = 'undefined'

            # find band statistics
            s = pd.Series(band_data)
            stats = s.describe(percentiles=[.10, .25, .75, .90])

            band_dict = {
                'uuid': str(uuid.uuid4()),
                'name': band_name,
                'count': stats[0],
                'min': stats[3],
                'max': stats[9],
                'average': stats[1],
                'stddev': stats[2],
                'median': stats[6],
                'kurtosis': s.kurtosis(),
                'skewness': s.skew(),
                'variance': s.var(),
                'percentile10': stats[4],
                'percentile25': stats[5],
                'percentile75': stats[7],
                'percentile90': stats[8],
            }

            img_dict['bands'].append(band_dict)

            if band_name != 'undefined':
                if band_name not in band_images:
                    band_images[band_name] = [img_dict['name']]
                else:
                    band_images[band_name].append(img_dict['name'])

        profile_dict['variables'].append(img_dict)

    # calculate combined_band_stats
    for k, v in band_images.items():
        if len(v) > 1:
            combined_band_dict = {
                'name': k,
                'n_of_imgs': len(v),
                'img_names': v,
                'count': 0,
                'min': math.inf,
                'average': 0,
                'max': -math.inf,
                'variance': 0
            }

            for image in profile_dict['variables']:
                if image['name'] in v:
                    for band in image['bands']:
                        if band['name'] == k:
                            combined_band_dict['count'] += band['count']
                            combined_band_dict['average'] += band['average'] * band['count']

                            if band['min'] < combined_band_dict['min']:
                                combined_band_dict['min'] = band['min']

                            if band['max'] > combined_band_dict['max']:
                                combined_band_dict['max'] = band['max']

                            break

            combined_band_dict['average'] = combined_band_dict['average'] / combined_band_dict['count']

            # calculate combined_variance
            # comb_var = (n*std1 + n*d_sqrt1 + m*std2 + m*d_sqrt2 + k*std3 + k*d_sqrt3)/ n + m + k
            for image in profile_dict['variables']:
                if image['name'] in v:
                    for band in image['bands']:
                        if band['name'] == k:
                            count = band['count']
                            std = band['stddev']
                            mean = band['average']
                            comb_mean = combined_band_dict['average']
                            d_sqrt = (mean - comb_mean) * (mean - comb_mean)

                            combined_band_dict['variance'] += count * std + count * d_sqrt

                            break

            combined_band_dict['variance'] = combined_band_dict['variance'] / combined_band_dict['count']

            profile_dict['table']['combined_band_stats'].append(combined_band_dict)

    # fill general image folder data
    profile_dict['table']['avg_width'] = float(profile_dict['table']['avg_width']) / float(
        profile_dict['table']['n_of_imgs'])
    profile_dict['table']['avg_height'] = float(profile_dict['table']['avg_height']) / float(
        profile_dict['table']['n_of_imgs'])

    # End time
    now = datetime.now()
    end_string = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    profile_dict['analysis']['date_end'] = end_string

    # Time Difference
    profile_dict['analysis']['duration'] = str(
        dateutil.parser.parse(profile_dict['analysis']['date_end']) - dateutil.parser.parse(
            profile_dict['analysis']['date_start']))

    return profile_dict


# ----------- MAIN FUNCTION ----------#
def profile_raster_with_config(config: dict) -> None:
    """
    This method performs profiling on raster data and writes the resulting profile dictionary based on a configuration dictionary.

    :param config: a dictionary with all configuration settings.
    :type config: dict
    :return: None.
    :rtype: None

    """
    # input file path(s)
    input_file_paths = config['input']['files']

    if isinstance(input_file_paths, list):
        if len(input_file_paths) == 1:
            my_path = os.path.abspath(input_file_paths[0])
        else:
            my_path = []
            for path in input_file_paths:
                my_path.append(os.path.abspath(input_file_paths))
    elif isinstance(input_file_paths, str) and os.path.isfile(os.path.abspath(input_file_paths)):
        my_path = os.path.abspath(input_file_paths)
    else:
        raise ValueError(f"Invalid input: {input_file_paths} must be a valid file path or list of file paths")

    # output file path
    output_json_path = os.path.abspath(config['output']['json'])

    # Run raster profile
    profile_dict = profile_raster(my_path=my_path)

    # Write resulting profile dictionary
    write_to_json(profile_dict, output_json_path)


def profile_raster(my_path: Union[str, List[str]]) -> dict:
    """
    This method performs profiling and generates a profiling dictionary for either a single image or many images.

    :param my_path: either the path to an image file or a list of paths to image files.
    :type my_path: Union[str, List[str]]
    :return: A dict which contains the results of the profiler for the image or images.
    :rtype: dict

    """
    if isinstance(my_path, list):
        # Handle list of paths
        return profile_multiple_rasters(my_path)
    elif isinstance(my_path, str) and os.path.isfile(my_path):
        # Handle single file path
        return profile_single_raster(my_path)
    else:
        raise ValueError(f"Invalid input: {my_path} must be a valid file path or list of file paths")


# ------ VISTA (RHD, RAS FILES) ------#
def profile_vista_rasters_with_config(config: dict) -> None:
    """
    This method performs profiling on ras data and writes the resulting profile dictionary based on a configuration dictionary.

    :param config: a dictionary with all configuration settings.
    :type config: dict
    :return: None.
    :rtype: None

    """
    # 2 input files (ras, rhd)
    my_rhd_file_path = os.path.abspath(config['input']['rhd_file'])
    my_ras_file_path = os.path.abspath(config['input']['ras_file'])

    # output file path
    output_json_path = os.path.abspath(config['output']['json'])

    # Run raster profile
    profile_dict = profile_vista_rasters(rhd_datapath=my_rhd_file_path, ras_datapath=my_ras_file_path)

    # Write resulting profile dictionary
    write_to_json(profile_dict, output_json_path)


def profile_vista_rasters(rhd_datapath: str, ras_datapath: str):
    """
    This method performs profiling and generates a profiling dictionary for a given ras file
    that exists in the given path using the contents of a rhd file that exists in the given path.

    :param rhd_datapath: the path to a rhd file.
    :type rhd_datapath: str
    :param ras_datapath: the path to a ras file.
    :type ras_datapath: str
    :return: A dict which contains the results of the profiler for the ras.
    :rtype: dict

    """

    def __read_image_rhd(rhd_datapath: str):
        with open(rhd_datapath, 'r') as f:
            lines = f.readlines()
            vista_data_type = int(lines[0])
            n_of_LAI = int(lines[1])
            split_third_row = " ".join(lines[2].split()).split(' ')
            columns = int(split_third_row[0])
            rows = int(split_third_row[1])
            split_fourth_row = " ".join(lines[3].split()).split(' ')
            resolution = float(split_fourth_row[0])
            upper_left_corner_x = float(split_fourth_row[1])
            upper_left_corner_y = float(split_fourth_row[2])
            UTM_x = float(split_fourth_row[3])
            UTM_y = float(split_fourth_row[4])
            UTM_zone = str(split_fourth_row[5])
            LAI_images = {'vista_data_type': vista_data_type, 'resolution': resolution,
                          'upper_left_corner_x': upper_left_corner_x, 'upper_left_corner_y': upper_left_corner_y,
                          'rows': rows, 'columns': columns, 'UTM_x': UTM_x, 'UTM_y': UTM_y, 'UTM_zone': UTM_zone}
            count_LAI_images = 0
            LAI_images['images'] = {}
            for value_LAI in range(5, n_of_LAI + 5):
                ras_file_name = rhd_datapath.split('/')[-1].split('.')[0]
                img_name = ras_file_name + '_' + str(count_LAI_images)
                prev_img_name = ras_file_name + '_' + str(count_LAI_images - 1)
                split_row = " ".join(lines[value_LAI].split()).split(' ')
                LAI_images['images'][img_name] = {}
                img_bytes = int(split_row[0])
                LAI_images['images'][img_name]['bytes'] = img_bytes
                LAI_images['images'][img_name]['date'] = datetime.strptime(
                    split_row[3] + ' ' + split_row[2] + ' ' + split_row[1], '%d %m %Y').date()

                record_length = img_bytes * columns
                LAI_images['images'][img_name]['record_length_bytes'] = record_length
                if count_LAI_images == 0:
                    LAI_images['images'][img_name]['image_start_pos_bytes'] = 0
                else:
                    LAI_images['images'][img_name]['image_start_pos_bytes'] = LAI_images['images'][prev_img_name][
                                                                                  'image_start_pos_bytes'] + ((
                                                                                                                      record_length / img_bytes) * rows)
                count_LAI_images += 1

            return LAI_images

    ras_dict = __read_image_rhd(rhd_datapath)

    profile_dict = {
        'analysis': {
            'title': 'Profiling Report',
            'date_start': '',
            'date_end': '',
            'duration': '',
            'filenames': [rhd_datapath,
                          ras_datapath]
        },
        'table': {
            'profiler_type': 'Vista_Raster',
            'byte_size': 0,
            'n_of_imgs': len(ras_dict['images']),
            'avg_width': 0.0,
            'avg_height': 0.0,
            'combined_bands': []
        },
        'variables': []
    }

    # initialize .ras NODATA value counts
    ras_zero_count = 0
    ras_missing_count = 0
    ras_forest_count = 0
    ras_urban_count = 0
    ras_water_count = 0
    ras_snow_count = 0
    ras_cloud_shadow_buffer_count = 0
    ras_cloud_shadow_count = 0
    ras_cloud_buffer_count = 0
    ras_cirrus_clouds_count = 0
    ras_clouds_count = 0

    __lai_f = lambda x: float(str(x)) / 1000 if (x > 0) else x
    # __lai_f = lambda x: float(str(x)[:-4])/40.0  if(x > 99) else ( x if(x < 0) else -999)

    # Start time
    now = datetime.now()
    start_string = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    profile_dict['analysis']['date_start'] = start_string

    img_names = []
    imgs = []
    lai_in_imgs = []
    with open(ras_datapath, 'r+') as f:
        ras_file_name = ras_datapath.split('/')[-1].split('.')[0]
        if ras_dict['vista_data_type'] == 7:
            ras_file_array = np.fromfile(f, dtype=np.int16).astype(float)
            ras_file_array[np.where(ras_file_array > 0)] = list(
                map(__lai_f, ras_file_array[np.where(ras_file_array > 0)]))
            n_of_imgs = len(ras_dict['images'])

            for n_img in range(0, n_of_imgs):

                # Create image dictionary
                img_dict = {
                    'name': '',
                    'type': 'Raster',
                    'crs': '',
                    'date': '',
                    'spatial_coverage': '',
                    'spatial_resolution': {
                        'pixel_size_x': 0,
                        'pixel_size_y': 0
                    },
                    'no_data_value': '',
                    'format': ''
                }

                img_name = ras_file_name + '_' + str(n_img)
                img_names.append(img_name)

                # image name
                img_dict['name'] = img_name

                next_img_name = ras_file_name + '_' + str(n_img + 1)
                if n_img == n_of_imgs - 1:
                    start_pos = int(ras_dict['images'][img_name]['image_start_pos_bytes'])
                    end_pos = len(ras_file_array)
                else:
                    start_pos = int(ras_dict['images'][img_name]['image_start_pos_bytes'])
                    end_pos = int(ras_dict['images'][next_img_name]['image_start_pos_bytes'])

                # data of the image
                img_data = ras_file_array[start_pos:end_pos]
                img_data = img_data.reshape((ras_dict['rows'], ras_dict['columns']))

                # Find Image General Data
                upper_left_corner_x = ras_dict['upper_left_corner_x']
                upper_left_corner_y = ras_dict['upper_left_corner_y']
                x_res = ras_dict['resolution']
                y_res = ras_dict['resolution']
                transform = from_origin(upper_left_corner_x, upper_left_corner_y, x_res, y_res)

                # create in-memory rasterio image
                mem_file = MemoryFile()

                with mem_file.open(driver='GTiff', height=ras_dict['rows'],
                                   width=ras_dict['columns'], count=1,
                                   dtype=str(ras_file_array.dtype), crs='+proj=utm +zone=' + str(ras_dict['UTM_zone']),
                                   transform=transform) as img:

                    img.update_tags(date=ras_dict['images'][img_name]['date'])

                    # image general metadata
                    img_dict.update(img.meta)

                    # image size
                    profile_dict['table']['byte_size'] += img_dict['width'] * img_dict['height'] * 4

                    # image date
                    img_dict['date'] = ras_dict['images'][img_name]['date'].strftime("%d.%m.%Y")

                    # making transform JSON-serializable
                    img_dict['transform'] = list(img_dict['transform'])

                    profile_dict['table']['avg_width'] += img_dict['width']
                    profile_dict['table']['avg_height'] += img_dict['height']

                    # change nodata and driver keys
                    img_dict['no_data_value'] = img_dict['nodata']
                    del img_dict['nodata']

                    img_dict['format'] = img_dict['driver']
                    del img_dict['driver']

                    # change crs format
                    if img.crs is not None:
                        crs_list = CRS.from_string(str(img_dict['crs']))
                        img_dict['crs'] = 'EPSG:' + str(crs_list.to_epsg())
                    else:
                        img_dict['crs'] = 'EPSG:4326'

                    # calculate spatial resolution
                    pixelSizeX, pixelSizeY = img.res
                    img_dict['spatial_resolution']['pixel_size_x'] = pixelSizeX
                    img_dict['spatial_resolution']['pixel_size_y'] = pixelSizeY

                    # calculate spatial coverage
                    # Bounding box (in the original CRS)
                    bounds = img.bounds

                    xmin, ymin, xmax, ymax = transform_bounds(CRS.from_string(img_dict['crs']), CRS.from_epsg(4326),
                                                              *bounds)

                    geom = box(xmin, ymin, xmax, ymax)

                    img_dict['spatial_coverage'] = geom.wkt

                    img.close()

                # statistics for LAI band
                img_dict['bands'] = []
                s = pd.Series(img_data[np.where(img_data > 0)])
                stats = s.describe(percentiles=[.10, .25, .75, .90])

                band_uuid = str(uuid.uuid4())

                band_dict = {
                    'uuid': band_uuid,
                    'name': 'LAI',
                    'count': stats[0],
                    'min': stats[3],
                    'max': stats[9],
                    'average': stats[1],
                    'stddev': stats[2],
                    'median': stats[6],
                    'kurtosis': s.kurtosis(),
                    'skewness': s.skew(),
                    'variance': s.var(),
                    'percentile10': stats[4],
                    'percentile25': stats[5],
                    'percentile75': stats[7],
                    'percentile90': stats[8],
                    'no_data_distribution': []
                }

                # percentages of no_data values
                img_no_data = img_data[np.where(img_data < 0)]
                width = img_dict['width']
                height = img_dict['height']

                missing_count = np.count_nonzero(img_no_data == -999)
                forest_count = np.count_nonzero(img_no_data == -961)
                urban_count = np.count_nonzero(img_no_data == -950)
                water_count = np.count_nonzero(img_no_data == -940)
                snow_count = np.count_nonzero(img_no_data == -930)
                cloud_shadow_buffer_count = np.count_nonzero(img_no_data == -923)
                cloud_shadow_count = np.count_nonzero(img_no_data == -920)
                cloud_buffer_count = np.count_nonzero(img_no_data == -913)
                cirrus_clouds_count = np.count_nonzero(img_no_data == -911)
                clouds_count = np.count_nonzero(img_no_data == -910)

                img_zeros = img_data[np.where(img_data == 0)]
                zero_count = img_zeros.size

                # add NODATA value counts to the .ras NODATA value counts
                ras_missing_count += missing_count
                ras_forest_count += forest_count
                ras_urban_count += urban_count
                ras_water_count += water_count
                ras_snow_count += snow_count
                ras_cloud_shadow_buffer_count += cloud_shadow_buffer_count
                ras_cloud_shadow_count += cloud_shadow_count
                ras_cloud_buffer_count += cloud_buffer_count
                ras_cirrus_clouds_count += cirrus_clouds_count
                ras_clouds_count += clouds_count

                # add zero value counts to the .ras zero value counts
                ras_zero_count += zero_count

                no_data_dict = {
                    'LAI': (band_dict['count'] / (width * height)) * 100,
                    'missing': (missing_count / (width * height)) * 100,
                    'forest': (forest_count / (width * height)) * 100,
                    'urban': (urban_count / (width * height)) * 100,
                    'water': (water_count / (width * height)) * 100,
                    'snow': (snow_count / (width * height)) * 100,
                    'cloud_shadow_buffer': (cloud_shadow_buffer_count / (width * height)) * 100,
                    'cloud_shadow': (cloud_shadow_count / (width * height)) * 100,
                    'cloud_buffer': (cloud_buffer_count / (width * height)) * 100,
                    'cirrus_clouds': (cirrus_clouds_count / (width * height)) * 100,
                    'clouds': (clouds_count / (width * height)) * 100,
                    'zeros': (zero_count / (width * height)) * 100
                }

                for k, v in no_data_dict.items():
                    band_dict['no_data_distribution'].append(
                        {'uuid': band_uuid, 'value': k, 'percentage': v}
                    )

                    if k == 'LAI':
                        imgs.append({'raster': img_dict['name'],
                                     'date': img_dict['date'],
                                     'percentage': no_data_dict['LAI']})

                        lai_in_imgs.append(no_data_dict['LAI'])

                img_dict['bands'].append(band_dict)

                profile_dict['variables'].append(img_dict)

            # calculate combined stats
            combined_band_stats_dict = {
                'name': 'LAI',
                'n_of_imgs': profile_dict['table']['n_of_imgs'],
                'img_names': img_names,
                'imgs': imgs,
                'count': 0,
                'min': math.inf,
                'average': 0,
                'max': -math.inf,
                'variance': 0,
                'no_data_distribution': [],
                'lai_distribution': {}
            }

            # calculate LAI numeric distribution for the images of the .ras
            s = pd.Series(lai_in_imgs)
            stats = s.describe(percentiles=[.10, .25, .75, .90])

            lai_dict = {
                'name': 'LAI',
                'count': stats[0],
                'min': stats[3],
                'max': stats[9],
                'average': stats[1],
                'stddev': stats[2],
                'median': stats[6],
                'kurtosis': s.kurtosis(),
                'skewness': s.skew(),
                'variance': s.var(),
                'percentile10': stats[4],
                'percentile25': stats[5],
                'percentile75': stats[7],
                'percentile90': stats[8]
            }

            combined_band_stats_dict['lai_distribution'] = lai_dict

            for image in profile_dict['variables']:
                lai_band = image['bands'][0]
                if lai_band['count'] != 0:
                    combined_band_stats_dict['count'] += lai_band['count']
                    combined_band_stats_dict['average'] += lai_band['average'] * lai_band['count']

                    if lai_band['min'] < combined_band_stats_dict['min']:
                        combined_band_stats_dict['min'] = lai_band['min']

                    if lai_band['max'] > combined_band_stats_dict['max']:
                        combined_band_stats_dict['max'] = lai_band['max']

            combined_band_stats_dict['average'] = combined_band_stats_dict['average'] / combined_band_stats_dict[
                'count']

            # calculate combined_variance
            # comb_var = (n*std1 + n*d_sqrt1 + m*std2 + m*d_sqrt2 + k*std3 + k*d_sqrt3)/ n + m + k
            for image in profile_dict['variables']:
                lai_band = image['bands'][0]
                if lai_band['count'] != 0:
                    count = lai_band['count']
                    std = lai_band['stddev']
                    mean = lai_band['average']
                    comb_mean = combined_band_stats_dict['average']
                    d_sqrt = (mean - comb_mean) * (mean - comb_mean)

                    combined_band_stats_dict['variance'] += count * std + count * d_sqrt

            combined_band_stats_dict['variance'] = combined_band_stats_dict['variance'] / combined_band_stats_dict[
                'count']

            # calculate no_data_distribution for LAI of the .ras
            width_all = profile_dict['table']['avg_width']
            height_all = profile_dict['table']['avg_height']

            no_data_dict = {
                'LAI': ((combined_band_stats_dict['count'] * n_of_imgs) / (width_all * height_all)) * 100,
                'missing': ((ras_missing_count * n_of_imgs) / (width_all * height_all)) * 100,
                'forest': ((ras_forest_count * n_of_imgs) / (width_all * height_all)) * 100,
                'urban': ((ras_urban_count * n_of_imgs) / (width_all * height_all)) * 100,
                'water': ((ras_water_count * n_of_imgs) / (width_all * height_all)) * 100,
                'snow': ((ras_snow_count * n_of_imgs) / (width_all * height_all)) * 100,
                'cloud_shadow_buffer': ((ras_cloud_shadow_buffer_count * n_of_imgs) / (width_all * height_all)) * 100,
                'cloud_shadow': ((ras_cloud_shadow_count * n_of_imgs) / (width_all * height_all)) * 100,
                'cloud_buffer': ((ras_cloud_buffer_count * n_of_imgs) / (width_all * height_all)) * 100,
                'cirrus_clouds': ((ras_cirrus_clouds_count * n_of_imgs) / (width_all * height_all)) * 100,
                'clouds': ((ras_clouds_count * n_of_imgs) / (width_all * height_all)) * 100,
                'zeros': ((ras_zero_count * n_of_imgs) / (width_all * height_all)) * 100
            }

            for k, v in no_data_dict.items():
                combined_band_stats_dict['no_data_distribution'].append(
                    {'name': 'LAI', 'value': k, 'percentage': v}
                )

            profile_dict['table']['combined_bands'].append(combined_band_stats_dict)

            # calculate avg_width and avg_height of .ras file
            profile_dict['table']['avg_width'] = profile_dict['table']['avg_width'] / profile_dict['table']['n_of_imgs']
            profile_dict['table']['avg_height'] = profile_dict['table']['avg_height'] / profile_dict['table'][
                'n_of_imgs']

    # End time
    now = datetime.now()
    end_string = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    profile_dict['analysis']['date_end'] = end_string

    # Time Difference
    profile_dict['analysis']['duration'] = str(
        dateutil.parser.parse(profile_dict['analysis']['date_end']) - dateutil.parser.parse(
            profile_dict['analysis']['date_start']))

    return profile_dict


def get_filename(path: str) -> Tuple[str, str]:
    """Helper to split filename and extension"""
    filename = os.path.basename(path)
    return filename

import argparse
import json
import cv2
import numpy as np
import os
import re

def overlay_images(background_img, overlay_img, img_id, output_folder_overlay, opacity=0.15):
    """
    Overlay one image with another using a specified opacity.
    """
    # Load the images
    if isinstance(background_img, str):
        background_img = cv2.imread(background_img)
    if isinstance(overlay_img, str):
        overlay_img = cv2.imread(overlay_img)

    overlayed_img = background_img.copy()
    overlay = cv2.addWeighted(overlay_img, opacity, overlayed_img, 1 - opacity, 0, overlayed_img)
    output_path = os.path.join(output_folder_overlay, f"overlay_{img_id}.png")
    cv2.imwrite(output_path, overlay)
    
    return overlay

def compute_masks(annotations_project_roots):
    completed = True
    not_completed = False

    def is_completed(val):    
        #gets welds flagged as completed, having no instances labelled, or having too few points to make a polygon in their labels
        #points < 3*2 ?
        b = (val['metadata']['status'] == "Completed" or (len(val['instances']) == 0 and val['metadata']['status'] == "Not started") or (len(val['instances']) == 1 and len(val['instances'][0]['points']) < 6))
        return b

    def is_not_completed(val):
        #gets any welds not flagged as completed
        return val['metadata']['status'] != "Completed"

    for annotations_project_root in annotations_project_roots:
        os.makedirs(os.path.join(annotations_project_root, "masks"), exist_ok=True)
        output_directory = os.path.join(annotations_project_root, "masks")
        with open(os.path.join(annotations_project_root, 'annotations_1.json')) as file:
            data = json.load(file)

        all_keys = [x for x in data.keys()]
        all_keys = all_keys[1:]

        if completed:
            all_keys = [x for x in all_keys if is_completed(data[x])]
        elif not_completed:
            all_keys = [x for x in all_keys if is_not_completed(data[x])]
        
        for key in all_keys:
            instances = data[key]['instances']
            img = cv2.imread(os.path.join(annotations_project_root, 'images', str(key)))
            mask_all = np.zeros_like(img[:,:,0], dtype=np.uint8)
            if instances is None or len(instances) == 0:
                instance_copy = np.zeros_like(img)
                class_id = 1
                background = np.ones_like(img, dtype=np.uint8) * 255
                background_image_output_filename = os.path.join(output_directory, key.split('.')[0] + '-' + str('bg') + '.png')
                cv2.imwrite(background_image_output_filename, background)
            else:
                background = np.ones_like(img, dtype=np.uint8) * 255
                for instance in instances:
                    instance_copy = np.zeros_like(img)

                    class_id = instance['classId']
                    points = instance['points']

                    image_output_filename = os.path.join(output_directory, key.split('.')[0] + '-' + str(class_id) + '.png')

                    if points:
                        points = np.array(points).reshape((-1, 2))
                        cv2.fillPoly(instance_copy, np.int32([points]), color=(255, 255, 255))
                        background[background == 255] -= instance_copy[background == 255]
                        cv2.fillPoly(mask_all, np.int32([points]), color=(class_id))
                    else:
                        cv2.imwrite(image_output_filename, instance_copy)
                background_image_output_filename = os.path.join(output_directory, key.split('.')[0] + '-' + str('bg') + '.png')
                cv2.imwrite(background_image_output_filename, background)
            
            #to get foreground mask, simply take the inverse of the background mask, so no need to do this separately



def compute_image_info(project_root, error_log_file, pattern_img, pattern_poly, file_format):
    all_images = []
    token = False
    
    for project_root in project_root:
        output_folder_overlay = f"{project_root}/overlayed_labels"
        os.makedirs(output_folder_overlay, exist_ok=True)
        img_list = []
        with open(os.path.join(project_root, 'annotations_1.json')) as file:
            data = json.load(file)
        dataset_name = project_root.split("_")[-4:]
        dataset_name = "_".join(dataset_name)
        dir_imgs = f"{project_root}/images"
        dir_masks = f"{project_root}/masks"
        directory_imgs = os.fsencode(dir_imgs)
        directory_masks = os.fsencode(dir_masks)
        list_dir_imgs = []

        for x in os.listdir(directory_imgs):
            #keep only files with image file format
            filename_img = os.fsdecode(x)
            if filename_img.lower().endswith(file_format):
                list_dir_imgs.append(filename_img)

        img_files = sorted(list_dir_imgs)
        poly_files = sorted(os.listdir(directory_masks))
        img_dict = {}
        poly_dict = {} 

        for filename_img in img_files:
            match = re.match(pattern_img, filename_img)
            if match:
                img_id = match.group(1)
                img_dict[img_id] = filename_img
        
        for filename_poly in poly_files:
            match = re.match(pattern_poly, os.fsdecode(filename_poly))
            if match:
                poly_id = match.group(1)
                poly_dict[poly_id] = filename_poly.decode()

        matched_pairs = [(img_dict[id], poly_dict[id]) for id in img_dict.keys() & poly_dict.keys()]
        #1
        with open(error_log_file, 'w') as error_file:
            error_file.write("Unmatched Image IDs:\n")
            error_file.write("\n".join(sorted(img_dict.keys() - poly_dict.keys())))
            error_file.write("\n\nUnmatched Mask IDs:\n")
            error_file.write("\n".join(sorted(poly_dict.keys() - img_dict.keys())))
        #2.
        for i, img_id in enumerate(img_dict.keys() & poly_dict.keys()):
            filepath_img = str(os.path.join(directory_imgs.decode(), img_dict[img_id]))
            filepath_poly = str(os.path.join(directory_masks.decode(), poly_dict[img_id]))
            overlayed_img = overlay_images(filepath_img, filepath_poly, img_id, output_folder_overlay)
            #overlayed_img = rgb_to_gray(overlayed_img)
            #3.
            filename_img = matched_pairs[i][0]
            filename_poly = matched_pairs[i][1]
            
            try:
                instance = data[filename_img]['instances']
            except KeyError:
                error_message = f'{dataset_name}: no annotation data for image {filename_img}'
                #print(error_message)
                error_file.write(error_message + '\n')
                instance = None
                token = True
            img = cv2.imread(filepath_img, cv2.IMREAD_GRAYSCALE)
            #get dimensions of image as tuple (height, width)
            h, w = img.shape
            #search in the annotation file 'annotation_1' for the image_id and get the meta information if token has not been set to True:
            if not token:
                label_status = data[filename_img]['metadata']['status']
                if label_status == "Not started" or instance is None or len(instance) == 0:
                    points = []
                    classId = None
                else:
                    points = instance[0]['points']
                    classId = instance[0]['classId']
            else:
                label_status = "No annotation data available"
                points = []
                classId = None
                
            token = False
            img_list.append([img, overlayed_img, filename_img, filename_poly, dataset_name, points, classId, label_status, (h,w)])

        all_images.append(img_list)

    return all_images

def main(args):
    masks_computed = args.masks_computed
    image_file_format = args.image_file_format
    base_path_lab = args.base_path_lab
    base_path_eur = args.base_path_eur
    lab_useable = args.lab_useable
    eur_useable = args.eur_useable
    annotations_project_roots_lab = [f'{base_path_lab}{x}' for x in lab_useable]
    annotations_project_roots_EUR = [f'{base_path_eur}{x}' for x in eur_useable]

    pattern_img = r"raw_image_(.*?)\.png"
    pattern_poly = r"raw_image_(.*?)-bg\.png"

    data_sources = args.data_sources
    data_folder = '../data'
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    for source in data_sources:
        if source == "eur":
            if not masks_computed:
                compute_masks(annotations_project_roots_EUR)
            error_log_file = os.path.join(f'../non_matches_Europe_Lab_Data.txt')
            all_images_data = compute_image_info(annotations_project_roots_EUR, error_log_file, pattern_img, pattern_poly, image_file_format)
            np.save(os.path.join(data_folder, 'all_images_data_EUR.npy'), np.array(all_images_data, dtype=object), allow_pickle=True)

            
        else:
            if not masks_computed:
                compute_masks(annotations_project_roots_lab)
            error_log_file = os.path.join(f'../non_matches_lab_rsw_segmentation.txt')
            all_images_data = compute_image_info(annotations_project_roots_lab, error_log_file, pattern_img, pattern_poly, image_file_format)
            np.save(os.path.join(data_folder, 'all_images_data_lab.npy'), np.array(all_images_data, dtype=object), allow_pickle=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for processing image and mask data')
    parser.add_argument('--masks_computed', type=bool, default=True, help='Flag indicating if masks are already computed')
    parser.add_argument('--image_file_format', type=str, default='.png', help='Image file format')
    parser.add_argument('--base_path_lab', type=str, required=True, help='Base path for lab data')
    parser.add_argument('--base_path_eur', type=str, required=True, help='Base path for EUR data')
    parser.add_argument('--lab_useable', type=int, nargs='+', required=True, help='List of usable lab indices')
    parser.add_argument('--eur_useable', type=int, nargs='+', required=True, help='List of usable EUR indices')
    parser.add_argument('--data_sources', type=str, nargs='+', default=["eur"], help='Data sources to process (e.g., ["eur", "lab"])')

    args = parser.parse_args()
    main(args)

#example CLI command to run the script:
#python DatasetCreation.py --masks_computed False --base_path_lab "/home/wk560263/Dokumente/SSQC-research/rsw/data/labdata/lab_rsw_segmentation_" --base_path_eur "/home/wk560263/Dokumente/SSQC-research/rsw/data/ford_tessonics_data/Ford_Tessonics_Europe_Lab_Data_" --lab_useable 1 2 3 4 5 8 9 10 11 --eur_useable 1 4 5 6 7 10 --data_sources eur lab

import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import StratifiedKFold, train_test_split
import matplotlib.pyplot as plt

class PrepareData:
    def __init__(self, dataset='EUR', random_state=42, n_splits=10):
        self.dataset = dataset
        self.random_state = random_state
        self.n_splits = n_splits
        self.data_list = self.load_data()
        self.dfs_img_has_mask = self.get_data_distribution()
        self.df_train_val, self.df_test, self.folds = self.divide_data()
        self.save_dataframes()
        self.trainval, self.trainval_names, self.trainval_labelmasks, self.trainval_idxs, self.test, self.test_names, self.test_labelmasks, self.test_idxs = self.build_dataset()
        self.print_fold_indices()

    def load_data(self):
        if self.dataset == 'EUR':
            npy_files = glob.glob("C:\\Users\Hendrik\\iCloudDrive\\Desktop\\WZL\\ssqc\\rsw_research\\code_data\\archive\\all_images_data_EUR.npy")
        elif self.dataset == 'lab':
            npy_files = glob.glob("C:\\Users\Hendrik\\iCloudDrive\\Desktop\\WZL\\ssqc\\rsw_research\\code_data\\archive\\all_images_data_lab.npy")
        else:
            npy_files = glob.glob("C:\\Users\Hendrik\\iCloudDrive\\Desktop\\WZL\\ssqc\\rsw_research\\code_data\\archive\\*.npy")

        npy_files = sorted(npy_files, key=lambda x: x.lower())

        data_list = []
        for npy_file in npy_files:
            data = np.load(npy_file, allow_pickle=True)
            data_list.append(data.tolist())
        
        return data_list

    def get_data_distribution(self):
        dfs = []
        for img_data in self.data_list:
            df_img_has_mask = self._get_data_distribution_for_img_data(img_data)
            dfs.append(df_img_has_mask)
        dfs_img_has_mask = pd.concat(dfs, ignore_index=True)
        return dfs_img_has_mask

    def _get_data_distribution_for_img_data(self, img_data):
        backslash_char = "\\"
        token = 0
        df_img_has_mask = pd.DataFrame(columns=['image_idx', 'dataset_idx', 'has_mask', 'dataset'])
        df_img_has_mask["has_mask"] = df_img_has_mask["has_mask"].astype(bool)

        try:
            dataset_name = img_data[0][0][4].split('\\')[1][:-2]
        except:
            dataset_name = img_data[0][0][4].split('\\')[0][:-2]

        for dataset_idx in range(len(img_data)):
            for image_idx in range(len(img_data[dataset_idx])):
                condition_result = len(img_data[dataset_idx][image_idx][5]) != 0
                dataset = img_data[dataset_idx][0][4].split('\\')[token]
                row_data = {'image_idx': image_idx, 'dataset_idx': dataset_idx, 'has_mask': condition_result, 'dataset': dataset.split(backslash_char)[-1]}

                df_img_has_mask = pd.concat([df_img_has_mask, pd.DataFrame([row_data])], ignore_index=True)

        mask_frac = []
        for dataset_idx in range(len(img_data)):
            cond = (df_img_has_mask['dataset_idx'] == dataset_idx) & (df_img_has_mask['has_mask'] == False)
            amt_no_mask = df_img_has_mask[cond].shape[0]
            amt_total = df_img_has_mask[df_img_has_mask['dataset_idx'] == dataset_idx].shape[0]
            neg_frac = amt_no_mask / amt_total

            dataset = img_data[dataset_idx][0][4].split('\\')[token]

            mask_frac.append((dataset.split(backslash_char)[-1], neg_frac))

        print(f"Fraction of images in datasets {dataset_name.split(backslash_char)[-1]} containing no welding nugget, per dataset: {mask_frac}")

        return df_img_has_mask

    def divide_data(self):
        train_val, test = train_test_split(self.dfs_img_has_mask, test_size=0.2, random_state=self.random_state, stratify=self.dfs_img_has_mask['has_mask'])

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        folds = [(train_idx, val_idx) for train_idx, val_idx in skf.split(train_val, train_val['has_mask'])]

        return train_val, test, folds
    
    def save_dataframes(self):
        self.df_train_val.to_csv('../train_val_data.csv', index=False)
        self.df_test.to_csv('../test_data.csv', index=False)

    def build_dataset(self):
        img_list_global = []
        img_names_global = []
        img_masks_global = []
        img_idx_global = []

        img_list_test_global = []
        img_names_test_global = []
        img_masks_test_global = []
        img_idx_test_global = []

        train_val_grouped = self.df_train_val['dataset'].str.split('_').str[0]
        test_grouped = self.df_test['dataset'].str.split('_').str[0]
        dataset_groups = sorted(train_val_grouped.unique(), key=lambda x: x.lower())
        
        for dataset, group in zip(self.data_list, dataset_groups):
            for num_set in range(len(dataset)):
                train_idxs = self.df_train_val[(self.df_train_val['dataset_idx'] == num_set) & (self.df_train_val['dataset'].str.startswith(group))]['image_idx']
                test_idxs = self.df_test[(self.df_test['dataset_idx'] == num_set) & (self.df_test['dataset'].str.startswith(group))]['image_idx']

                img_list = [dataset[num_set][ind][0] for ind in train_idxs]
                img_names = [dataset[num_set][ind][2] for ind in train_idxs]
                img_masks = [dataset[num_set][ind][5] for ind in train_idxs]

                img_list_test = [dataset[num_set][ind][0] for ind in test_idxs]
                img_names_test = [dataset[num_set][ind][2] for ind in test_idxs]
                img_masks_test = [dataset[num_set][ind][5] for ind in test_idxs]

                img_list_global.extend(img_list)
                img_names_global.extend(img_names)
                img_masks_global.extend(img_masks)
                img_idx_global.extend([(ind, num_set) for ind in train_idxs])

                img_list_test_global.extend(img_list_test)
                img_names_test_global.extend(img_names_test)
                img_masks_test_global.extend(img_masks_test)
                img_idx_test_global.extend([(ind, num_set) for ind in test_idxs])

        return img_list_global, img_names_global, img_masks_global, img_idx_global, img_list_test_global, img_names_test_global, img_masks_test_global, img_idx_test_global

    def print_fold_indices(self):
        for fold_num, (train_idx, val_idx) in enumerate(self.folds):
            train_fold = self.df_train_val.iloc[train_idx]
            val_fold = self.df_train_val.iloc[val_idx]

            first_10_train_indices = train_idx[:10]

            print(f"Fold {fold_num + 1}")
            print(f"Train fold shape: {train_fold.shape}")
            print(f"Validation fold shape: {val_fold.shape}")
            print(f"First 10 train indices: {first_10_train_indices}")

    # def display_img(self, img_data, dataset_idx, target_filename, show_annot_mask=1, color_space='gray', figsize=(16, 10)):
    #     def find_image_idx(img_data, dataset_idx, target_filename):
    #         for image_idx, image_info in enumerate(img_data[dataset_idx]):
    #             if image_info[2] == target_filename:
    #                 return image_idx
    #         return None
        
    #     def plot_img(img_data, dataset_idx, image_idx, show_annot_mask, color_space, figsize):
    #         plt.figure(figsize=figsize)
    #         if color_space == 'gray' or color_space == 'grey':
    #             plt.imshow(img_data[dataset_idx][image_idx][show_annot_mask], cmap='gray')
    #         elif color_space == 'rgb' and show_annot_mask == 1:
    #             plt.imshow(cv2.cvtColor(img_data[dataset_idx][image_idx][show_annot_mask], cv2.COLOR_LAB2RGB))

    #         plt.axis('off')
    #         dataset = img_data[dataset_idx][image_idx][4].split('/')[-1]
    #         plt.title(f"{img_data[dataset_idx][image_idx][2]}, {dataset}, Image {image_idx}, Color Space: {color_space}")
    #         plt.show()
        

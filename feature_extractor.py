"""
Feature Extraction Pipeline
Extracts CNN features + HU moments from images
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
from tqdm import tqdm

from models import SparseVGGNet


class HUMomentExtractor:
    """Extract HU moment features from images"""
    
    @staticmethod
    def order_moment(image, p, q, x_average=0.0, y_average=0.0):
        x_matrix = np.asmatrix(np.arange(1, image.shape[0] + 1)).T - x_average
        y_matrix = np.asmatrix(np.arange(1, image.shape[1] + 1)) - y_average
        xy_matrix = np.dot(np.power(x_matrix, p), np.power(y_matrix, q))
        sum_moment = np.sum(np.multiply(xy_matrix, image))
        return sum_moment
    
    @staticmethod
    def central_moment(image, p, q):
        m_00 = HUMomentExtractor.order_moment(image, 0, 0)
        if m_00 == 0:
            return 0
        x_average = HUMomentExtractor.order_moment(image, 1, 0) / m_00
        y_average = HUMomentExtractor.order_moment(image, 0, 1) / m_00
        return HUMomentExtractor.order_moment(image, p, q, x_average, y_average)
    
    @staticmethod
    def normalized_central_moment(image, p, q):
        mu_00 = HUMomentExtractor.central_moment(image, 0, 0)
        if mu_00 == 0:
            return 0
        eta = HUMomentExtractor.central_moment(image, p, q) / (mu_00 ** ((p + q) / 2))
        return eta
    
    @staticmethod
    def extract_hu_moments(image):
        """Extract 7 HU moment invariants"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        temp_image = image.T
        
        eta_20 = HUMomentExtractor.normalized_central_moment(temp_image, 2, 0)
        eta_02 = HUMomentExtractor.normalized_central_moment(temp_image, 0, 2)
        eta_11 = HUMomentExtractor.normalized_central_moment(temp_image, 1, 1)
        eta_30 = HUMomentExtractor.normalized_central_moment(temp_image, 3, 0)
        eta_03 = HUMomentExtractor.normalized_central_moment(temp_image, 0, 3)
        eta_21 = HUMomentExtractor.normalized_central_moment(temp_image, 2, 1)
        eta_12 = HUMomentExtractor.normalized_central_moment(temp_image, 1, 2)
        
        hu_moments = np.zeros(7)
        hu_moments[0] = eta_20 + eta_02
        hu_moments[1] = (eta_20 - eta_02) ** 2 + 4 * eta_11 ** 2
        hu_moments[2] = (eta_30 - 3 * eta_12) ** 2 + (3 * eta_21 - eta_03) ** 2
        hu_moments[3] = (eta_30 + eta_12) ** 2 + (eta_21 + eta_03) ** 2
        hu_moments[4] = (eta_30 - 3 * eta_12) * (eta_30 + eta_12) * ((eta_30 + eta_12) ** 2 - 3 * (eta_21 + eta_03) ** 2) + \
                        (3 * eta_21 - eta_03) * (eta_21 + eta_03) * (3 * (eta_30 + eta_12) ** 2 - (eta_21 + eta_03) ** 2)
        hu_moments[5] = (eta_20 - eta_02) * ((eta_30 + eta_12) ** 2 - (eta_21 + eta_03) ** 2) + \
                        4 * eta_11 * (eta_30 + eta_12) * (eta_21 + eta_03)
        hu_moments[6] = (3 * eta_21 - eta_03) * (eta_30 + eta_12) * ((eta_30 + eta_12) ** 2 - 3 * (eta_21 + eta_03) ** 2) + \
                        (3 * eta_12 - eta_30) * (eta_21 + eta_03) * (3 * (eta_30 + eta_12) ** 2 - (eta_21 + eta_03) ** 2)
        
        return hu_moments


class ImageDataset(Dataset):
    """Dataset for feature extraction"""
    
    def __init__(self, csv_path, image_column, label_columns, base_path='', 
                 image_size=(190, 400)):
        self.df = pd.read_csv(csv_path)
        self.image_column = image_column
        self.label_columns = label_columns
        self.base_path = base_path
        self.image_size = image_size
        
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.df.iloc[idx][self.image_column])
        
        try:
            image = Image.open(img_path).convert('L')
            image = image.resize(self.image_size[::-1])
            image_np = np.array(image)
            image_tensor = self.transform(Image.fromarray(image_np))
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image_np = np.zeros(self.image_size)
            image_tensor = torch.zeros(1, *self.image_size)
        
        # Get label
        labels = self.df.iloc[idx][self.label_columns].values.astype(np.float32)
        label = np.argmax(labels)
        
        return {
            'image': image_tensor,
            'image_np': image_np,
            'label': label
        }


class FeatureExtractor:
    """Extract features from CNN + HU moments"""
    
    def __init__(self, cnn_model_path, num_classes=12, batch_size=32):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.num_classes = num_classes
        
        # Load CNN model
        print(f"Loading CNN model from {cnn_model_path}...")
        self.cnn = SparseVGGNet(num_classes=num_classes, sparse=True)
        checkpoint = torch.load(cnn_model_path, map_location=self.device)
        self.cnn.load_state_dict(checkpoint['model_state_dict'])
        self.cnn = self.cnn.to(self.device)
        self.cnn.eval()
        print("✓ CNN model loaded\n")
        
        self.hu_extractor = HUMomentExtractor()
    
    def extract_from_csv(self, csv_path, image_column, label_columns, 
                        base_path='', image_size=(190, 400)):
        """Extract features from a single CSV"""
        
        dataset = ImageDataset(
            csv_path=csv_path,
            image_column=image_column,
            label_columns=label_columns,
            base_path=base_path,
            image_size=image_size
        )
        
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )
        
        all_feat_128 = []
        all_flatten = []
        all_hu = []
        all_labels = []
        
        print(f"Extracting features from {len(dataset)} images...")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Feature extraction"):
                images = batch['image'].to(self.device)
                _, flatten_feat, feat_128 = self.cnn(images)
                
                all_feat_128.append(feat_128.cpu().numpy())
                all_flatten.append(flatten_feat.cpu().numpy())
                
                # Extract HU moments
                hu_batch = []
                for img_np in batch['image_np']:
                    hu = self.hu_extractor.extract_hu_moments(img_np.numpy())
                    hu_batch.append(hu)
                all_hu.append(np.array(hu_batch))
                
                all_labels.extend(batch['label'].numpy())
        
        feat_128 = np.vstack(all_feat_128)
        flatten_feat = np.vstack(all_flatten)
        hu_moments = np.vstack(all_hu)
        labels = np.array(all_labels)
        
        # Combine: 128-dim CNN + 7-dim HU = 135-dim
        combined_features = np.hstack([feat_128, hu_moments])
        
        print(f"✓ Features extracted:")
        print(f"  128-dim features: {feat_128.shape}")
        print(f"  Combined features: {combined_features.shape}")
        print(f"  HU moments: {hu_moments.shape}\n")
        
        return {
            'features_128': feat_128,
            'features_combined': combined_features,
            'flatten_features': flatten_feat,
            'hu_moments': hu_moments,
            'labels': labels
        }
    
    def extract_from_multiple_csvs(self, train_csv, val_csv, test_csv,
                                   image_column, label_columns, base_path='',
                                   image_size=(190, 400), output_dir='./extracted_features'):
        """Extract features from train/val/test splits"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        splits = {
            'train': train_csv,
            'val': val_csv,
            'test': test_csv
        }
        
        all_results = {}
        
        for split_name, csv_path in splits.items():
            if csv_path and os.path.exists(csv_path):
                print(f"\n{'='*70}")
                print(f"Extracting {split_name.upper()} features")
                print(f"{'='*70}\n")
                
                results = self.extract_from_csv(
                    csv_path=csv_path,
                    image_column=image_column,
                    label_columns=label_columns,
                    base_path=base_path,
                    image_size=image_size
                )
                
                # Save features
                save_path = os.path.join(output_dir, f'{split_name}_features.npz')
                np.savez(
                    save_path,
                    features_128=results['features_128'],
                    features_combined=results['features_combined'],
                    flatten_features=results['flatten_features'],
                    hu_moments=results['hu_moments'],
                    labels=results['labels']
                )
                
                print(f"✓ {split_name.upper()} features saved to: {save_path}\n")
                all_results[split_name] = results
        
        return all_results

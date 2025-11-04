import os
import numpy as np
import torch
import cv2
from PIL import Image
import joblib
import pandas as pd
from tqdm import tqdm

from models import SparseVGGNet, MLFDClassifier
from feature_extractor import HUMomentExtractor


class FaultPredictor:
    """Unified predictor for all model types"""
    
    def __init__(self, cnn_model_path, mlfd_model_path=None, 
                 classifier_path=None, scaler_path=None,
                 label_columns=None):
        """
        Initialize predictor
        
        Args:
            cnn_model_path: Path to trained CNN model
            mlfd_model_path: Path to trained MLFD model (optional)
            classifier_path: Path to trained ML classifier (optional)
            scaler_path: Path to feature scaler (required for ML classifier)
            label_columns: List of label names
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_columns = label_columns
        
        # Load CNN
        print("Loading CNN model...")
        self.cnn = SparseVGGNet(num_classes=len(label_columns), sparse=True)
        checkpoint = torch.load(cnn_model_path, map_location=self.device)
        self.cnn.load_state_dict(checkpoint['model_state_dict'])
        self.cnn = self.cnn.to(self.device)
        self.cnn.eval()
        print("✓ CNN loaded\n")
        
        # Load MLFD if provided
        self.mlfd = None
        if mlfd_model_path and os.path.exists(mlfd_model_path):
            print("Loading MLFD model...")
            mlfd_data = np.load(mlfd_model_path)
            self.mlfd = MLFDClassifier(
                d=mlfd_data['W'].shape[0],
                l=mlfd_data['W'].shape[1],
                lambda1=float(mlfd_data['lambda1']),
                lambda2=float(mlfd_data['lambda2']),
                lambda3=float(mlfd_data['lambda3']),
                lambda4=float(mlfd_data['lambda4']),
                lambda5=float(mlfd_data['lambda5']),
                lambda6=float(mlfd_data['lambda6']),
                mu=float(mlfd_data['mu']),
                K=int(mlfd_data['K'])
            )
            self.mlfd.W = mlfd_data['W']
            self.mlfd.C = mlfd_data['C']
            self.mlfd.P = mlfd_data['P']
            self.mlfd.N = mlfd_data['N']
            print("✓ MLFD loaded\n")
        
        # Load ML classifier if provided
        self.classifier = None
        self.scaler = None
        if classifier_path and os.path.exists(classifier_path):
            print("Loading ML classifier...")
            self.classifier = joblib.load(classifier_path)
            if scaler_path and os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            print("✓ ML classifier loaded\n")
        
        self.hu_extractor = HUMomentExtractor()
    
    def preprocess_image(self, image_path, image_size=(190, 400)):
        """Preprocess single image"""
        image = Image.open(image_path).convert('L')
        image = image.resize(image_size[::-1])
        image_np = np.array(image)
        
        # Normalize for CNN
        image_tensor = torch.from_numpy(np.expand_dims(image_np, axis=0)).float()
        image_tensor = (image_tensor - 127.5) / 127.5
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        return image_tensor, image_np
    
    def extract_features(self, image_tensor, image_np):
        """Extract CNN + HU moment features"""
        with torch.no_grad():
            _, _, feat_128 = self.cnn(image_tensor)
            feat_128 = feat_128.cpu().numpy().flatten()
        
        hu_moments = self.hu_extractor.extract_hu_moments(image_np)
        combined_features = np.hstack([feat_128, hu_moments]).reshape(1, -1)
        
        return combined_features
    
    def predict_cnn(self, image_path):
        """Predict using CNN only"""
        image_tensor, _ = self.preprocess_image(image_path)
        
        with torch.no_grad():
            outputs, _, _ = self.cnn(image_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            predicted_class = np.argmax(probabilities)
        
        return {
            'predicted_class': self.label_columns[predicted_class],
            'predicted_index': int(predicted_class),
            'confidence': float(probabilities[predicted_class]),
            'all_probabilities': {self.label_columns[i]: float(prob) 
                                 for i, prob in enumerate(probabilities)}
        }
    
    def predict_mlfd(self, image_path):
        """Predict using MLFD"""
        if self.mlfd is None:
            raise ValueError("MLFD model not loaded")
        
        image_tensor, image_np = self.preprocess_image(image_path)
        features = self.extract_features(image_tensor, image_np)
        
        Y_pred, scores = self.mlfd.predict(features)
        probs = self.mlfd.predict_proba(features)
        
        # Convert multi-label to single label (highest probability)
        predicted_class = np.argmax(probs[0])
        
        return {
            'predicted_class': self.label_columns[predicted_class],
            'predicted_index': int(predicted_class),
            'confidence': float(probs[0][predicted_class]),
            'multi_label_predictions': Y_pred[0],
            'all_probabilities': {self.label_columns[i]: float(prob) 
                                 for i, prob in enumerate(probs[0])}
        }
    
    def predict_classifier(self, image_path):
        """Predict using ML classifier"""
        if self.classifier is None:
            raise ValueError("ML classifier not loaded")
        
        image_tensor, image_np = self.preprocess_image(image_path)
        features = self.extract_features(image_tensor, image_np)
        
        # Scale features
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        prediction = self.classifier.predict(features)[0]
        
        # Get probabilities if available
        if hasattr(self.classifier, 'predict_proba'):
            probabilities = self.classifier.predict_proba(features)[0]
            confidence = probabilities[int(prediction)]
            all_probs = {self.label_columns[i]: float(prob) 
                        for i, prob in enumerate(probabilities)}
        else:
            confidence = 1.0
            all_probs = None
        
        return {
            'predicted_class': self.label_columns[int(prediction)],
            'predicted_index': int(prediction),
            'confidence': float(confidence),
            'all_probabilities': all_probs
        }
    
    def predict_all(self, image_path):
        """Get predictions from all available models"""
        results = {}
        
        # CNN prediction
        results['cnn'] = self.predict_cnn(image_path)
        
        # MLFD prediction
        if self.mlfd is not None:
            results['mlfd'] = self.predict_mlfd(image_path)
        
        # ML classifier prediction
        if self.classifier is not None:
            results['classifier'] = self.predict_classifier(image_path)
        
        return results
    
    def predict_folder(self, folder_path, model_type='classifier', output_csv='predictions.csv'):
        """
        Predict all images in a folder
        
        Args:
            folder_path: Path to folder containing images
            model_type: 'cnn', 'mlfd', 'classifier', or 'all'
            output_csv: Output CSV filename
        """
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(exts)]
        
        print(f"\nProcessing {len(image_files)} images from {folder_path}...")
        
        results = []
        
        for img_file in tqdm(image_files, desc="Predicting"):
            img_path = os.path.join(folder_path, img_file)
            
            try:
                if model_type == 'all':
                    pred = self.predict_all(img_path)
                    result = {
                        'filename': img_path,
                        'cnn_prediction': pred['cnn']['predicted_class'],
                        'cnn_confidence': pred['cnn']['confidence']
                    }
                    if 'mlfd' in pred:
                        result['mlfd_prediction'] = pred['mlfd']['predicted_class']
                        result['mlfd_confidence'] = pred['mlfd']['confidence']
                    if 'classifier' in pred:
                        result['classifier_prediction'] = pred['classifier']['predicted_class']
                        result['classifier_confidence'] = pred['classifier']['confidence']
                elif model_type == 'cnn':
                    pred = self.predict_cnn(img_path)
                    result = {
                        'filename': img_path,
                        'predicted_class': pred['predicted_class'],
                        'confidence': pred['confidence']
                    }
                elif model_type == 'mlfd':
                    pred = self.predict_mlfd(img_path)
                    result = {
                        'filename': img_path,
                        'predicted_class': pred['predicted_class'],
                        'confidence': pred['confidence']
                    }
                elif model_type == 'classifier':
                    pred = self.predict_classifier(img_path)
                    result = {
                        'filename': img_path,
                        'predicted_class': pred['predicted_class'],
                        'confidence': pred['confidence']
                    }
                
                results.append(result)
                
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
        
        # Save to CSV
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"\n✓ Predictions saved to {output_csv}")
        
        return df


def main():
    """Example usage"""
    
    # Configuration
    CNN_MODEL = './trained_models/best_model.pth'
    MLFD_MODEL = './trained_mlfd/mlfd_model.npz'
    CLASSIFIER_MODEL = './trained_classifiers/random_forest.pkl'
    SCALER = './trained_classifiers/feature_scaler.pkl'
    
    LABEL_COLUMNS = [
        'Bottom_Tagging', 'Fluid_Pounding', 'Gas_Interference', 'Gas_Lock',
        'Good_Dynamograph', 'Polish_Rod_Tagging', 'Pump_Wear',
        'Standing_Valve_Leak', 'Standing_Valve_Sticky_Open', 'Stuck_Plunger',
        'Traveling_Valve_Leak', 'Traveling_Valve_Sticking'
    ]
    
    # Initialize predictor
    predictor = FaultPredictor(
        cnn_model_path=CNN_MODEL,
        mlfd_model_path=MLFD_MODEL,
        classifier_path=CLASSIFIER_MODEL,
        scaler_path=SCALER,
        label_columns=LABEL_COLUMNS
    )
    
    # Example 1: Predict single image
    print("\n" + "="*70)
    print("SINGLE IMAGE PREDICTION")
    print("="*70)
    
    # Change this to your image path
    test_image = './test_image.jpg'
    
    if os.path.exists(test_image):
        results = predictor.predict_all(test_image)
        
        print(f"\nPredictions for: {test_image}")
        print("-" * 70)
        
        for model_name, pred in results.items():
            print(f"\n{model_name.upper()}:")
            print(f"  Predicted: {pred['predicted_class']}")
            print(f"  Confidence: {pred['confidence']:.4f}")
    
    # Example 2: Predict folder
    print("\n" + "="*70)
    print("BATCH PREDICTION ON FOLDER")
    print("="*70)
    
    # Change this to your folder path
    test_folder = './test_images/'
    
    if os.path.exists(test_folder):
        df = predictor.predict_folder(
            folder_path=test_folder,
            model_type='classifier',  # or 'cnn', 'mlfd', 'all'
            output_csv='predictions.csv'
        )
        
        print(f"\nSample predictions:")
        print(df.head())


if __name__ == '__main__':
    main()

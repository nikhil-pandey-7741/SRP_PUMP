import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm




class SparseConvBlock(nn.Module):
    """Sparse convolution block with spectral normalization"""
    
    def __init__(self, in_channels, out_channels, sparse=True):
        super(SparseConvBlock, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        if sparse:
            conv = nn.utils.spectral_norm(conv)
        self.conv = conv
        self.batchnorm = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.batchnorm(x)
        return x


class SparseVGGNet(nn.Module):
    """Improved Sparse VGGNet with temperature scaling for feature extraction"""
    
    def __init__(self, num_classes=12, sparse=True):
        super(SparseVGGNet, self).__init__()
        self.sparse = sparse
        
        # Convolutional blocks
        self.conv1 = SparseConvBlock(1, 16, self.sparse)
        self.conv2 = SparseConvBlock(16, 32, self.sparse)
        self.conv3 = SparseConvBlock(32, 64, self.sparse)
        self.conv4 = SparseConvBlock(64, 128, self.sparse)
        
        # Reduced dropout from 0.5 to 0.3 for better test-time confidence
        self.dropout = nn.Dropout(0.3)
        
        # Temperature scaling parameter for probability calibration
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        
        # Fully connected layers
        self.flatten_size = 128 * 11 * 25  # 35200
        
        self.fc1 = nn.Linear(self.flatten_size, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, num_classes)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(2048)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(128)
    
    def forward(self, x):
        # Convolutional layers with max pooling
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        
        x = self.conv3(x)
        x = F.max_pool2d(x, 2, 2)
        
        x = self.conv4(x)
        x = F.max_pool2d(x, 2, 2)
        
        # Flatten for fully connected layers
        flatten_x = torch.flatten(x, 1)
        
        # Fully connected layers with dropout
        x = self.fc1(flatten_x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        feat_128 = F.relu(x)  # 128-dimensional features
        
        x = self.fc4(feat_128)
        
        # Apply temperature scaling for better calibrated probabilities
        x = x / self.temperature
        
        return x, flatten_x, feat_128




class MLFDClassifier:
    """
    Multi-Label Fault Diagnosis using label, feature, and instance correlations
    
    Exploits:
    - Label correlations (C matrix)
    - Feature correlations (M matrix via PCA)
    - Instance correlations (N matrix via k-NN)
    """
    
    def __init__(self, d, l, lambda1=0.1, lambda2=0.1, lambda3=0.01,
                 lambda4=0.1, lambda5=0.1, lambda6=0.1, mu=1.0, K=5):
        """
        Args:
            d: Feature dimension
            l: Number of labels
            lambda1-lambda6: Regularization parameters
            mu: ADMM penalty parameter
            K: Number of nearest neighbors for instance correlation
        """
        self.d = d
        self.l = l
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        self.lambda5 = lambda5
        self.lambda6 = lambda6
        self.mu = mu
        self.K = K
        
        # Model parameters
        self.W = np.random.randn(d, l) * 0.01  # Feature-to-label mapping
        self.C = np.eye(l)  # Label correlation matrix
        self.Q = np.zeros((d, l))  # Auxiliary variable for ADMM
        self.Lambda = np.zeros((d, l))  # Lagrange multipliers
        self.P = None  # Feature correlation (PCA projection)
        self.N = None  # Instance correlation (k-NN)
    
    def compute_feature_correlation(self, X):
        """Compute feature correlation matrix M using PCA"""
        print("Computing feature correlation matrix...")
        
        # Normalize features
        X_normalized = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
        
        # Compute covariance
        cov = np.cov(X_normalized.T)
        
        # Eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(cov)
        
        # Sort by eigenvalues (descending)
        idx = np.argsort(eigvals)[::-1]
        self.P = eigvecs[:, idx[:min(self.d, 50)]]  # Keep top components
        
        # Compute M
        M = self.P @ self.P.T
        return M
    
    def compute_instance_correlation(self, X):
        """Compute instance correlation matrix N using k-NN"""
        print("Computing instance correlation matrix...")
        n = X.shape[0]
        N = np.zeros((n, n))
        
        from sklearn.metrics import pairwise_distances
        distances = pairwise_distances(X, metric='euclidean')
        
        for i in range(n):
            neighbors_idx = np.argsort(distances[i])[1:self.K+1]
            d_c = np.max(distances[i, neighbors_idx])
            
            for j in neighbors_idx:
                if d_c > 0:
                    N[i, j] = np.exp(-((distances[i, j] / d_c) ** 2))
        
        self.N = N
        return N
    
    def soft_threshold_l1(self, W, threshold):
        """Soft thresholding operator for L1 norm"""
        return np.sign(W) * np.maximum(np.abs(W) - threshold, 0)
    
    def singular_value_threshold(self, W, threshold):
        """Singular value thresholding for nuclear norm"""
        U, s, Vt = np.linalg.svd(W, full_matrices=False)
        s_threshold = np.maximum(s - threshold, 0)
        return U @ np.diag(s_threshold) @ Vt
    
    def fit(self, X, Y, max_iter=100, tol=1e-4, verbose=True):
        """
        Train MLFD model
        
        Args:
            X: Feature matrix (n x d)
            Y: Multi-label matrix (n x l) with values {-1, +1}
            max_iter: Maximum iterations
            tol: Convergence tolerance
            verbose: Print progress
        """
        n, d = X.shape
        _, l = Y.shape
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Training MLFD Classifier")
            print(f"{'='*70}")
            print(f"Samples: {n}, Features: {d}, Labels: {l}")
            print(f"Parameters: λ1={self.lambda1}, λ2={self.lambda2}, λ3={self.lambda3}")
            print(f"            λ4={self.lambda4}, λ5={self.lambda5}, λ6={self.lambda6}")
            print(f"{'='*70}\n")
        
        # Compute correlation matrices
        M = self.compute_feature_correlation(X)
        N = self.compute_instance_correlation(X)
        
        # Precompute matrices for efficiency
        XtX = X.T @ X
        XtY = X.T @ Y
        YtY = Y.T @ Y
        
        loss_history = []
        
        pbar = tqdm(range(max_iter), desc="MLFD Training") if verbose else range(max_iter)
        for iter in pbar:
            W_old = self.W.copy()
            
            # Update W using gradient descent
            grad_W = (XtX @ self.W - XtY + 
                     self.lambda4 * XtX @ self.W @ self.C @ self.C.T +
                     self.lambda5 * M @ self.W @ YtY +
                     self.lambda6 * (X.T - X.T @ N) @ (X - N @ X) @ self.W +
                     self.mu * (self.W - self.Q + self.Lambda / self.mu))
            
            lr = 0.001 / (1 + 0.01 * iter)
            self.W = self.W - lr * grad_W
            
            # Update C using proximal gradient descent
            grad_C = (self.lambda2 * YtY @ (self.C - np.eye(l)) + 
                     self.lambda4 * (X @ self.W).T @ X @ self.W @ self.C)
            
            L = np.sqrt(2 * np.linalg.norm(self.lambda2 * YtY, 2)**2 + 
                       2 * np.linalg.norm(self.lambda4 * (X @ self.W).T @ X @ self.W, 2)**2)
            
            if L > 0:
                C_temp = self.C - grad_C / L
                self.C = self.soft_threshold_l1(C_temp, self.lambda3 / L)
            
            # Update Q using singular value thresholding
            self.Q = self.singular_value_threshold(
                self.W + self.Lambda / self.mu, 
                self.lambda1 / self.mu
            )
            
            # Update Lagrange multipliers
            self.Lambda = self.Lambda + self.mu * (self.W - self.Q)
            
            # Compute loss
            loss = (np.linalg.norm(X @ self.W - Y, 'fro')**2 +
                   self.lambda1 * np.sum(np.linalg.svd(self.Q, compute_uv=False)) +
                   self.lambda2 / 2 * np.linalg.norm(Y @ self.C - Y, 'fro')**2 +
                   self.lambda3 * np.sum(np.abs(self.C)) +
                   self.mu / 2 * np.linalg.norm(self.W - self.Q, 'fro')**2)
            
            loss_history.append(loss)
            
            # Check convergence
            W_diff = np.linalg.norm(self.W - W_old, 'fro') / (np.linalg.norm(W_old, 'fro') + 1e-8)
            
            if verbose and isinstance(pbar, tqdm):
                pbar.set_postfix({'loss': f'{loss:.4f}', 'W_diff': f'{W_diff:.6f}'})
            
            if W_diff < tol and iter > 10:
                if verbose:
                    print(f"\nConverged at iteration {iter+1}")
                break
        
        if verbose:
            print(f"\n✓ MLFD training completed!")
            print(f"  Final loss: {loss:.4f}")
            print(f"  Label correlation matrix C sparsity: {np.sum(np.abs(self.C) < 1e-3) / (l*l):.2%}")
        
        return loss_history
    
    def predict(self, X):
        """
        Predict multi-labels
        
        Returns:
            Y_pred: Predictions in {-1, +1} format
            scores: Raw scores before thresholding
        """
        scores = X @ self.W
        scores_refined = scores @ self.C
        Y_pred = (scores_refined > 0).astype(int)
        Y_pred[Y_pred == 0] = -1  # Convert 0 to -1
        
        return Y_pred, scores_refined
    
    def predict_proba(self, X):
        """
        Predict label probabilities
        
        Returns:
            probs: Probabilities for each label
        """
        scores = X @ self.W
        scores_refined = scores @ self.C
        probs = 1 / (1 + np.exp(-scores_refined))
        return probs

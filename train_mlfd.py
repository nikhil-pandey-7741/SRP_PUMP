import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from models import MLFDClassifier


def train_mlfd(features_dir='./extracted_features', 
               label_columns=None,
               output_dir='./trained_mlfd',
               lambda1=0.1, lambda2=0.1, lambda3=0.01,
               lambda4=0.1, lambda5=0.1, lambda6=0.1,
               mu=1.0, K=5, max_iter=50):
    """
    Train MLFD classifier on extracted features
    
    Args:
        features_dir: Directory containing extracted features
        label_columns: List of label names
        output_dir: Directory to save MLFD model
        lambda1-lambda6: MLFD regularization parameters
        mu: ADMM penalty parameter
        K: Number of nearest neighbors
        max_iter: Maximum training iterations
    """
    
    print(f"\n{'='*70}")
    print("TRAINING MLFD: Multi-Label Fault Diagnosis")
    print(f"{'='*70}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load training features
    train_data = np.load(os.path.join(features_dir, 'train_features.npz'))
    X_train = train_data['features_combined']
    y_train_single = train_data['labels']
    
    n, d = X_train.shape
    l = len(label_columns)
    
    print(f"Training data:")
    print(f"  Samples: {n}")
    print(f"  Features: {d}")
    print(f"  Labels: {l}\n")
    
    # Convert single-label to multi-label format
    # Format: Y[i, j] = +1 if label j is present, -1 otherwise
    Y_train = np.ones((n, l)) * -1
    for i, label in enumerate(y_train_single):
        Y_train[i, int(label)] = 1
    
    print("Label distribution:")
    for i, name in enumerate(label_columns):
        count = np.sum(Y_train[:, i] == 1)
        print(f"  {name}: {count} ({count/n*100:.1f}%)")
    print()
    
    # Initialize MLFD
    mlfd = MLFDClassifier(
        d=d,
        l=l,
        lambda1=lambda1,
        lambda2=lambda2,
        lambda3=lambda3,
        lambda4=lambda4,
        lambda5=lambda5,
        lambda6=lambda6,
        mu=mu,
        K=K
    )
    
    # Train MLFD
    loss_history = mlfd.fit(X_train, Y_train, max_iter=max_iter, verbose=True)
    
    # Save MLFD model
    model_path = os.path.join(output_dir, 'mlfd_model.npz')
    np.savez(
        model_path,
        W=mlfd.W,
        C=mlfd.C,
        P=mlfd.P,
        N=mlfd.N,
        lambda1=lambda1,
        lambda2=lambda2,
        lambda3=lambda3,
        lambda4=lambda4,
        lambda5=lambda5,
        lambda6=lambda6,
        mu=mu,
        K=K
    )
    print(f"\n✓ MLFD model saved to {model_path}")
    
    # Predict on all splits and save
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(features_dir, f'{split}_features.npz')
        if os.path.exists(split_path):
            data = np.load(split_path)
            X = data['features_combined']
            
            Y_pred, scores = mlfd.predict(X)
            probs = mlfd.predict_proba(X)
            
            # Save predictions
            pred_path = os.path.join(output_dir, f'{split}_mlfd_predictions.npz')
            np.savez(
                pred_path,
                predictions=Y_pred,
                scores=scores,
                probabilities=probs,
                true_labels=data['labels']
            )
            print(f"✓ {split.upper()} predictions saved to {pred_path}")
    
    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, linewidth=2, color='steelblue')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('MLFD Training Loss', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mlfd_loss_curve.png'), dpi=150)
    plt.close()
    print(f"✓ Loss curve saved")
    
    # Visualize label correlation matrix C
    plt.figure(figsize=(12, 10))
    sns.heatmap(mlfd.C, annot=True, fmt='.2f', cmap='coolwarm', center=0,
               xticklabels=label_columns if label_columns else range(l),
               yticklabels=label_columns if label_columns else range(l),
               cbar_kws={'label': 'Correlation'})
    plt.title('MLFD Label Correlation Matrix C', fontsize=14, fontweight='bold')
    plt.xlabel('Label', fontsize=12)
    plt.ylabel('Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'label_correlation_matrix.png'), dpi=150)
    plt.close()
    print(f"✓ Label correlation matrix saved")
    
    # Evaluate on test set
    test_data = np.load(os.path.join(features_dir, 'test_features.npz'))
    X_test = test_data['features_combined']
    y_test_single = test_data['labels']
    
    # Convert to multi-label
    Y_test = np.ones((len(y_test_single), l)) * -1
    for i, label in enumerate(y_test_single):
        Y_test[i, int(label)] = 1
    
    Y_pred_test, _ = mlfd.predict(X_test)
    
    # Calculate metrics
    print(f"\n{'='*70}")
    print("MLFD TEST SET EVALUATION")
    print(f"{'='*70}\n")
    
    # Hamming Loss
    hamming = np.mean(Y_pred_test != Y_test)
    print(f"Hamming Loss: {hamming:.4f}")
    
    # Exact Match Accuracy
    exact_match = np.mean(np.all(Y_pred_test == Y_test, axis=1))
    print(f"Exact Match Accuracy: {exact_match*100:.2f}%")
    
    # Per-label accuracy
    print(f"\nPer-Label Accuracy:")
    for i, name in enumerate(label_columns):
        acc = np.mean(Y_pred_test[:, i] == Y_test[:, i])
        print(f"  {name}: {acc*100:.2f}%")
    
    # Save evaluation metrics
    metrics = {
        'hamming_loss': hamming,
        'exact_match_accuracy': exact_match,
        'loss_history': loss_history
    }
    
    metrics_path = os.path.join(output_dir, 'mlfd_metrics.npz')
    np.savez(metrics_path, **metrics)
    print(f"\n✓ Metrics saved to {metrics_path}")
    
    print(f"\n{'='*70}")
    print("✅ MLFD TRAINING COMPLETED")
    print(f"{'='*70}\n")
    
    return mlfd, metrics


if __name__ == '__main__':
    label_columns = [
        'Bottom_Tagging', 'Fluid_Pounding', 'Gas_Interference', 'Gas_Lock',
        'Good_Dynamograph', 'Polish_Rod_Tagging', 'Pump_Wear',
        'Standing_Valve_Leak', 'Standing_Valve_Sticky_Open', 'Stuck_Plunger',
        'Traveling_Valve_Leak', 'Traveling_Valve_Sticking'
    ]
    
    mlfd, metrics = train_mlfd(
        features_dir='./extracted_features',
        label_columns=label_columns,
        output_dir='./trained_mlfd'
    )

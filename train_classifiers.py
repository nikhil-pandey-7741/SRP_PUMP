import os
import time
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)


def train_classifiers_with_hpt(features_dir='./extracted_features',
                                label_columns=None,
                                output_dir='./trained_classifiers',
                                use_grid_search=True,
                                n_jobs=-1):
    """
    Train multiple classifiers with hyperparameter tuning
    
    Args:
        features_dir: Directory containing extracted features
        label_columns: List of label names
        output_dir: Directory to save trained models
        use_grid_search: Whether to use GridSearchCV for hyperparameter tuning
        n_jobs: Number of parallel jobs (-1 for all cores)
    """
    
    print(f"\n{'='*70}")
    print("TRAINING ML CLASSIFIERS WITH HYPERPARAMETER TUNING")
    print(f"{'='*70}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load features
    print("Loading features...")
    train_data = np.load(os.path.join(features_dir, 'train_features.npz'))
    val_data = np.load(os.path.join(features_dir, 'val_features.npz'))
    test_data = np.load(os.path.join(features_dir, 'test_features.npz'))
    
    X_train = train_data['features_combined']
    y_train = train_data['labels']
    X_val = val_data['features_combined']
    y_val = val_data['labels']
    X_test = test_data['features_combined']
    y_test = test_data['labels']
    
    # Combine train + val for final training
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.hstack([y_train, y_val])
    
    print(f"✓ Features loaded:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}\n")
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    scaler_path = os.path.join(output_dir, 'feature_scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"✓ Scaler saved to {scaler_path}\n")
    
    # Define classifiers with hyperparameter grids
    classifiers = {
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            } if use_grid_search else {}
        },
        'SVM (RBF)': {
            'model': SVC(kernel='rbf', probability=True, random_state=42),
            'params': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01]
            } if use_grid_search else {}
        },
        'SVM (Linear)': {
            'model': SVC(kernel='linear', probability=True, random_state=42),
            'params': {
                'C': [0.1, 1, 10, 100]
            } if use_grid_search else {}
        },
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=1000, random_state=42),
            'params': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            } if use_grid_search else {}
        },
        'K-Nearest Neighbors': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            } if use_grid_search else {}
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]
            } if use_grid_search else {}
        },
        'Naive Bayes': {
            'model': GaussianNB(),
            'params': {}
        }
    }
    
    # Train classifiers
    results = []
    trained_models = {}
    
    print(f"{'='*70}")
    print("TRAINING CLASSIFIERS")
    print(f"{'='*70}\n")
    
    for clf_name, clf_config in classifiers.items():
        print(f"\n{'='*70}")
        print(f"Training: {clf_name}")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        if use_grid_search and clf_config['params']:
            print(f"Running GridSearchCV...")
            print(f"  Parameter grid: {clf_config['params']}")
            
            grid_search = GridSearchCV(
                clf_config['model'],
                clf_config['params'],
                cv=3,
                scoring='accuracy',
                n_jobs=n_jobs,
                verbose=1
            )
            
            grid_search.fit(X_train_scaled, y_train_full)
            best_model = grid_search.best_estimator_
            
            print(f"\n✓ Best parameters: {grid_search.best_params_}")
            print(f"  Best CV score: {grid_search.best_score_:.4f}")
        else:
            print(f"Training with default parameters...")
            best_model = clf_config['model']
            best_model.fit(X_train_scaled, y_train_full)
        
        # Evaluate on test set
        y_pred = best_model.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        train_time = time.time() - start_time
        
        print(f"\n✓ Test Results:")
        print(f"  Accuracy:  {accuracy*100:.2f}%")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  Time:      {train_time:.2f}s")
        
        # Save model
        model_filename = clf_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        model_path = os.path.join(output_dir, f'{model_filename}.pkl')
        joblib.dump(best_model, model_path)
        print(f"✓ Model saved to {model_path}")
        
        results.append({
            'Classifier': clf_name,
            'Test Accuracy': accuracy * 100,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Training Time (s)': train_time
        })
        
        trained_models[clf_name] = best_model
    
    # Create comparison DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Test Accuracy', ascending=False)
    
    # Save results
    results_path = os.path.join(output_dir, 'classifier_comparison.csv')
    results_df.to_csv(results_path, index=False)
    
    print(f"\n{'='*70}")
    print("CLASSIFIER COMPARISON")
    print(f"{'='*70}\n")
    print(results_df.to_string(index=False))
    print(f"\n✓ Results saved to {results_path}")
    
    # Create ensemble model from top 3 classifiers
    print(f"\n{'='*70}")
    print("CREATING ENSEMBLE MODEL")
    print(f"{'='*70}\n")
    
    top_3 = results_df.head(3)['Classifier'].tolist()
    print(f"Top 3 classifiers:")
    for i, clf_name in enumerate(top_3, 1):
        acc = results_df[results_df['Classifier'] == clf_name]['Test Accuracy'].values[0]
        print(f"  {i}. {clf_name}: {acc:.2f}%")
    
    ensemble_estimators = [(name, trained_models[name]) for name in top_3]
    ensemble = VotingClassifier(estimators=ensemble_estimators, voting='soft')
    ensemble.fit(X_train_scaled, y_train_full)
    
    # Evaluate ensemble
    y_pred_ensemble = ensemble.predict(X_test_scaled)
    ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
    
    print(f"\n✓ Ensemble Accuracy: {ensemble_accuracy*100:.2f}%")
    
    # Save ensemble
    ensemble_path = os.path.join(output_dir, 'ensemble_model.pkl')
    joblib.dump(ensemble, ensemble_path)
    print(f"✓ Ensemble saved to {ensemble_path}")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Accuracy comparison
    axes[0].barh(results_df['Classifier'], results_df['Test Accuracy'], color='steelblue')
    axes[0].set_xlabel('Test Accuracy (%)', fontsize=12)
    axes[0].set_title('Classifier Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    # F1-Score comparison
    axes[1].barh(results_df['Classifier'], results_df['F1-Score'], color='coral')
    axes[1].set_xlabel('F1-Score', fontsize=12)
    axes[1].set_title('Classifier F1-Score Comparison', fontsize=14, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'classifier_comparison.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"✓ Comparison plot saved to {plot_path}")
    
    # Generate confusion matrix for best classifier
    best_clf_name = results_df.iloc[0]['Classifier']
    best_clf = trained_models[best_clf_name]
    y_pred_best = best_clf.predict(X_test_scaled)
    
    cm = confusion_matrix(y_test, y_pred_best)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=label_columns if label_columns else range(len(cm)),
               yticklabels=label_columns if label_columns else range(len(cm)))
    plt.title(f'Confusion Matrix - {best_clf_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, f'confusion_matrix_{best_clf_name.lower().replace(" ", "_")}.png')
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"✓ Confusion matrix saved to {cm_path}")
    
    # Classification report
    report = classification_report(y_test, y_pred_best, 
                                   target_names=label_columns if label_columns else None,
                                   output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_path = os.path.join(output_dir, f'classification_report_{best_clf_name.lower().replace(" ", "_")}.csv')
    report_df.to_csv(report_path)
    print(f"✓ Classification report saved to {report_path}")
    
    print(f"\n{'='*70}")
    print("✅ CLASSIFIER TRAINING COMPLETED")
    print(f"{'='*70}\n")
    
    return results_df, trained_models, scaler, ensemble


if __name__ == '__main__':
    label_columns = [
        'Bottom_Tagging', 'Fluid_Pounding', 'Gas_Interference', 'Gas_Lock',
        'Good_Dynamograph', 'Polish_Rod_Tagging', 'Pump_Wear',
        'Standing_Valve_Leak', 'Standing_Valve_Sticky_Open', 'Stuck_Plunger',
        'Traveling_Valve_Leak', 'Traveling_Valve_Sticking'
    ]
    
    results_df, models, scaler, ensemble = train_classifiers_with_hpt(
        features_dir='./extracted_features',
        label_columns=label_columns,
        output_dir='./trained_classifiers',
        use_grid_search=True,
        n_jobs=-1
    )

import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')


def run_complete_pipeline():
    """Execute complete training pipeline"""
    
    start_time = time.time()
    
    print("\n" + "="*80)
    print("🚀 COMPLETE TRAINING PIPELINE - ALL COMPONENTS")
    print("="*80)
    
    print("\n📋 PIPELINE STEPS:")
    print("  1. Train CNN (SparseVGGNet with enhancements)")
    print("  2. Extract Features (CNN + HU Moments)")
    print("  3. Train MLFD (Multi-label classifier)")
    print("  4. Train ML Classifiers (7 models with hyperparameter tuning)")
    print("  5. Create Ensemble (Top 3 models)")
    print("  6. Generate Reports & Plots")
    
    print("\n🚀 ENHANCEMENTS INCLUDED:")
    print("-" * 80)
    print("  CNN Training:")
    print("    ✓ Reduced Dropout (0.3)")
    print("    ✓ Label Smoothing (0.1)")
    print("    ✓ Cosine Annealing LR")
    print("    ✓ Temperature Calibration")
    print("    ✓ Early Stopping")
    print("\n  MLFD Training:")
    print("    ✓ Multi-Label Classification")
    print("    ✓ Label Correlation Exploitation")
    print("    ✓ Feature Correlation (PCA)")
    print("    ✓ Instance Correlation (k-NN)")
    print("\n  Classifier Training:")
    print("    ✓ GridSearchCV Hyperparameter Tuning")
    print("    ✓ 7 Different Classifiers")
    print("    ✓ Ensemble Model (Top 3)")
    print("    ✓ Comprehensive Metrics")
    print("="*80 + "\n")
    
    # ========================================================================
    # CONFIGURATION - EDIT THIS SECTION
    # ========================================================================
    
    BASE_PATH = r"C:\Users\Nikhil Pandey\Desktop\kongsberg\codes\research_paper_approach\data\dataset_split"
    
    TRAIN_CSV = os.path.join(BASE_PATH, 'train.csv')
    VAL_CSV = os.path.join(BASE_PATH, 'val.csv')
    TEST_CSV = os.path.join(BASE_PATH, 'test.csv')
    
    LABEL_COLUMNS = [
        'Bottom_Tagging', 'Fluid_Pounding', 'Gas_Interference', 'Gas_Lock',
        'Good_Dynamograph', 'Polish_Rod_Tagging', 'Pump_Wear',
        'Standing_Valve_Leak', 'Standing_Valve_Sticky_Open', 'Stuck_Plunger',
        'Traveling_Valve_Leak', 'Traveling_Valve_Sticking'
    ]
    
    # Training parameters
    CNN_EPOCHS = 100
    CNN_BATCH_SIZE = 64
    CNN_LEARNING_RATE = 0.001
    
    MLFD_MAX_ITER = 50
    MLFD_LAMBDA1 = 0.1
    MLFD_LAMBDA2 = 0.1
    MLFD_LAMBDA3 = 0.01
    MLFD_LAMBDA4 = 0.1
    MLFD_LAMBDA5 = 0.1
    MLFD_LAMBDA6 = 0.1
    MLFD_MU = 1.0
    MLFD_K = 5
    
    USE_HYPERPARAMETER_TUNING = True  # Set False for faster training
    
    # Output directories
    CNN_MODEL_DIR = './trained_models'
    FEATURES_DIR = './extracted_features'
    MLFD_DIR = './trained_mlfd'
    CLASSIFIERS_DIR = './trained_classifiers'
    
    # ========================================================================
    # VALIDATION
    # ========================================================================
    
    print("📋 CONFIGURATION:")
    print(f"  Dataset: {BASE_PATH}")
    print(f"  Classes: {len(LABEL_COLUMNS)}")
    print(f"  CNN Epochs: {CNN_EPOCHS}")
    print(f"  Hyperparameter Tuning: {'Enabled' if USE_HYPERPARAMETER_TUNING else 'Disabled'}")
    print("="*80 + "\n")
    
    if not os.path.exists(BASE_PATH):
        print(f"❌ ERROR: Dataset path not found: {BASE_PATH}")
        print("   Edit BASE_PATH in CONFIGURATION section and run again.")
        return False
    
    if not all([os.path.exists(f) for f in [TRAIN_CSV, VAL_CSV, TEST_CSV]]):
        print(f"❌ ERROR: CSV files not found!")
        return False
    
    print("✓ Configuration validated!\n")
    
    # ========================================================================
    # STEP 1: TRAIN CNN
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 1: TRAINING CNN (SparseVGGNet)")
    print("="*80)
    
    from train_cnn import train_cnn
    
    try:
        model, history = train_cnn(
            train_csv=TRAIN_CSV,
            val_csv=VAL_CSV,
            test_csv=TEST_CSV,
            base_path=BASE_PATH,
            label_columns=LABEL_COLUMNS,
            num_epochs=CNN_EPOCHS,
            batch_size=CNN_BATCH_SIZE,
            learning_rate=CNN_LEARNING_RATE,
            save_dir=CNN_MODEL_DIR
        )
        
        cnn_model_path = os.path.join(CNN_MODEL_DIR, 'best_model.pth')
        
        print(f"\n✓ CNN training complete!")
        print(f"  Best validation accuracy: {history['best_val_acc']:.2f}%")
        print(f"  Test accuracy: {history['test_acc']:.2f}%")
        print(f"  Model saved to: {cnn_model_path}")
        
    except Exception as e:
        print(f"\n❌ ERROR in CNN training: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ========================================================================
    # STEP 2: EXTRACT FEATURES
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 2: EXTRACTING FEATURES (CNN + HU Moments)")
    print("="*80)
    
    from feature_extractor import FeatureExtractor
    
    try:
        extractor = FeatureExtractor(
            cnn_model_path=cnn_model_path,
            num_classes=len(LABEL_COLUMNS),
            batch_size=CNN_BATCH_SIZE
        )
        
        all_features = extractor.extract_from_multiple_csvs(
            train_csv=TRAIN_CSV,
            val_csv=VAL_CSV,
            test_csv=TEST_CSV,
            image_column='image_path',
            label_columns=LABEL_COLUMNS,
            base_path=BASE_PATH,
            image_size=(190, 400),
            output_dir=FEATURES_DIR
        )
        
        print(f"\n✓ Feature extraction complete!")
        print(f"  Features saved to: {FEATURES_DIR}")
        
    except Exception as e:
        print(f"\n❌ ERROR in feature extraction: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ========================================================================
    # STEP 3: TRAIN MLFD
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 3: TRAINING MLFD (Multi-Label Fault Diagnosis)")
    print("="*80)
    
    from train_mlfd import train_mlfd
    
    try:
        mlfd, mlfd_metrics = train_mlfd(
            features_dir=FEATURES_DIR,
            label_columns=LABEL_COLUMNS,
            output_dir=MLFD_DIR,
            lambda1=MLFD_LAMBDA1,
            lambda2=MLFD_LAMBDA2,
            lambda3=MLFD_LAMBDA3,
            lambda4=MLFD_LAMBDA4,
            lambda5=MLFD_LAMBDA5,
            lambda6=MLFD_LAMBDA6,
            mu=MLFD_MU,
            K=MLFD_K,
            max_iter=MLFD_MAX_ITER
        )
        
        print(f"\n✓ MLFD training complete!")
        print(f"  Exact Match Accuracy: {mlfd_metrics['exact_match_accuracy']*100:.2f}%")
        print(f"  Model saved to: {MLFD_DIR}")
        
    except Exception as e:
        print(f"\n❌ ERROR in MLFD training: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ========================================================================
    # STEP 4: TRAIN CLASSIFIERS
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 4: TRAINING ML CLASSIFIERS")
    print("="*80)
    
    from train_classifiers import train_classifiers_with_hpt
    
    try:
        results_df, models, scaler, ensemble = train_classifiers_with_hpt(
            features_dir=FEATURES_DIR,
            label_columns=LABEL_COLUMNS,
            output_dir=CLASSIFIERS_DIR,
            use_grid_search=USE_HYPERPARAMETER_TUNING,
            n_jobs=-1
        )
        
        print(f"\n✓ Classifier training complete!")
        print(f"  Models saved to: {CLASSIFIERS_DIR}")
        
    except Exception as e:
        print(f"\n❌ ERROR in classifier training: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    end_time = time.time()
    total_time = end_time - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    
    print("\n" + "="*80)
    print("✅ COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
    print("="*80)
    
    print(f"\n⏱️  Total Time: {hours}h {minutes}m")
    
    print(f"\n📊 FINAL RESULTS:")
    print("-" * 80)
    print(f"  CNN Test Accuracy:          {history['test_acc']:.2f}%")
    print(f"  MLFD Exact Match Accuracy:  {mlfd_metrics['exact_match_accuracy']*100:.2f}%")
    
    if results_df is not None:
        best_clf = results_df.iloc[0]
        print(f"  Best Classifier:            {best_clf['Classifier']}")
        print(f"  Best Classifier Accuracy:   {best_clf['Test Accuracy']:.2f}%")
    
    print(f"\n📁 OUTPUT DIRECTORIES:")
    print(f"  CNN Models:        {CNN_MODEL_DIR}/")
    print(f"  Features:          {FEATURES_DIR}/")
    print(f"  MLFD:              {MLFD_DIR}/")
    print(f"  Classifiers:       {CLASSIFIERS_DIR}/")
    
    print(f"\n📈 GENERATED FILES:")
    print(f"  • training_curves.png")
    print(f"  • mlfd_loss_curve.png")
    print(f"  • label_correlation_matrix.png")
    print(f"  • classifier_comparison.png")
    print(f"  • confusion_matrix_*.png")
    print(f"  • classification_report_*.csv")
    print(f"  • All trained models (.pth, .pkl)")
    
    print("\n" + "="*80)
    print("🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80 + "\n")
    
    return True


if __name__ == "__main__":
    print("\n" + "="*80)
    print("DYNAMOGRAPH FAULT CLASSIFICATION - COMPLETE PIPELINE")
    print("="*80)
    print("\n⚠️  IMPORTANT:")
    print("  1. Edit BASE_PATH in CONFIGURATION section")
    print("  2. Ensure train.csv, val.csv, test.csv exist")
    print("  3. Estimated time: 2-4 hours")
    print("\n" + "="*80 + "\n")
    
    # Run pipeline directly without asking
    success = run_complete_pipeline()
    
    if success:
        print("\n✅ ALL DONE! Check the output directories for results.")
    else:
        print("\n❌ Pipeline failed! Check the error messages above.")

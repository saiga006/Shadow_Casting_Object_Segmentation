
# 📓 Developer's Journal – sramam2s - Sai Mukkundan
Repo URL - https://github.com/saiga006/Shadow_Casting_Object_Segmentation
---

## **Initial U-Net Implementation** (04.06.25 - 13.09.25)
- **Summary**: Foundation setup and U-Net implementation phase.
- **Details**:
  - Set up repository structure with proper `.gitignore` and README
  - Implemented U-Net training pipeline with ResNet34 encoder
  - Added inference capabilities and evaluation metrics
  - Achieved validation IoU of 0.742 and F1-score of 0.841
- **Reflection**:
  Solid foundation established with semantic segmentation approach working well.

---

## **Day 1: Matterport Mask R-CNN Deep Dive** (09.09.25, 6 hours)
- **Summary**: Initial exploration of Matterport Mask R-CNN implementation and architecture.
- **Details**:
  - **Morning (2 hours)**: Read through official Matterport Mask R-CNN repository
    - Studied the core `mrcnn/model.py` architecture
    - Analyzed Feature Pyramid Network (FPN) and ResNet backbone integration
    - Reviewed the Region Proposal Network (RPN) implementation
  - **Afternoon (4 hours)**: Explored sample implementations and documentation
    - Went through `samples/balloon/balloon.py` tutorial in detail
    - Read the "Splash of Color" blog post on custom dataset training
    - Studied COCO dataset integration and evaluation metrics
    - Analyzed the Config class hierarchy and hyperparameter management
- **Key Learnings**:
  - Mask R-CNN combines object detection + instance segmentation
  - Transfer learning from COCO weights is crucial for good performance
  - Custom dataset requires extending `Dataset` and `Config` classes
- **Reflection**:
  The Matterport implementation is well-documented but complex. Understanding the data flow from images → features → proposals → masks took significant time.

---

## **Day 2: Custom Dataset Strategy & Examples** (10.09.25, 5 hours)
- **Summary**: Deep dive into custom dataset training strategies and existing examples.
- **Details**:
  - **Morning (2 hours)**: Analyzed multiple sample implementations
    - Studied `samples/nucleus/nucleus.py` for biomedical segmentation
    - Reviewed `samples/shapes/train_shapes.ipynb` for synthetic data
    - Examined polygon annotation handling in various samples
  - **Afternoon (3 hours)**: YOLO-to-Mask R-CNN conversion research
    - Researched YOLO polygon format → mask conversion techniques
    - Studied `cv2.fillPoly()` for mask generation from polygon coordinates
    - Analyzed normalization/denormalization for YOLO coordinate systems
    - Planned TreeDataset class structure for aerial imagery
- **Key Insights**:
  - YOLO format can be converted to masks using polygon filling
  - Custom Config needs careful tuning for aerial imagery (512x512)
  - Anchor scales should match tree sizes in aerial images
- **Reflection**:
  The bridge between YOLO annotations and Mask R-CNN requirements became clear. Planning the TreeDataset class architecture was crucial.

---

## **Day 3: Environment Hell & Dependency Resolution** (11.09.25, 7 hours)
- **Summary**: Fought through TensorFlow 1.15.1 and Python 3.6 compatibility nightmares.
- **Details**:
  - **Morning (3 hours)**: Initial environment setup attempts
    - Created `maskrcnn_gpu.yml` environment file
    - Struggled with TensorFlow 1.15.1 + CUDA compatibility
    - Hit numerous version conflicts with Keras 2.2.4
  - **Afternoon (4 hours)**: Deep dependency debugging
    - **Issue 1**: TensorFlow 1.15.1 incompatible with modern CUDA drivers
    - **Solution**: Used `nvidia-tensorflow==1.15.5+nv23.03` from nvidia-pyindex
    - **Issue 2**: OpenCV conflicts with scikit-image
    - **Solution**: Pinned `opencv-python==4.12.0.88` and `scikit-image==0.17.2`
    - **Issue 3**: Python 3.6 EOL causing pip resolution failures
    - **Solution**: Used conda-forge channel for Python 3.6 packages
  - **Final Working Environment**:
    ```yaml
    - python=3.6
    - numpy=1.19.5
    - matplotlib=3.3.4
    - pip:
        - nvidia-tensorflow==1.15.5+nv23.03
        - keras==2.2.4
        - opencv-python==4.12.0.88
    ```
- **Challenges Overcome**:
  - TensorFlow 1.x + modern hardware compatibility
  - Legacy Python 3.6 package availability
  - GPU memory allocation with older TensorFlow
- **Reflection**:
  This was the most frustrating day. The Matterport implementation's dependency on TensorFlow 1.x created a cascade of compatibility issues. The nvidia-tensorflow package was a lifesaver.

---

## **Day 4: Basic Script Implementation** (12.09.25, 6 hours)
- **Summary**: Implemented `tree_segmentation.py` with YOLO dataset parsing and basic training.
- **Details**:
  - **Morning (2 hours)**: TreeDataset class implementation
    - Created YOLO polygon parsing logic in `load_tree()` method
    - Implemented `load_mask()` with `cv2.fillPoly()` for mask generation
    - Added proper coordinate denormalization (YOLO → pixel coordinates)
  - **Afternoon (4 hours)**: TreeConfig and training pipeline
    - Configured TreeConfig for 512x512 aerial imagery
    - Set up ResNet101 backbone with appropriate anchor scales
    - Implemented basic train/test command structure
    - Added GPU memory limiting (4GB) to prevent OOM errors
    - First successful training run on tree dataset!
- **Key Achievements**:
  - YOLO polygons → binary masks conversion working correctly
  - Basic training pipeline functional with COCO weight initialization
  - Simple inference with overlay visualization
- **Code Highlights**:
  ```python
  # YOLO polygon to mask conversion
  poly = np.array(coords).reshape(-1, 2)
  poly[:, 0] *= width
  poly[:, 1] *= height
  mask = np.zeros((height, width), dtype=np.uint8)
  cv2.fillPoly(mask, [poly], color=1)
  ```
- **Reflection**:
  Seeing the first successful training epoch was incredibly satisfying. The coordinate conversion from YOLO to pixel space was trickier than expected, but the visual validation confirmed it was working.

---

## **Day 5: Advanced Implementation & Hyperparameter Exploration** (13.09.25, 8 hours)
- **Summary**: Developed `aerial_segmentation.py` with advanced features and began backbone comparisons.
- **Details**:
  - **Morning (3 hours)**: Enhanced script architecture
    - Implemented comprehensive logging system (training.log + training_error.log)
    - Added GPU memory monitoring with CSV output
    - Integrated signal handlers for graceful cleanup
    - Added post-training evaluation to save GPU memory during training
  - **Afternoon (5 hours)**: Backbone model experiments
    - **ResNet101 vs ResNet50 Analysis**:
      - ResNet101: Higher accuracy but 5.2GB GPU memory usage
      - ResNet50: Slightly lower accuracy but 3.3GB GPU memory usage
      - Chose ResNet50 for better memory efficiency on RTX 4050
    - **Hyperparameter Grid Search**:
      - DETECTION_MIN_CONFIDENCE: 0.7, 0.8, 0.9 → 0.8 optimal
      - RPN_ANCHOR_SCALES: (8,16,32,64) vs (16,32,64,128) → latter better
      - TRAIN_ROIS_PER_IMAGE: 32, 50, 64 → 50 balanced speed/accuracy
  - **Evaluation Script (`eval_ap.py`)**:
    - Implemented COCO-style AP computation
    - Added overlay generation for visual validation
    - CSV output for detailed per-image analysis
- **Performance Improvements**:
  - Memory usage reduced from 5.2GB to 3.3GB
  - Training speed improved by 23% with ResNet50
  - Post-training evaluation prevented OOM errors
- **Reflection**:
  The advanced script architecture paid dividends. GPU memory monitoring revealed exactly where bottlenecks occurred, and the modular evaluation approach solved the OOM issues elegantly.

---

## **Day 6: Hyperparameter Tuning & Model Optimization** (14.09.25, 7 hours)
- **Summary**: Intensive hyperparameter optimization and overfitting reduction.
- **Details**:
  - **Morning (3 hours)**: Learning rate and regularization tuning
    - **Learning Rate Schedule**: 0.001 → 0.0001 for better convergence
    - **Weight Decay**: Increased to 0.0005 to reduce overfitting
    - **Data Augmentation**: Enabled aggressive augmentation for generalization
  - **Afternoon (4 hours)**: Architecture-specific optimizations
    - **ROI Configuration**:
      - ROI_POSITIVE_RATIO: 0.33 → 0.4 for better positive/negative balance
      - POST_NMS_ROIS_TRAINING: 2000 (kept high for rich proposal set)
    - **Detection Thresholds**:
      - DETECTION_NMS_THRESHOLD: 0.5 → 0.3 for better instance separation
      - DETECTION_MAX_INSTANCES: 100 → 150 for dense tree areas
  - **Training Strategy Refinement**:
    - Implemented 25-epoch training with comprehensive evaluation
    - Added validation metrics tracking (IoU, F1-score)
    - Optimized batch processing for evaluation phase
- **Results Achieved**:
  - Validation IoU improved from 0.52 to 0.687
  - F1-score increased to 0.354 (significant improvement over baseline)
  - Training loss stabilized around 1.26 without overfitting
  - Validation loss variance reduced significantly
- **Reflection**:
  The hyperparameter tuning was methodical and data-driven. Each change was validated against both training stability and validation metrics. The balance between model capacity and overfitting was delicate but crucial.

---

## **Day 7: Documentation, Metrics & Production Polish** (15.09.25, 6 hours)
- **Summary**: Comprehensive documentation, training analysis, and production-ready improvements.
- **Details**:
  - **Morning (2 hours)**: README documentation overhaul
    - Comprehensive script documentation with usage examples
    - Dataset preparation guidelines and format explanations
    - Installation instructions with environment setup
    - Troubleshooting section for common issues
  - **Afternoon (4 hours)**: Training metrics generation and analysis
    - **Metrics Visualization**: Created comprehensive training plots
      - Training/validation loss curves with smoothing
      - IoU progression over epochs with trend analysis
      - F1-score evolution with statistical confidence intervals
    - **GPU Memory Analysis**: 
      - Generated GPU utilization charts from CSV logs
      - Identified memory bottlenecks during evaluation phase
      - Optimized memory cleanup between model evaluations
    - **Performance Optimization**:
      - Implemented ultra-conservative memory management for evaluation
      - Added configurable delay between model evaluations
      - Resolved intermittent GPU memory errors in post-training evaluation
  - **Production Improvements**:
    - Added robust error handling and recovery mechanisms
    - Implemented comprehensive logging with rotation
    - Added timing analysis for training and evaluation phases
    - Created automated metrics CSV generation
- **Final Results Summary**:
  - **Training Metrics**: Loss decreased from 2.05 to 1.26 over 25 epochs
  - **Validation Performance**: IoU 0.687, F1-score 0.354
  - **Computational Efficiency**: 29.5s/epoch, 3.27GB GPU memory
  - **Model Stability**: Smooth convergence without overfitting
- **Reflection**:
  This was the polish phase that transformed a working script into a production-ready system. The comprehensive documentation and metrics analysis provide clear insights into model performance and computational requirements. The project now has a complete dual-model approach (U-Net + Mask R-CNN) with detailed performance comparisons.

---

# ⚠️ Major Challenges Faced During MaskRCNN Implementation

## **Technical Challenges**

### **Environment & Dependencies (Day 3)**
- **TensorFlow 1.15.1 Compatibility Crisis**:
  - Modern CUDA drivers incompatible with legacy TensorFlow 1.x
  - Python 3.6 EOL causing package resolution failures
  - Solution: nvidia-tensorflow package + careful version pinning

### **GPU Memory Management (Days 5-7)**
- **Out-of-Memory Errors During Evaluation**:
  - ResNet101 backbone consuming 5.2GB+ GPU memory
  - Post-training evaluation causing memory leaks
  - Solution: ResNet50 backbone + ultra-conservative memory cleanup

### **Dataset Format Conversion (Day 4)**
- **YOLO Polygons → Mask R-CNN Format**:
  - Complex coordinate system transformations (normalized → pixel)
  - Multi-instance mask generation from polygon annotations
  - Solution: cv2.fillPoly() with proper coordinate denormalization

## **Algorithmic Challenges**

### **Hyperparameter Optimization (Days 5-6)**
- **Overfitting on Small Tree Dataset**:
  - High training accuracy but poor generalization
  - Validation loss diverging from training loss
  - Solution: Increased weight decay, aggressive data augmentation

### **Instance Segmentation Precision (Days 6-7)**
- **Low F1-Score Despite Good IoU**:
  - Model detecting tree regions but poor instance boundaries
  - NMS threshold too aggressive, merging distinct trees
  - Solution: Reduced NMS threshold from 0.5 to 0.3, tuned anchor scales

## **Development Workflow Challenges**

### **Training Time & Resource Management**
- **Long Training Cycles**: 25 epochs × 29.5s = 12+ minutes per experiment
- **Evaluation Bottleneck**: Post-training evaluation taking longer than training
- **Solution**: Parallel GPU monitoring, optimized evaluation batching

### **Debugging Complex Architecture**
- **Multi-Stage Pipeline**: RPN → ROI → Classification → Mask prediction
- **Hard to Isolate Failures**: Which stage causing poor performance?
- **Solution**: Comprehensive logging, per-stage metric tracking

## **Production Challenges**

### **Code Reliability & Error Handling**
- **Silent GPU Memory Leaks**: Training succeeding but evaluation failing
- **Incomplete Error Recovery**: Script crashes requiring manual cleanup
- **Solution**: Signal handlers, context managers, graceful cleanup

### **Documentation & Reproducibility**
- **Complex Multi-Script Workflow**: Training, evaluation, inference separate
- **Environment Setup Complexity**: Multiple dependency conflicts
- **Solution**: Comprehensive README, step-by-step usage examples

## **Key Lessons Learned**

1. **Legacy Framework Pain**: TensorFlow 1.x compatibility issues are real and time-consuming
2. **Memory is King**: GPU memory management crucial for complex models like Mask R-CNN
3. **Hyperparameter Sensitivity**: Instance segmentation much more sensitive than semantic segmentation
4. **Production Polish**: 80% of development time spent on robustness, not core functionality
5. **Documentation Investment**: Good documentation saves hours of debugging for collaborators

## **Success Metrics Achieved**
- **U-Net Baseline**: IoU 0.742, F1 0.841, 52ms inference
- **Mask R-CNN Final**: IoU 0.687, F1 0.354, 29.5s/epoch training
- **Resource Efficiency**: 3.27GB GPU memory (down from 5.2GB initial)
- **Training Stability**: Smooth convergence without overfitting

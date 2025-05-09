# Direction Prediction
This implementation is a sophisticated deep learning solution for angle prediction from images, using a multi-task learning approach with a modern ConvNeXt architecture. The model predicts both the sine and cosine components of angles to handle circular continuity, along with an auxiliary classification task for cardinal directions.

## Model Architecture
### Base Model
- **Backbone**: ConvNeXt-Large (pre-trained on ImageNet)
  - A modern CNN architecture that adapts design principles from Vision Transformers while maintaining CNN efficiency
  - Used with pretrained weights from timm library
  - First 60% of layers frozen during training
### Custom Heads
   - 3-layer MLP with BatchNorm, GELU activation, and Dropout
   - Outputs 2 values (sin and cos of angle)
   - 2-layer MLP for predicting cardinal directions (8 classes)
   - Helps with learning directional features
### Innovative Components
1. **Circular Regression**:
   - Predicts sin/cos values instead of direct angles to handle circular continuity
   - Uses custom circular loss function
2. **Multi-Task Learning**:
   - Combines angle regression with cardinal direction classification
   - Weighted loss function balances both tasks
3. **Advanced Training Techniques**:
   - Mixed precision training (FP16) for faster computation
   - Cosine annealing with warm restarts for learning rate scheduling
   - Test-time augmentation (horizontal flip) for improved predictions

### Data Augmentation
Comprehensive Albumentations pipeline including:
- Random flips (horizontal/vertical), Rotation (±30°), Brightness/contrast adjustments, Hue/saturation shifts, Coarse dropout (random erasing), Gaussian noise, Normalization (ImageNet stats)

## Training Details
### Optimization
- **Optimizer**: AdamW (weight decay 1e-5)
- **Initial LR**: 5e-5 with cosine annealing
- **Batch Size**: 32
- **Epochs**: 23 (early stopping based on validation MAAE)

### Loss Function
Custom `CombinedLoss` with:
1. Circular regression loss (1 - cosine similarity)
2. Cross-entropy loss for cardinal directions
3. Weighting factor (α=0.2) for auxiliary loss

## Results
The model achieved a best validation MAAE of 36.69° after 17 epochs (with early stopping). The final submission includes predictions on both validation and test sets using test-time augmentation.

## Model Weights - Link:
https://iiithydstudents-my.sharepoint.com/:u:/g/personal/jagan_krishna_students_iiit_ac_in/EQ7IviDtOoBBkbD7TwmWwM8BZOdL1N4gkGENiCX7_JaJoA?e=M8CaOz

------------------------------------------------------------------------------------------------>
# Latitide abd Longitude Prediction

## Overview
This implementation is a deep learning solution for geographic coordinate prediction (latitude/longitude) from images, using an attention-enhanced EfficientNetV2 architecture. The model predicts normalized coordinates which are later transformed back to real-world values.

## Model Architecture
- **Backbone**: EfficientNetV2-M (pre-trained on ImageNet)
  - Modern CNN architecture with progressive training scaling
  - Used with pretrained weights from timm library
  - Features-only extraction (no classification head)
### Custom Components
1. **Attention Mechanism**:
   - 2-layer MLP with SiLU activation
   - Generates spatial attention weights (sigmoid-activated)
   - Multiplies with backbone features for adaptive feature selection
2. **Coordinate Regression Head**:
   - 3-layer MLP with SiLU activation and dropout
   - Outputs 2 values (normalized latitude and longitude)
   - Xavier initialization for all linear layers

## Key Features
### Data Processing
1. **Outlier Removal**:
   - 3×IQR rule applied to training/validation coordinates
   - Ensures model trains on plausible geographic locations
2. **Robust Normalization**:
   - RobustScaler for coordinate normalization
   - Resistant to outliers in the remaining data
   - Separate scalers for latitude and longitude
### Advanced Augmentation
Comprehensive Albumentations pipeline including:
- Random flips (horizontal/vertical), 90° rotations and transpositions, Affine transformations (translation, scaling, rotation), Brightness/contrast adjustments, Hue/saturation shifts, Coarse dropout (random erasing), Normalization (ImageNet stats)

### Optimization
- **Optimizer**: AdamW (weight decay 1e-5)
- **Initial LR**: 5e-3 with cosine annealing
- **Batch Size**: 20
- **Epochs**: 43 (early stopping patience=11)

## Model Weights - Link:
https://iiithydstudents-my.sharepoint.com/:u:/g/personal/jagan_krishna_students_iiit_ac_in/EUY1kszEy_NGjiqiqpVHW8cBTTC0A6k4K7lGa9nA3CNcYg?e=A050iB

-------------------------------------------------------------------------------------------------->

# Region ID Identification with EfficientNet

## Overview
This implementation is a sophisticated multimodal deep learning solution for geographic region identification (15 classes) that combines visual features from images with tabular data (coordinates, angle, and time). The model uses an attention-based fusion mechanism to effectively combine both data modalities.

## Model Architecture
### Base Model
- **Visual Backbone**: EfficientNet-B5 (pre-trained on ImageNet)
  - State-of-the-art CNN architecture with compound scaling
  - Only last 15 blocks unfrozen for fine-tuning
  - Features extracted from final convolutional layer (2048-dim)
### Custom Components
1. **Tabular Network**:
   - 2-layer MLP with BatchNorm and GELU activation
   - Processes 4 normalized features (lat, lon, angle, hour)
   - Outputs 512-dimensional embeddings
2. **Attention Mechanism**:
   - Computes importance weights for fused features
   - 2-layer MLP with sigmoid activation
   - Helps focus on most relevant multimodal features
3. **Classifier Head**:
   - 3-layer MLP with decreasing dimensionality (2048+512 → 1024 → 512 → 15)
   - BatchNorm and dropout for regularization
   - Outputs probabilities over 15 region classes

## Key Features
### Multimodal Fusion
- Concatenates image (2048-dim) and tabular (512-dim) features
- Attention-weighted feature combination
- Enables complementary information usage from both modalities

### Advanced Training Techniques
- Gradient accumulation (steps=2), Mixed precision training (FP16), Gradient clipping (max_norm=1.0), Label smoothing (α=0.1), Learning rate warmup (5 epochs), Cosine LR scheduling with minimum LR, Early stopping (patience=10)

### Optimization
- **Optimizer**: AdamW (weight_decay=1e-4)
- **Initial LR**: 5e-4 with cosine decay to 9e-5
- **Batch Size**: 28
- **Epochs**: 25

### Augmentation Pipeline
Comprehensive torchvision transforms including:
- Random flips (horizontal/vertical), Affine transformations (rotation, translation, scaling), 
(Color, brightness, contrast, saturation, hue), Random resized crops, Random erasing, Normalization (ImageNet stats)

## Model Weights - Link:
https://iiithydstudents-my.sharepoint.com/:u:/g/personal/jagan_krishna_students_iiit_ac_in/EXTuPnufj_1Ejy2sVbMRGCwBQbC2mgDldRMuDBpSDC-Jyw?e=L3zvhR






#                                                                        Thank You ..
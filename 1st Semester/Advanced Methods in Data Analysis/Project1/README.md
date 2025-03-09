# YOLO Object Detection


This project provides a practical implementation of YOLOv8 for object detection across three diverse datasets. The project covers data preprocessing, training, evaluation, and plotting of performance metrics. The experiments highlight:
- The high potential of YOLOv8 in well-balanced and structured datasets.
- The challenges that come with imbalanced classes and tiny objects.
- Strategies for improving performance through data augmentation and targeted training.

## Datasets
1. [**SkyFusion: Aerial Object Detection**](https://www.kaggle.com/datasets/kailaspsudheer/tiny-object-detection): Dataset for detecting tiny objects in satellite images, addressing the gap in datasets focused on small objects like vehicles, ships, and aircraft. It is a curated subset of the AiTOD v2 and Airbus Aircraft Detection datasets, optimized for accessibility and research in tiny object detection.
2. [**Traffic Sign Detection**](https://www.kaggle.com/datasets/pkdarabi/cardetection): Dataset focusing on identifying and classifying common traffic signs to support computer vision applications. It includes images of various traffic signs, such as traffic lights (Green, Red), speed limit signs (ranging from 10 to 120 km/h), and Stop signs, representing a diverse range of real-world road scenarios.
3. [**Fruits Detection**](https://www.kaggle.com/datasets/lakshaytyagi01/fruit-detection): Dataset containing images of six fruit types: Apple, Grapes, Pineapple, Orange, Banana, and Watermelon, annotated for object detection tasks. Each image has undergone preprocessing, including auto-orientation with EXIF data stripping.

## Results
| Dataset       | Classes | Training Samples | Validation Samples | Precision | Recall    | mAP50     | mAP50+90  |
|---------------|---------|------------------|--------------------|-----------|-----------|-----------|-----------|
| SkyFusion     | 3       | 2094 (82.3%)     | 450 (17.7%)        | 0.686     | 0.588     | 0.601     | 0.342     |
| Traffic Signs | 15      | 3530 (81.5%)     | 801 (18.5%)        | **0.972** | **0.899** | **0.965** | **0.832** |
| Fruits        | 6       | 7108 (88.6%)     | 914 (11.4%)        | 0.598     | 0.450     | 0.491     | 0.329     |


## Key findings
- **Performance Strengths**: YOLOv8 excels on balanced datasets with average-sized objects, as shown by the high performance on the Traffic Signs dataset.
- **Performance Weaknesses**: The model struggles with tiny objects and imbalanced datasets (e.g., SkyFusion and Multilabel Fruits), where underrepresented classes lead to lower precision and recall.
- **Future Improvements**:
  - *Data Augmentation*: Use oversampling or synthetic data generation to balance the datasets. 
  - *Enhanced Feature Extraction*: Focus on methods to better detect small or occluded objects. 
  - *Custom Training Approaches*: Implement training techniques specific to each dataset's challenges.

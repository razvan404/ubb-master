# Room Occupancy Prediction


This project estimates room occupancy in a 6m × 4.6m room using data from multiple sensors (temperature, light, sound, $CO_2$, and PIR motion). For that, two machine learning approaches are implemented:

- **Random Forest Classification**: To determine discrete occupancy levels.
- **Support Vector Regression (SVR)**: To predict fine-grained continuous occupancy estimates.

The project covers extensive data preprocessing, feature engineering, model training, evaluation, and performance analysis. It also provides comparative insights between classification and regression methods within a smart building context.

## Problem Description

The goal is to predict the number of people present in a room using sensor measurements and timestamp data. This task is approached in two ways:
- **Supervised Classification**: Categorizing occupancy into levels (0 for room unoccupied and 1 for room occupied).
- **Supervised Regression**: Estimating a continuous value which is later interpreted (and rounded) as the number of occupants.


## Dataset and Sensor Specifications

The [**Room Occupancy Estimation** dataset](https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation) includes:
- **Sensor Readings**:
  - **Temperature (S1–S4)**: Recorded in °C.
  - **Light Intensity (S1–S4)**: Measured in Lux.
  - **Sound Levels (S1–S4)**: Measured in Volts.
  - **$\mathbf{CO_2}$ Concentration (S5)** and its **Slope**.
  - **PIR Motion Sensors (S6 & S7)**: Binary indicators of movement.
- **Timestamps**: Date and time data.
- **Ground Truth**: Annotated occupancy counts ranging from 0 to 3.

Additional details such as sensor accuracy, resolution, and the sensor layout are described in the full documentation.

## Data Preprocessing

Key preprocessing steps include:
- **Data Cleaning and Balancing**: Days with only zero occupancy measurements were removed to prevent overfitting.
- **Time Labeling**: Raw time data are converted into categorical intervals (e.g., Night, Morning, Noon) to capture temporal patterns.
- **Feature Engineering**: Sensor readings are normalized and transformed to enhance model performance.


## Results

<table>
  <tr><td colspan="6"></td></tr>
  <tr>
    <th rowspan="2">Model</th>
    <th colspan="2">Optimized Parameters</th>
    <th colspan="3">Metrics</th>
  </tr>
  <tr>
    <th>Hyperparameter</th>
    <th>Value</th>
    <th>Metric</th>
    <th>Mean</th>
    <th>Confidence Interval</th>
  </tr>
  <tr><td colspan="6"></td></tr>
  <tr>
    <td rowspan="4">
      <strong>Random Forest Classification</strong>
    </td>
    <td>criterion</td>
    <td>entropy</td>
    <td>Accuracy</td>
    <td>0.9970000000000001</td>
    <td>(0.9935444978557783, 1.0004555021442219)</td>
  </tr>
  <tr>
    <td>n_estimators</td>
    <td>20</td>
    <td>Precision</td>
    <td>0.9915407183499289</td>
    <td>(0.9817556021517583, 1.0013258345480995)</td>
  </tr>
  <tr>
    <td>max_depth</td>
    <td>3</td>
    <td>Recall</td>
    <td>0.9999999999714664</td>
    <td>(0.9999999999680588, 0.999999999974874)</td>
  </tr>
  <tr>
    <td>max_features</td>
    <td>5</td>
    <td>F1 score</td>
    <td>0.995709502051524</td>
    <td>(0.9907459173251566, 1.0006730867778915)</td>
  </tr>
  <tr><td colspan="6"></td></tr>
  <tr>
    <td rowspan="5"><strong>SVR</strong></td>
    <td>kernel</td>
    <td>poly</td>
    <td>MAE</td>
    <td>0.05812253529103369</td>
    <td>(0.05217437169300859, 0.0640706988890588)</td>
  </tr>
  <tr>
    <td>degree</td>
    <td>2</td>
    <td>RMSE</td>
    <td>0.19163508171063265</td>
    <td>(0.17325496486953246, 0.21001519855173284)</td>
  </tr>
  <tr>
    <td>C</td>
    <td>1.0</td>
    <td>NRMSE</td>
    <td>0.31804111771478255</td>
    <td>(0.2894375624178123, 0.3466446730117528)</td>
  </tr>
  <tr>
    <td>epsilon</td>
    <td>0.01</td>
    <td>R2</td>
    <td>0.8386224413550899</td>
    <td>(0.8096632092415159, 0.8675816734686639)</td>
  </tr>
  <tr>
    <td>tol</td>
    <td>0.01</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr><td colspan="6"></td></tr>
</table>


## Key Findings and Insights

- **Model Effectiveness**:
  - *Random Forest* demonstrates excellent classification performance even with imbalanced data.
  - *SVR* accurately predicts continuous occupancy levels, offering a better understanding of the room usage.
- **Challenges**:
  - Variability in sensor data quality and inter-sensor correlations may impact prediction robustness.
  - Balancing model complexity with interpretability requires careful hyperparameter tuning.
- **Future Directions**:
  - Expand sensor modalities to further enrich the dataset.
  - Explore advanced feature extraction and data augmentation techniques to improve predictive performance.
  - Further optimize hyperparameters and experiment with other methods for enhanced accuracy.

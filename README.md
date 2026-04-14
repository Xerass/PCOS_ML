# PCOS Machine Learning Models

Welcome to the PCOS ML project. This repository contains machine learning models designed to predict and classify Polycystic Ovary Syndrome (PCOS) using two distinct modalities: Tabular Clinical Data and Ultrasound Image Data. 

## Project Overview

The project is structured into two main components, each focusing on a different type of input data to detect signs of PCOS. Both approaches utilize modern machine learning techniques and ultimately export their trained models into the ONNX format for straightforward deployment and inference, particularly focusing on mobile or application readiness.

### 1. Tabular Model (Clinical Data)
Located in the `TabularModel` directory, this component focuses on patient clinical metadata and physiological measurements.
*   **Data Characteristics:** The dataset includes features such as Age, BMI, hormonal levels (FSH, LH, TSH), and physical symptoms (weight gain, hair loss, skin darkening), as well as physical measurements like Follicle numbers and sizes.
*   **Data Augmentation:** The clinical dataset exhibited minor class imbalances. To resolve this, we utilized CTGAN to generate sythetic data points.
*   **Validation:** Principal Component Analysis (PCA) was used to verify the biological accuracy of the synthetic data, confirming that the generated samples closely match the real data distribution.
*   **Algorithm:** We implemented an XGBoost classifier, validating its performance using cross validation across the expanded dataset.
*   **Export:** The refined XGBoost model is exported to ONNX format for interoperability.

### 2. Image Model (Ultrasound Images)
Located in the `ImageModel` directory, this component analyzes ultrasound images to detect symptoms of PCOS.
*   **Algorithm:** We utilized a MobileNet Image Classifier. This lightweight architecture was deliberately chosen as its highly suitable for mobile and app deployment scenarios.
*   **Data Processing:** We applied various data transformations and augmentations to ensure the model generalizes well to "messy", real world data. 
*   **Optimization:** Custom data sanitizers and caching mechanisms were implemented to accelerate data loading and training phases.
*   **Export:** The trained PyTorch MobileNet model is exported to ONNX format.

## Findings

*   **Synthetic Data Viability:** The CTGAN approach proved highly successful. Generating synthetic tabular data points effectively mitigated class imbalance while retaining strict biological accuracy, making it an excellent bedrock for the XGBoost model.
*   **Model Suitability:** The MobileNet structure proved exceptionally capable for image classification tasks in this context while keeping computational overhead low.
*   **Efficacy:** Both ONNX models demonstrated highly confident, accurate prediction probabilities across their respective test sets, maintaining true class predictions without degradation post conversion.

## How to Use

The provided ONNX models can be seamlessly loaded into standard ONNX Runtime environments for inference. We have included tester scripts to quickly run predictions.

### Testing the Tabular Model
1. Navigate to the `TabularModel` directory.
2. Ensure you have the required dependencies (like `onnxruntime`, `pandas`, `numpy`).
3. Run the tabular test script:
   ```bash
   python onnx_tester_tabular.py
   ```
4. Check the generated `onnx_test_report_tabular.csv` to review the model's predictions output against expected labels.

### Testing the Image Model
1. Navigate to the `ImageModel` directory.
2. Ensure you have the required dependencies for image processing and ONNX.
3. Run the image test script:
   ```bash
   python onnx_tester.py
   ```
4. Review the generated `onnx_test_report.csv` for prediction certainty and correctness.

## Requirements
*   Python 3.x
*   ONNX Runtime
*   Specific python environment packages are provided in their respective directories (e.g., `requirements.txt` in the ImageModel directory).

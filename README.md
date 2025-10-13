# Smart-product-pricing-
An end-to-end ML solution that predicts product prices using catalog text and product images. Includes data preprocessing, feature engineering, model training                                                                                                                                                                 
1. Executive Summary
Our solution predicts product prices by creating a robust, multimodal feature set from text, image, and engineered data. We combine TF-IDF vectorized text, EfficientNetB0 image embeddings, and an extracted Item Pack Quantity (IPQ) feature to train a highly efficient LightGBM Regressor, achieving a validated SMAPE score of 53.86%.

2. Methodology Overview

2.1 Problem Analysis
We interpreted the challenge as a regression task requiring the model to learn complex relationships from diverse data sources. Our analysis, reflected in our feature engineering, indicated that product price is dependent on its description (pack size, brand), visual characteristics (product type, quality), and core textual content.

Key Observations:
•	Textual Importance: The catalog_content field contains rich information, including brand names, product specifications, and, crucially, package sizes (e.g., "pack of 12", "12 ct").
•	Visual Context: Product images provide non-textual cues about an item's quality, category, and potential use case, which are implicitly linked to its market price.
•	Feature Engineering Value: Explicitly extracting the Item Pack Quantity (IPQ) as a numerical feature was identified as a critical step to normalize price predictions across different package sizes.


2.2 Solution Strategy
Our strategy is to build a single, powerful model that learns from a unified representation of all available data modalities. By transforming and combining features from text and images, we create a rich input vector that allows a gradient boosting model to capture intricate patterns effectively.
Approach Type: Single Model with Hybrid Multimodal Features Core Innovation: The primary innovation is the effective fusion of three distinct feature types—NLP-derived text features (TF-IDF), computer vision-based image embeddings (EfficientNetB0), and rule-based engineered features (IPQ)—into a single sparse matrix for a LightGBM model.


3. Model Architecture

3.1 Architecture Overview
Our architecture is a sequential feature engineering pipeline that feeds into a final regression model. Text and image data are processed in parallel, concatenated with the engineered IPQ feature, and then used for training.
A simplified flow is as follows:
1.	Text Processing: Product Text -> TF-IDF Vectorizer -> [Text Features]
2.	Image Processing: Product Image -> EfficientNetB0 -> [Image Embeddings]
3.	Feature Engineering: Product Text -> IPQ Extraction -> [Engineered Feature]
4.	Fusion & Training: [Text Features + Image Embeddings + Engineered Feature] -> Concatenate -> LightGBM Regressor -> Predicted Price

3.2 Model Components

Text Processing Pipeline:
•	Preprocessing steps: Lowercasing, stop-word removal (via TF-IDF), and a custom regex-based function to extract the Item Pack Quantity (IPQ).
•	Model type: TF-IDF Vectorizer.
•	Key parameters: max_features=10000, stop_words='english'.

Image Processing Pipeline:
•	Preprocessing steps: Images are resized to 224x224 pixels and normalized according to ImageNet standards.
•	Model type: Pre-trained EfficientNetB0, used as a feature extractor (not fine-tuned).
•	Key parameters: pretrained=True, num_classes=0 (to extract feature vectors).

4. Model Performance
4.1 Validation Results
The model was evaluated on a hold-out validation set (20% of the training data) that it did not see during training.
•	SMAPE Score: 53.86%
•	Other Metrics: The model's objective was optimized for Mean Absolute Error (MAE) (regression_l1) during the training phase.
5. Conclusion
Our solution successfully demonstrates that a hybrid approach, combining features from text, images, and domain-specific engineering, provides a robust foundation for product price prediction. The LightGBM model proved highly effective at learning from the combined high-dimensional feature set, establishing a solid performance baseline. Key lessons include the significant impact of targeted feature engineering (IPQ) and the efficiency of using pre-trained deep learning models for feature extraction.
Appendix
A. Code artefacts
•	Link: https://github.com/Rit222518/Smart-product-pricing


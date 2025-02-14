# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

- **Model name:** Census Income Classification Model
- **Model type:** Binary Classification (Predicting income: >50K or <=50K)
- **Architecture:** Random Forest Classifier (n_estimators=100, random_state=42)
- **Training process:** The model was trained on data from the United States Census Bureau, predicting whether an individual's annual income exceeds $50,000 based on demographic features. The data was cleaned by removing spaces, as per the project instructions. The numerical features were scaled using StandardScaler. The categorical features were one-hot encoded, and the target variable was label binarized.

## Intended Use

- The intended use of this model is to predict whether an individual's annual income exceeds $50,000 based on demographic features.

## Training Data

- **Dataset:** United States Census Bureau data (donated on 4/30/1996).
- **Features:** Age, workclass, education, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country.
- **Size:** 32,561 instances after cleaning.

## Evaluation Data

- The model was evaluated using a hold-out test set. This test set was created by randomly splitting the original dataset into training and testing portions using an 80/20 split (80% for training, 20% for testing) with `train_test_split` from scikit-learn. A `random_state` of 42 was used to ensure reproducibility of the split. The test set was held out during training and used only for final model evaluation to assess how well the model generalizes to unseen data.

## Metrics

- **Overall Metrics (Test Set):**
    - Precision: 0.7391
    - Recall: 0.6384
    - F1-Score: 0.6851

- **Metrics on Data Slices:**

    - **Workclass:**
        - Federal-gov: Precision: 0.7971, Recall: 0.7857, F1-Score: 0.7914 (Count: 191)
        - Private: Precision: 0.7362, Recall: 0.6384, F1-Score: 0.6838 (Count: 4578)
        - ?: Precision: 0.6800, Recall: 0.4048, F1-Score: 0.5075 (Count: 389)

    - **Education:**
        - Bachelors: Precision: 0.7569, Recall: 0.7333, F1-Score: 0.7449 (Count: 1053)
        - HS-grad: Precision: 0.6460, Recall: 0.4232, F1-Score: 0.5114 (Count: 2085)
        - 7th-8th: Precision: 1.0000, Recall: 0.0000, F1-Score: 0.0000 (Count: 141)

    - **Race:**
        - White: Precision: 0.7372, Recall: 0.6366, F1-Score: 0.6832 (Count: 5595)
        - Black: Precision: 0.7407, Recall: 0.6154, F1-Score: 0.6723 (Count: 599)
        - Amer-Indian-Eskimo: Precision: 0.6000, Recall: 0.6000, F1-Score: 0.6000 (Count: 71)

    - **Sex:**
        - Male: Precision: 0.7410, Recall: 0.6607, F1-Score: 0.6985 (Count: 4387)
        - Female: Precision: 0.7256, Recall: 0.5107, F1-Score: 0.5995 (Count: 2126)

    - **Native Country:**
        - United-States: Precision: 0.7362, Recall: 0.6321, F1-Score: 0.6802 (Count: 5870)
        - ?: Precision: 0.7333, Recall: 0.7097, F1-Score: 0.7213 (Count: 125)
        - Greece: Precision: 0.0000, Recall: 0.0000, F1-Score: 0.0000 (Count: 7)

## Ethical Considerations

- **Bias and Fairness:** The training data exhibits some imbalances in representation across different demographic groups, particularly race and native country. This could lead to biased model predictions. For example, the model's performance may vary across racial groups due to the differences in sample sizes and the inherent biases present in the data itself.  The lower recall score for the "Black" racial group (0.6154) compared to the "White" group (0.6366) suggests the model might be less sensitive in correctly identifying individuals in the "Black" group who earn over $50,000.  Similarly, the low F1-score for the "Amer-Indian-Eskimo" group (0.6000) indicates potential issues with this category.  The model's performance on the "Female" sex category also reveals a lower recall (0.5107) compared to the "Male" category (0.6607), raising concerns about potential gender bias.  The significant data imbalance in the "Native Country" feature, where the vast majority of instances are "United-States," could also contribute to biased predictions for other, less represented countries.
- **Fairness in Prediction:** It is crucial to evaluate the model's predictions across different demographic groups to ensure fairness. Disparities in performance metrics across groups could indicate that the model is not making equitable predictions. Further investigation and mitigation strategies may be required.  Mitigation strategies could include collecting more data for underrepresented groups, using fairness-aware machine learning techniques, or adjusting decision thresholds for different groups.

## Caveats and Recommendations

- **Data Imbalances:** Several categories (e.g., specific education levels, less common native countries) have limited representation in the data. This can negatively impact the model's performance on these underrepresented groups.  For example, the "7th-8th" education level and the "Greece" native country have very few samples, leading to unreliable metrics.
- **Potential Model Improvements:**
    - Address data imbalances through techniques like oversampling and undersampling.
    - Explore more advanced models (e.g., gradient boosting, neural networks) to potentially improve predictive performance.
    - Conduct more thorough fairness testing and consider fairness-aware model training techniques.
    - Collect more data for underrepresented groups to improve model robustness and reduce bias.  Specifically, collecting more data for the "Amer-Indian-Eskimo" racial group and less represented native countries could improve the model's performance and reduce bias.  Further investigation is needed to determine why the "7th-8th" education level has such poor performance (zero recall) as this could point to data quality issues.
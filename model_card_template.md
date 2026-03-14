# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This repository contains a classification model leveraging the Scikit-Learn RandomForestClassifier trained to predict whether an individual's annual salary exceeds $50K (see ` apply_label()` in `./ml/data.py`). The model was developed as part of a Udacity MLOps course and is implemented using scikit-learn with preprocessing and training logic organized under the `ml` package. Trained artifacts (model and encoder) are saved in the `./model/` directory. 
## Intended Use
This model is intended for educational purposes to illustrate an end-to-end MLOps workflow, including data processing, training, evaluation, and model persistence. It is not designed or validated for production use.
## Training Data
The model was trained on census data gathered from the Census Bureau located at `./data/census.csv`. The dataset includes the following columns: age, workclass, fnlgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country, and the label `salary`. The training pipeline uses an 80/20 random train/test split (see `./train_model.py`, which calls `train_test_split()`).
## Evaluation Data
Evaluation was performed on the 20% test split derived from the same `./data/census.csv` file. In addition to overall metrics, slice-level performance is calculated for categorical features listed in the training script and written to `./slice_output.txt` for inspection.
## Metrics
The primary evaluation metrics reported for the test set are precision, recall, and F1-score. Usage of `random_state` throughout the codebase ensures deterministic output thusly:

```
Precision: 0.7419
Recall: 0.6384
F1: 0.6863
```

These metrics summarize the model's performance for predicting whether salary is greater than $50K.
## Ethical Considerations
The dataset contains demographic attributes (for example, race, sex, age, and native-country) that can reflect historical and societal biases. As a result, the model may reproduce or amplify those biases when making predictions. Practitioners should evaluate fairness across sensitive groups and attempt to mitigate bias.
## Caveats and Recommendations
This model and dataset are intended for learning and experimentation, so usage is not recommended in high-stakes decision-making. As mentioned above, efforts should be made to mitigate social bias.

Recommendations:
- Perform analysis with respect to race, gender and native country, especially when analysis informs sensitive decisions.
- Ensure human oversight and review when iterating and tuning analysis, especially when analysis informs sensitive decisions.

# Model Performance Report

Best model: **Linear SVM** (saved to `models/sentiment_model.joblib`)

## Comparison

```
                            0                    1           2              3
model             Naive Bayes  Logistic Regression  Linear SVM  Random Forest
random_accuracy           1.0                  1.0         1.0            1.0
random_precision          1.0                  1.0         1.0            1.0
random_recall             1.0                  1.0         1.0            1.0
random_f1                 1.0                  1.0         1.0            1.0
random_auc_roc            1.0                  1.0         1.0            1.0
unseen_accuracy         0.524                0.602       0.852            0.6
unseen_precision       0.5276               0.6978      0.8755         0.6163
unseen_recall          0.5318               0.5986      0.8749         0.5954
unseen_f1              0.5278               0.6286       0.875         0.6015
unseen_auc_roc         0.6373               0.7586      0.9169         0.7447
```

* `random_*` = stratified 80/20 split (optimistic - identical sentences appear in train and test)
* `unseen_*` = 5-fold group CV where test sentences never appear in training (honest estimate)

## Classification report (unseen texts)

```
              precision    recall  f1-score   support

    Negative       0.82      0.79      0.81       195
     Neutral       1.00      1.00      1.00       105
    Positive       0.80      0.83      0.82       200

    accuracy                           0.85       500
   macro avg       0.88      0.87      0.87       500
weighted avg       0.85      0.85      0.85       500
```

## Top words per class

n/a for this model type

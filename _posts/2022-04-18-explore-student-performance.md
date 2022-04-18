---
layout: post
title: Exploration Of Student Performance
subtitle: What helps determine good grades?
gh-repo: Aidzillafont/Student-Performace-
gh-badge: [star, fork, follow]
tags: [EDA, Data Science]
comments: true
---

## The Data Set
The dataset for this exploration was taken from the popular website kaggle see [source](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams).  
This data contains 5 features and 3 exam scores.

### Features
| Name | Description | Type |
| :------ |:--- | :--- |
| gender | male or female | object |
| race/ethnicity | Groups A-E | object |
| parental level of education | how educated were the students parents?  | object |
| lunch | whether the student got a free/reduced or standard lunch | object |
| test preparation course | completed or none done at all? | object |

### Exam Scores
* math score (0-100)
* reading score (0-100)
* writing score (0-100)

## The Goal
The goal of this exploration is to determine relationship of the features on the students performance. We will attempt to do this using
some groupby aggregation in pandas and some vizulizations using seaborn all in a python jupyter notebook. Finally we will construct a random forest based model
using sklearn to try to predict a given students performance.

```python
features = []
for col in df.columns:
  if df[col].dtype==object:
    features.append(col)
    print(df.groupby(col).agg(['mean']).round(2),'\n')
```

## Boxes
You can add notification, warning and error boxes like this:

### Notification

{: .box-note}
**Note:** This is a notification box.

### Warning

{: .box-warning}
**Warning:** This is a warning box.

### Error

{: .box-error}
**Error:** This is an error box.

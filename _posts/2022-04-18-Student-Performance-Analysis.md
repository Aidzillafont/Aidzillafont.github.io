---
layout: post
title: Exploration Of Student Performance Notebook
subtitle: What helps get good grades?
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
| :------ | :------ | :------ |
| gender | male or female | object |
| race/ethnicity | Groups A-E | object |
| parental level of education | how educated were the students parents  | object |
| lunch | whether the student got a reduced or standard lunch | object |
| test preparation course | completed or none done at all | object |

### Exam Scores
* math score (0-100)
* reading score (0-100)
* writing score (0-100)

## The Goal
The goal of this exploration is to determine relationship of the features on the students performance. We will attempt to do this using
some groupby aggregation in pandas and some visualizations using seaborn all in a python jupyter notebook. Finally we will construct a random forest based model
using sklearn to try to predict a given students performance.


## EDA


```python
import pandas as pd

df = pd.read_csv('https://github.com/Aidzillafont/Student-Performace-/blob/37be2cfff7c6c02ecb002231fb88e0b8647cc0b3/StudentsPerformance.csv?raw=true')

```


```python
df.head()
```





  <div id="df-14a4c87f-ed0f-4f33-9773-d4e678ec7357">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>race/ethnicity</th>
      <th>parental level of education</th>
      <th>lunch</th>
      <th>test preparation course</th>
      <th>math score</th>
      <th>reading score</th>
      <th>writing score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>female</td>
      <td>group B</td>
      <td>bachelor's degree</td>
      <td>standard</td>
      <td>none</td>
      <td>72</td>
      <td>72</td>
      <td>74</td>
    </tr>
    <tr>
      <th>1</th>
      <td>female</td>
      <td>group C</td>
      <td>some college</td>
      <td>standard</td>
      <td>completed</td>
      <td>69</td>
      <td>90</td>
      <td>88</td>
    </tr>
    <tr>
      <th>2</th>
      <td>female</td>
      <td>group B</td>
      <td>master's degree</td>
      <td>standard</td>
      <td>none</td>
      <td>90</td>
      <td>95</td>
      <td>93</td>
    </tr>
    <tr>
      <th>3</th>
      <td>male</td>
      <td>group A</td>
      <td>associate's degree</td>
      <td>free/reduced</td>
      <td>none</td>
      <td>47</td>
      <td>57</td>
      <td>44</td>
    </tr>
    <tr>
      <th>4</th>
      <td>male</td>
      <td>group C</td>
      <td>some college</td>
      <td>standard</td>
      <td>none</td>
      <td>76</td>
      <td>78</td>
      <td>75</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-14a4c87f-ed0f-4f33-9773-d4e678ec7357')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-14a4c87f-ed0f-4f33-9773-d4e678ec7357 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-14a4c87f-ed0f-4f33-9773-d4e678ec7357');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
df.dtypes
```




    gender                         object
    race/ethnicity                 object
    parental level of education    object
    lunch                          object
    test preparation course        object
    math score                      int64
    reading score                   int64
    writing score                   int64
    dtype: object




```python
df.describe()
```





  <div id="df-40970fbc-0cdc-4cf3-9f80-cff32d0b95be">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>math score</th>
      <th>reading score</th>
      <th>writing score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1000.00000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>66.08900</td>
      <td>69.169000</td>
      <td>68.054000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>15.16308</td>
      <td>14.600192</td>
      <td>15.195657</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.00000</td>
      <td>17.000000</td>
      <td>10.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>57.00000</td>
      <td>59.000000</td>
      <td>57.750000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>66.00000</td>
      <td>70.000000</td>
      <td>69.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>77.00000</td>
      <td>79.000000</td>
      <td>79.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>100.00000</td>
      <td>100.000000</td>
      <td>100.000000</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-40970fbc-0cdc-4cf3-9f80-cff32d0b95be')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-40970fbc-0cdc-4cf3-9f80-cff32d0b95be button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-40970fbc-0cdc-4cf3-9f80-cff32d0b95be');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




As can be seen there is a number of scores on maths, reading and writing. These are for varying types of people. Lets aggregate some of the groups to see if there is an things standing out.


```python
features = []
for col in df.columns:
  if df[col].dtype==object:
    features.append(col)
    print(df.groupby(col).agg(['mean']).round(2),'\n')

```

           math score reading score writing score
                 mean          mean          mean
    gender                                       
    female      63.63         72.61         72.47
    male        68.73         65.47         63.31 
    
                   math score reading score writing score
                         mean          mean          mean
    race/ethnicity                                       
    group A             61.63         64.67         62.67
    group B             63.45         67.35         65.60
    group C             64.46         69.10         67.83
    group D             67.36         70.03         70.15
    group E             73.82         73.03         71.41 
    
                                math score reading score writing score
                                      mean          mean          mean
    parental level of education                                       
    associate's degree               67.88         70.93         69.90
    bachelor's degree                69.39         73.00         73.38
    high school                      62.14         64.70         62.45
    master's degree                  69.75         75.37         75.68
    some college                     67.13         69.46         68.84
    some high school                 63.50         66.94         64.89 
    
                 math score reading score writing score
                       mean          mean          mean
    lunch                                              
    free/reduced      58.92         64.65         63.02
    standard          70.03         71.65         70.82 
    
                            math score reading score writing score
                                  mean          mean          mean
    test preparation course                                       
    completed                    69.70         73.89         74.42
    none                         64.08         66.53         64.50 
    


A quick review of the above we can make some of the following observations, in this data set we can see that:

*   Women score on average higher than men in every subject excluding maths
*   Group A perform poorly where Group E are on average score the highest
*   The higher your parents education is the higher yours should be.
*   Students who received a free/reduced lunch performed worse than those who received a standard lunch.
*   Students who dis a test preparation course performed better in exams.

Lets try an visualize some of these patterns in a few visualizations



## Visualizations


```python
import seaborn as sns
import matplotlib.pyplot as plt
#targets = ['math score', 'reading score', 'writing score']

#first we need to melt the data set to pass to out seaborn box plot
df_melt = pd.melt(df, id_vars=features, var_name='exam type', value_name='score')

for feat in features:
  plt.figure(figsize=(12,6))
  plot = sns.boxplot(x=feat,hue='exam type', y='score', data=df_melt)

```


![png](/assets/img/Student_Performance_Analysis_files/Student_Performance_Analysis_10_0.png)



![png](/assets/img/Student_Performance_Analysis_files/Student_Performance_Analysis_10_1.png)



![png](/assets/img/Student_Performance_Analysis_files/Student_Performance_Analysis_10_2.png)



![png](/assets/img/Student_Performance_Analysis_files/Student_Performance_Analysis_10_3.png)



![png](/assets/img/Student_Performance_Analysis_files/Student_Performance_Analysis_10_4.png)


The above plots visualize the spread of scores for each subject across each variable. As we can see here even though we observed different means there is a lot of overlap. This suggests that given you may have features that should reduce your score you can over come it.

However if you have every thing against you or everything for you this does seem to have a clear indication of performance.
below we can see results of the students with the worst set of features vs those with the best. Here you can see a clear separation


```python
df_worst = df[(df['gender']=='male')&(df['race/ethnicity']=='group A')&(df['parental level of education']=='high school')&(df['lunch']=='free/reduced')&(df['test preparation course']=='none')]
df_best = df[(df['gender']=='female')&(df['race/ethnicity']=='group E')&(df['parental level of education']=='master\'s degree')&(df['lunch']=='standard')&(df['test preparation course']=='completed')]
```


```python
df_worst
```





  <div id="df-0aa5ee51-50c0-4792-99a3-1f00ab15145a">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>race/ethnicity</th>
      <th>parental level of education</th>
      <th>lunch</th>
      <th>test preparation course</th>
      <th>math score</th>
      <th>reading score</th>
      <th>writing score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>395</th>
      <td>male</td>
      <td>group A</td>
      <td>high school</td>
      <td>free/reduced</td>
      <td>none</td>
      <td>48</td>
      <td>45</td>
      <td>41</td>
    </tr>
    <tr>
      <th>688</th>
      <td>male</td>
      <td>group A</td>
      <td>high school</td>
      <td>free/reduced</td>
      <td>none</td>
      <td>53</td>
      <td>58</td>
      <td>44</td>
    </tr>
    <tr>
      <th>811</th>
      <td>male</td>
      <td>group A</td>
      <td>high school</td>
      <td>free/reduced</td>
      <td>none</td>
      <td>45</td>
      <td>47</td>
      <td>49</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-0aa5ee51-50c0-4792-99a3-1f00ab15145a')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-0aa5ee51-50c0-4792-99a3-1f00ab15145a button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-0aa5ee51-50c0-4792-99a3-1f00ab15145a');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
df_best
```





  <div id="df-31be8d29-8975-4704-a5ce-3f644fbba538">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>race/ethnicity</th>
      <th>parental level of education</th>
      <th>lunch</th>
      <th>test preparation course</th>
      <th>math score</th>
      <th>reading score</th>
      <th>writing score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>685</th>
      <td>female</td>
      <td>group E</td>
      <td>master's degree</td>
      <td>standard</td>
      <td>completed</td>
      <td>94</td>
      <td>99</td>
      <td>100</td>
    </tr>
    <tr>
      <th>995</th>
      <td>female</td>
      <td>group E</td>
      <td>master's degree</td>
      <td>standard</td>
      <td>completed</td>
      <td>88</td>
      <td>99</td>
      <td>95</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-31be8d29-8975-4704-a5ce-3f644fbba538')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-31be8d29-8975-4704-a5ce-3f644fbba538 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-31be8d29-8975-4704-a5ce-3f644fbba538');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




Its worth noting that a lot of these features are out of the students control. In fact the only two that a prospective student can control is their lunch and test course preparation. So lets look at the combined effect of these in some plots and groupings and see how the separate a students performance



```python
df.groupby(['lunch', 'test preparation course']).agg(['mean'])
```





  <div id="df-c21bce10-7052-4bb4-a2af-831cbd74c420">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th>math score</th>
      <th>reading score</th>
      <th>writing score</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>mean</th>
      <th>mean</th>
      <th>mean</th>
    </tr>
    <tr>
      <th>lunch</th>
      <th>test preparation course</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">free/reduced</th>
      <th>completed</th>
      <td>63.045802</td>
      <td>69.870229</td>
      <td>70.351145</td>
    </tr>
    <tr>
      <th>none</th>
      <td>56.508929</td>
      <td>61.602679</td>
      <td>58.736607</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">standard</th>
      <th>completed</th>
      <td>73.533040</td>
      <td>76.215859</td>
      <td>76.766520</td>
    </tr>
    <tr>
      <th>none</th>
      <td>68.133971</td>
      <td>69.177033</td>
      <td>67.595694</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-c21bce10-7052-4bb4-a2af-831cbd74c420')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-c21bce10-7052-4bb4-a2af-831cbd74c420 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-c21bce10-7052-4bb4-a2af-831cbd74c420');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
pd.options.mode.chained_assignment = None 

df_melt_fn = df_melt[(df_melt['lunch']=='free/reduced')&(df_melt['test preparation course']=='none')]
df_melt_sc = df_melt[(df_melt['lunch']=='standard')&(df_melt['test preparation course']=='completed')]

df_melt_fn.loc[:,'prep and lunch']='neither'
df_melt_sc.loc[:,'prep and lunch']='both'

df_melt2 = df_melt_fn.append(df_melt_sc, ignore_index=True)
plot = sns.boxplot(x='prep and lunch',hue='exam type', y='score', data=df_melt2)

```


![png](/assets/img/Student_Performance_Analysis_files/Student_Performance_Analysis_17_0.png)


As can be seen from the above plot people who have improved there lunch and taken an test preparation course have seen better results according to the data set.

## Build a Pipeline and Model

Now we have done our EDA showing how all of the feature variables separate performance of students lets build a model to predict performance of a student in the maths subject.

First we need to make a test train split


```python
from sklearn.model_selection import train_test_split
target = ['math score']
X, y = df[features], df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=69)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
```

    (700, 5) (300, 5) (700, 1) (300, 1)



```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

pipe = Pipeline([('hot', OneHotEncoder()), ('rf', RandomForestRegressor())])

param_grid = {
    "rf__n_estimators": [300, 500, 1000],
    "rf__criterion": ['absolute_error'],
    "rf__max_depth": [None,2,3,4],
}


cv = GridSearchCV(pipe, param_grid, n_jobs=-1, cv=3, verbose=10, scoring='neg_mean_absolute_error')
cv.fit(X_train, y_train.values.ravel())

print('The best score was: ', cv.best_score_, ' with the following parameters.\n')
print(cv.best_params_)
```

    Fitting 3 folds for each of 12 candidates, totalling 36 fits
    The best score was:  -11.100714033886424  with the following parameters.
    
    {'rf__criterion': 'absolute_error', 'rf__max_depth': 4, 'rf__n_estimators': 300}



```python
import numpy as np
from sklearn.metrics import mean_absolute_error
preds = cv.best_estimator_.predict(X_test)
#print(np.sqrt(np.mean(np.power(preds - y_test.values,2))))
mean_absolute_error(preds,y_test)
```




    10.68906111111111



Here we can see our model gets a absolute error of 11.1 and 10.69 on train and test respectively. This means our model was on average predicting a score around 11 percentage points away from our true score.


Lets change our regression problem into a classification problem by changing scores to grades following the below schema

Grade|Score
---|---
A| 90-100
B| 70-89
C| 50-69
D| 40-49
F| 0-39

In python we can very easily build a function to do this.


```python
def grade_conv(x):
  if x>=90:
    return 'A'
  elif x>=70:
    return 'B'
  elif x>=50:
    return 'C'
  elif x>=40:
    return 'D'
  else:
    return 'F'
```

apply the function to our y_test and y_train


```python
y_train_grade = y_train[target[0]].apply(lambda x: grade_conv(x))
y_test_grade = y_test[target[0]].apply(lambda x: grade_conv(x))
```

Below we build a random forest classifier and run a grid search cross validation to train our hyper parameters.


```python
from sklearn.ensemble import RandomForestClassifier

pipe = Pipeline([('hot', OneHotEncoder()), ('rf', RandomForestClassifier())])

param_grid = {
    "rf__n_estimators": [300, 500, 700],
    "rf__criterion": ['gini', 'entropy'],
    "rf__max_depth": [None,2,3,4],
    "rf__max_features": ['auto','sqrt','log2',None]
}


cv = GridSearchCV(pipe, param_grid, n_jobs=-1, cv=3, verbose=10)
cv.fit(X_train, y_train_grade)

print('The best score was: ', cv.best_score_, ' with the following parameters.\n')
print(cv.best_params_)
```

    Fitting 3 folds for each of 96 candidates, totalling 288 fits
    The best score was:  0.49283469669735763  with the following parameters.
    
    {'rf__criterion': 'gini', 'rf__max_depth': 2, 'rf__max_features': 'sqrt', 'rf__n_estimators': 700}



```python
from sklearn.metrics import accuracy_score
preds = cv.best_estimator_.predict(X_test)
accuracy_score(preds,y_test_grade)
```




    0.53



here we get an accuracy score of 53% which is only slightly better than guessing the most common class the C grade

## Conclusions

Where the random forest model did not perform as well as hoped we can certainly look to investigate more predictive models in the future.

As for insights to the data they are summarised by:

*   Women score on average higher than men in every subject excluding maths
*   Group A perform poorly where Group E are on average score the highest
*   The higher your parents education is the higher yours should be.
*   Students who received a free/reduced lunch performed worse than those who received a standard lunch.
*   Students who did a test preparation course performed better in exams.

We saw these by looking at some bar-plots using seaborn along with some groupby in pandas

If you would like to check out this notebook yourself please take a look at my [github](https://github.com/Aidzillafont/Student-Performace-)

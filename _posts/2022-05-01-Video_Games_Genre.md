---
layout: post
title: Visualizations of Video Games sales and features
subtitle: Using visualizations to explore video game data using plotly and pandas
gh-repo: Aidzillafont/Video-Games-
gh-badge: [star, fork, follow]
tags: [EDA, Data Science, plotly, pandas]
share-img: /assets/img/Video_Games/gamer.png
thumbnail-img: /assets/img/Video_Games/gamer.png
comments: true
---

# Data Exploration

Our goal here is to examine some factors in our dataset and see if we can visualize some of the relationships between variables. We will be using a dataset taken from the kaggle website. See [here](https://www.kaggle.com/datasets/gregorut/videogamesales) for the source

# The Dataset

From below output we can see the dataset has 11 columns 6 being features of the game such as platform or genre and 5 being sales figures. 


```python
import pandas as pd
df = pd.read_csv('https://github.com/Aidzillafont/Video-Games-/blob/843e8c4c47db94fc39b083f82226d4b88c8924a1/vgsales.csv?raw=true')
df.dtypes
```




    Rank              int64
    Name             object
    Platform         object
    Year            float64
    Genre            object
    Publisher        object
    NA_Sales        float64
    EU_Sales        float64
    JP_Sales        float64
    Other_Sales     float64
    Global_Sales    float64
    dtype: object



# Visualizations
Lets check the genres and see how genres rank on global sales


```python
#lets check the best selling genre of all time
best_genre_df = df.iloc[:,4:].groupby(['Genre']).sum().sort_values(['Global_Sales'], ascending=False)
best_genre_df
```





  <div id="df-36feb7e5-59d3-44e0-ab5f-74271223c975">
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
      <th>NA_Sales</th>
      <th>EU_Sales</th>
      <th>JP_Sales</th>
      <th>Other_Sales</th>
      <th>Global_Sales</th>
    </tr>
    <tr>
      <th>Genre</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Action</th>
      <td>877.83</td>
      <td>525.00</td>
      <td>159.95</td>
      <td>187.38</td>
      <td>1751.18</td>
    </tr>
    <tr>
      <th>Sports</th>
      <td>683.35</td>
      <td>376.85</td>
      <td>135.37</td>
      <td>134.97</td>
      <td>1330.93</td>
    </tr>
    <tr>
      <th>Shooter</th>
      <td>582.60</td>
      <td>313.27</td>
      <td>38.28</td>
      <td>102.69</td>
      <td>1037.37</td>
    </tr>
    <tr>
      <th>Role-Playing</th>
      <td>327.28</td>
      <td>188.06</td>
      <td>352.31</td>
      <td>59.61</td>
      <td>927.37</td>
    </tr>
    <tr>
      <th>Platform</th>
      <td>447.05</td>
      <td>201.63</td>
      <td>130.77</td>
      <td>51.59</td>
      <td>831.37</td>
    </tr>
    <tr>
      <th>Misc</th>
      <td>410.24</td>
      <td>215.98</td>
      <td>107.76</td>
      <td>75.32</td>
      <td>809.96</td>
    </tr>
    <tr>
      <th>Racing</th>
      <td>359.42</td>
      <td>238.39</td>
      <td>56.69</td>
      <td>77.27</td>
      <td>732.04</td>
    </tr>
    <tr>
      <th>Fighting</th>
      <td>223.59</td>
      <td>101.32</td>
      <td>87.35</td>
      <td>36.68</td>
      <td>448.91</td>
    </tr>
    <tr>
      <th>Simulation</th>
      <td>183.31</td>
      <td>113.38</td>
      <td>63.70</td>
      <td>31.52</td>
      <td>392.20</td>
    </tr>
    <tr>
      <th>Puzzle</th>
      <td>123.78</td>
      <td>50.78</td>
      <td>57.31</td>
      <td>12.55</td>
      <td>244.95</td>
    </tr>
    <tr>
      <th>Adventure</th>
      <td>105.80</td>
      <td>64.13</td>
      <td>52.07</td>
      <td>16.81</td>
      <td>239.04</td>
    </tr>
    <tr>
      <th>Strategy</th>
      <td>68.70</td>
      <td>45.34</td>
      <td>49.46</td>
      <td>11.36</td>
      <td>175.12</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-36feb7e5-59d3-44e0-ab5f-74271223c975')"
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
          document.querySelector('#df-36feb7e5-59d3-44e0-ab5f-74271223c975 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-36feb7e5-59d3-44e0-ab5f-74271223c975');
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




Here we can see that Action sells the most and Strategy sells the least. Lets visualize this


```python
import plotly.express as px

fig = px.bar(best_genre_df,  x=['NA_Sales',	'EU_Sales',	'JP_Sales',	'Other_Sales'], y=best_genre_df.index, title="Best Genre by Sales", text_auto=True, width=1250)
#to reverse order so action is at the top
fig.update_yaxes(autorange="reversed")
fig.show()
```


![png](/assets/img/Video_Games/img_1.png)


From the above visualization we can see not only the highest selling genre but how they sell in each region. Something to note here is North America is responsible for most sales in every genre bar Role-Playing. The best region to sell role playing is Japan according to the data.

So we know the best selling genre for games per region but what platforms have the most games? Are specific genres being targeted by specific platforms? We will look to answer these questions and more with some visualizations.

But first we are gonna need to construct a dataframe to examine this! 


```python
df_genre_by_platform = df[['Genre', 'Platform','Rank']].groupby(['Genre', 'Platform']).count()
#reseting the index here so we can use column names in our plotly sunburst
df_genre_by_platform.reset_index(inplace=True)
#rename rank to represent what it really is count of games on that platform
df_genre_by_platform.rename(columns={'Rank':'# of Games'}, inplace=True)

fig = px.sunburst(df_genre_by_platform, path=['Genre', 'Platform'], values='# of Games', width=500, title='Sunburst of Genres and Platforms # of games')
fig.show()

fig = px.sunburst(df_genre_by_platform, path=['Platform','Genre'], values='# of Games', width=500, title='Sunburst of Genres and Platforms by # of games')
fig.show()
```


![png](/assets/img/Video_Games/img_2.png)

![png](/assets/img/Video_Games/img_3.png)


Oh no! This visualization looks a little squishy. Sunburst can be a great visualization tool but it has a downside being when you have too many categories then it becomes all squished together and loses some of its eligibility.  
Fortunately for us plotly is interactive so if you check out my python notebook you can click into the graph yourself and zoom in to categories. If you clicked Action you would see something a little like this.


```python
fig = px.sunburst(df_genre_by_platform[df_genre_by_platform['Genre']=='Action'], path=['Genre', 'Platform'], values='# of Games', width=500)
fig.show()
```


![png](/assets/img/Video_Games/img_4.png)


Lets try and get a less squishy view of everything by using a bubble chart. In the bubble chart below the size of the bubble is is gong to be the number of games on that platform genre pair.


```python
fig = px.scatter(df_genre_by_platform, x='Platform', y='Genre',
	         size='# of Games', color='Platform', width=1500, title='Number of Games on Platform Genre Pairs')
#to reverse order so action is at the top
fig.update_yaxes(autorange="reversed")
fig.show()
```


![png](/assets/img/Video_Games/img_5.png)


Cool so the bubble lays it out with more clarity but you lose proportionality that you with a sunburst or pie chart.

It is interesting to note here that there are some platforms where there really is not alot of games on for example the PCFX only has 1 game! 

Its also worth noting that some platforms really don't cater to some game genre for example all the playstation platforms don't really have many puzzle games.

So with the combination of the bubble and sunburst we can see that most genres are dominates by only a handful of platforms and some platforms really have most of the games. Lets visualize that second part in a sunburst

From the above we can see nearly half of all games are on only 5 platforms.

So lets summarize what we have found:

*   The best selling games are Action games 
*   Only 5 platforms account for half of all the games
  * Those platforms being DS, PS2, PS3, Wii and X360
* Different platforms cater more to different genres
  * for example DS has the most simulation, puzzle and misc games

This is for the number of games on platforms what about sales?


```python
df_gp_sales = df[['Genre', 'Platform', 'Global_Sales']].groupby(['Genre', 'Platform']).sum()
#reseting the index here so we can use column names in our plotly sunburst
df_gp_sales.reset_index(inplace=True)

fig = px.sunburst(df_gp_sales, path=['Genre', 'Platform'], values='Global_Sales', width=500, title='Sunburst of Genres and Platforms by sales')
fig.show()

fig = px.sunburst(df_gp_sales, path=['Platform', 'Genre'], values='Global_Sales', width=500, title='Sunburst of Genres and Platforms by sales')
fig.show()
```


![png](/assets/img/Video_Games/img_6.png)


![png](/assets/img/Video_Games/img_7.png)

We can see from above similar patterns when compared to game count. This makes sense since generally if there are more games to be sold you would expect that there would be proportionally more games sold overall.

So if your a gamer who plays action games mainly you probably want to stick with playstation or xbox.

Lets look at how games sold at different genres over time and see if any genres are growing in popularity


```python
#df_sale_over_time = df.gro
df_sale_over_time = df[['Genre', 'Year','Global_Sales']].groupby(['Genre', 'Year']).sum()
df_sale_over_time.reset_index(inplace=True)
df_sale_over_time = df_sale_over_time.sort_values(['Year'], ascending=(True))

fig = px.line(df_sale_over_time, x='Year', y='Global_Sales', color='Genre')
fig.show()
```


![png](/assets/img/Video_Games/img_8.png)


Whats this has there been a collapse in the gaming industry over the last decade. Well no its really just a result of data set not having a much games from later years. Lets check this out with out another line chart.


```python
df_number_of_over_time = df[['Year','Global_Sales']].groupby(['Year']).count()
df_number_of_over_time.reset_index(inplace=True)
df_number_of_over_time.rename(columns={'Global_Sales':'# of Games'}, inplace=True)
df_number_of_over_time = df_number_of_over_time.sort_values(['Year'], ascending=(True))


fig = px.line(df_number_of_over_time, x='Year', y='# of Games')
fig.show()
```


![png](/assets/img/Video_Games/img_9.png)


As can be seen here the data set seems to have reduced collection of data in the early 2000's. I can say for certain 2017 onward is not a true picture of the gaming industry.

Our original line cart has a lot going on and is hard to parse visually so lets make a clear graph to show what genres sold more when


```python
fig = px.bar(df_sale_over_time,  x='Year', y='Global_Sales', color='Genre', title="Best Genre by Sales over time",  width=1250)
fig.show()
```


![png](/assets/img/Video_Games/img_10.png)


This is an improvement on line chart but lets make our y value a % of sales in a year so we can see what genres became popular in which years.

Below we use .transform() in pandas to divide the global sales figure by the sum of global sales in the given year and then store that number in '% of yearly sales' to get our proportion


```python
df_sale_over_time['% of yearly sales']=100*df_sale_over_time['Global_Sales']/df_sale_over_time.groupby(['Year'])['Global_Sales'].transform('sum')

fig = px.bar(df_sale_over_time,  x='Year', y='% of yearly sales', color='Genre', title="Best Genre by Sales over time",  width=1250)
fig.show()
```


![png](/assets/img/Video_Games/img_11.png)


This more clearly shows the decline in the sale of puzzle games over the year and rise of popularity of action and shooter games in the 2000's

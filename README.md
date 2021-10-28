# New York Taxi Spending Analysis
In this project, I applied and demonstrated the data science pipeline to explore and use deterministic features to predict how much a cab driver can earn per hour in different areas of New York. The knowledge of which areas earn more or earn less in any given day and any given hour allows  the union to distribute cab assignments more equitably across different areas, and to rotate cab drivers between higher and lower-income areas on a daily basis. This way, there will not be cab driver would dominate in a high-income area while another cab driver would be continuously evicted to a low-income area. The data set used can be retrieved in [TLC Trip Record Data - TLC](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page). 

## Problem definition
Predict the average money spent on taxi rides for different regions of New York per given day and hour. This problem is a supervised regression problem and we are trying to predict a continuous variable. 

## Data Cleaning 
![Negative and zero values graph](/images/Negative_and_zero_values_graph.png)

1. Negative total_amount values: It has been removed as it does not make sense to see negative money values. I found that those negative numbers are highly associated with abnormal payment types, such as dispute, voided trip etc. 

2. Zero total_amount values: It has been removed as it is not normal to ride a a cab paying zero dollars. I found that those zeros are highly associated with extremely low trip milage in certain pick-up locations. 

![Graph showing the too high values](/images/too_high_values_graph.png)

3. Too-high values for total_amount: It has been removed as it is unlikely for taxi fare to be $600,000. This is an outlier that will skew the distribution positively. I decided to impose an upper limit of $200 as there were only 1155 data points higher than that value, meaning that it will not cause a great loss of information while enhancing the validity of the data. 

8,302 data points has been removed from the original data set of 7,667,792 rows.

## Original features of the model
The original data set included the following features that can be used for model development: 

[‘PULocationID’, ‘transaction_date’,’ transaction_month’,’ transaction_day’, ‘transaction_hour’, ‘trip_distance’,’ total_amount’, ‘transactions_aggregated’]

You can refer to the [Data dictionary](https://www1.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf)  for the meaning of each feature.

Benchmark Model Performance using decision tree algorithm:

| Algorithm         | MAE   | RMSE   | R2    |
|-------------------|-------|--------|-------|
| Benchmark model   | 9.758 | 14.704 | 0.215 |

## Feature engineering
I’ve added 3 sets of new features to the model. 

1. First set is time-based feature. These include, weekend and holiday boolean.

2. Second set is location-based information. We have Location IDs per region but there is a higher level abstraction for regions called Boroughs. This information came from the source of the main data.

3. The third set is weather related data. I’ve downloaded this data from [New York, New York, United States of America Historical Weather Almanac](https://www.worldweatheronline.com/new-york-weather-history/new-york/us.aspx). 

Here is a list of all features used in the final model: 
[‘PULocationID’, ‘transaction_month’, ‘transaction_day’, ‘transaction_hour’, ‘transaction_week_day’, ‘weekend’, ‘is_holiday’, ‘Borough’, ‘temperature’, ‘humidity’, ‘wind speed’, ‘cloud cover’, ‘amount of precipitation’, ‘total_amount’]

## The algorithms applied and the results
I applied Decision Tree with the original features as the benchmark model.

I also applied the newly added features for the normal models, including Decision tree, Random forest and Gradient boosting. 

Here are the performance results before tuning: 

| Algorithm         | MAE   | RMSE   | R2    |
|-------------------|-------|--------|-------|
| Benchmark model   | 9.778 | 14.739 | 0.225 |
| Decision tree     | 8.534 | 14.011 | 0.308 |
| Random forest     | 7.426 | 13.212 | 0.385 |
| Gradient boosting | 8.388 | 13.378 | 0.369 |

The Random Forest model is selected to be tuned. Here are the best parameter values: n_estimators=600,min_samples_split=10,min_samples_leaf=2,max_features='sqrt',max_depth=500,bootstrap=False

The performance compares to previous models is:

| Algorithm           | MAE   | RMSE   | R2    |
|---------------------|-------|--------|-------|
|   Benchmark model   | 9.778 | 14.739 | 0.225 |
| Decision tree       | 8.534 | 14.011 | 0.308 |
| Random forest       | 7.426 | 13.212 | 0.385 |
| Gradient boosting   | 8.388 | 13.378 | 0.369 |
| Tuned Random forest | 7.275 | 12.507 | 0.432 |

We can see a notable increase in R squared after tuning our Random forest model, R2 increases from 0.385 to 0.429. 

Here is the True vs. Predicted value plot for the tuned random forest model. X-axis is the true values and y-axis the predicted values.

![Performance graph of tuned Random Forest](/images/tuned_random_forest_graph.png)

## Next steps
As you can see from the plot above, the performance can be improved by adding the 3 features mentioned above. However, three are other ways that wasn’t tried in this notebook:

1. Limiting the region/borough included in this analysis. This might be a good action to take depending on the problem at hand. If the goal is to make predictions only in Manhattan region, we may increase the model performance by only keeping data points in Manhattan borough. 

2. Filtering out the location ID that do not normally get a lot of taxi traffic in any given hour. If the goal is solely to increase model performance, only including data points with location ID that has at least 3 or more aggregated_transactions may increase the performance, because we are making sure that each row is an average of taxi income based on multiple records, not just based on one or two records that may or may not represent a solid trend in a given area. Note that Manhattan borough alone already has 68 unique location IDs. Among the 45,309 entries, there are just 2,249 entries with hourly aggregated transaction less than 3. As a result, the filtering criteria will base on the problem definition.
For example, if the model has to work in all region of NYC, then we cannot filter out all entries with aggregated counts less than 6 because 6 is the median across all region, removing them may add a risk of removing representable data points. In contrast, if you want the model to only work in Manhattan, then we can take into account that the mean aggregated counts is 152, median is 83, so the filtering criteria will be relatively higher.

## Problem transformation: Classification 
I can also transform the problem into a classification type problem, so the model can be used for classifying regions with high earning class or low earning class. The target feature will be binary. 

In this case, I used an average earnings of 15.5 as a separation point between high and low earning class.  By using exactly the same features from above, [‘PULocationID’, ‘transaction_month’, ‘transaction_day’, ‘transaction_hour’, ‘transaction_week_day’, ‘weekend’, ‘is_holiday’, ‘Borough’, ‘temperature’, ‘humidity’, ‘wind speed’, ‘cloud cover’, ‘amount of precipitation’, ‘total_amount’], the Random forest classifier produced the following performance results:

| Algorithm                 | Accuracy | Precision | Recall |
|---------------------------|----------|-----------|--------|
| Random Forest Classifier  | 0.748    | 0.726     | 0.771  |

| Confusion_matrix   | Actually Positive | Actually Negative |
|--------------------|-------------------|-------------------|
| Predicted Positive | 12702             | 4749              |
| Predicted Negative | 3744              | 12637             |

The accuracy score of 74.8% means the model could make 74 correct predictions out of 100 total examples. The model has slightly higher Recall score than the precision score, that means the model tends to allow more false positive, in other words it tends to classify an area to be high earning class when it is low in reality.  

With this model, the union could arrange the cab assignment even more easily because the output is simple. The regression model in the first section is good for overviewing the income predictions for each region, while the classifier is good for defining the region beforehand with simple highs or lows then make data-driven cab arrangement based on the output. 

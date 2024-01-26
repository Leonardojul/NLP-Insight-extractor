## NLP Insight extractor for document collections

### Description

<img src ="https://raw.githubusercontent.com/Leonardojul/NLP-Insight-extractor/main/Collection of documents.png" width="100%" height="100%">

Description:
The Insight Extractor Engine leverages the Language-agnostic BERT Sentence Encoder (LaBSE) to efficiently categorize and subcategorize large document collections, allowing its users to query a document collection on topic and sub-topic.

The algorithm includes data extraction, pre-processing, and LaBSE-based embedding, with results stored in a database. Users define categories, sub-categories, and keywords to query the document pool, receiving categorized and sub-categorized results along with topic distributions.

Originally developed to aid Basic-Fit in understanding customer inquiries, this solution can be used to categorize and sub-categorize any set of documents based on custom topics and sub-topics.

**INDEX**
1. [How does it work?](#how-does-it-work)
    - [Data collection](#data-collection)
    - [Pre-processing and embedding](#pre-processing-and-embedding)
    - [Topic enquiring](#topic-definition-and-enquiring)
2. [Delivery and reporting](#delivery-and-reporting)
3. [Conclussion](#conclussion)


### How does it work?

The Forecasting System is a series of processes and subprocesses that:
1. Gets the data from different places
2. Processes this data to remove any artifacts or effects we do not want to be carried over to the forecast
3. Produces a forecast for a given timeframe
4. Processes this forecast to add any effects known in advance, such as holidays effects or small corrections
5. Saves the forecast where it can be retrieved in the future by other programs or processes

The following flowchart summarizes the building blocks of the system:



### Data collection

In an ideal scenario, all the data we need four our data science project would be directly ready for us to use. As many times, this was not the case for us and the channels and languages we needed to forecast come from different places, requiring different methods and integrations to get each one of them.

For security reasons I will not be sharing the speecifics of how this data was retrieved, but suffice to say that it was distributed between:
1. An internal SQL server database
2. An Azure storage account (where there is an excel file from which we retrieve the data)
3. A custom thir-party API

The packages used for these three processes were:
``` python
import pyodbc
from azure.identity import DefaultAzureCredential, AzureCliCredential
```
``` python
from azure.storage.blob import BlobServiceClient, __version__
```
``` python
import requests
```

### Pre-processing and embedding

Before using the historical data to produce any forecasts we need to make sure that all the data present in the time-series is representative of the distribution we are working with and, therefore, we can extrapolate it to the future. For that we will remove any outliers. Considering that we are working with a poisson distributio, we will divide these outliers into two categories:

1. Values significantly higher than the baseline
2. Values abnormally lower than the baseline

For case 1 we will use:

![](https://latex.codecogs.com/svg.image?x_%7Bi%7D%20%3E%201.5%5Ctimes%20IQR)

For this we just need to calculate the IQR (interquartile range) of our historic data, and compare each reading to this value. All those readings that are **higher** than this will be considered outliers. All those values will be replaced by the median of the distribution. Here is the code to achieve it:

``` python
def detect_outliers(column: pd.DataFrame):
        """
        Detects outliers in a single-column dataframe, based on the 1.5*IQR method
        Args:
            column (pd.dataframe): Single-column dataframe to be used
        Returns:
            Single-column dataframe with the collection of outliers found
        """

        if column.empty:
            return column
        
        x = column.to_numpy(dtype=int)

        #Calculate interquartile range
        lqt, hqt = np.quantile(x,[0.25,0.75])

        iqr = hqt - lqt

        #Calculate outlier detection upper bound (we won't bother with the lower bound for now
        #as this specific dataset cannot have negative values and therefore is left skewed)
        upper_bound = hqt + 1.5*iqr

        #Return a dataset with outliers only
        return column.loc[column[column.columns[0]] >= upper_bound]
```

For case 2 we use a different method. Since what we want to remove are the effects caused by bank holidays (lower contacts than usual) we forecast what would have been a "normal" day and then use that forecast to substitute the abnormally low value:

<img src="https://raw.githubusercontent.com/Leonardojul/Forecasting-System/main/holiday-graph-example.jpg" width="100%" height="100%">

As we can see, on the third Wednesday of this time series we got an abnormally low number of contacts. Since we know this was a bank holiday, we can "forecast the past" to find out what would have happened should there had not been a bank holiday on that Wednesday. We will then use the "forecasted" value to imputate the data for a more realistic time-series and avoid carrying over this effect into our forecast:

<img src="https://raw.githubusercontent.com/Leonardojul/Forecasting-System/main/holiday-graph-corrected.jpg" width="100%" height="100%">

### Topic definition and enquiring

After our data is clear from any undesired effects that could pollute our forecasts it is time to fit a model and forecast. A question often asked in this regard is what is the besst set of hyperparameters to fit our model with. The statsmodels package in Python provides an automatic way to finding these out, which is the function auto_arima. Since we wanted to implement our own testing criteria, we decided to create our own auto s-arima function:

``` python
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
def auto_sarimax(oc_train: pd.DataFrame, p: int = 3, 
                 q: int = 3, P: int = 3, Q:int = 3):
    """
    Runs a sarimax fitting for values from 0 to (parameter - 1) for p, q, P and Q, calculates AIC and BIC values for each
    forecast and then saves the results to Test_results.csv. Accepts only single-column DataFranes. The number of tests
    will be p*q*P*Q. By default these values are 3, therefore auto_sarimax will perform 81 different tests.

    Args:
        oc_train (pd.DataFrame): Single-column dataset to be analyzed.
        p_arima (int): Autoregressive (AR) component of model expressed in order p
        q_arima (int): Moving average (MA) component of model expressed in order q
        P_arima (int): Seasonal autoregressive (SAR) component of model expressed in order P
        Q_arima (int): Seasonal moving average (SMA) component of model expressed in order Q

    Returns:
        Test_results.csv

    Example:

        .. code-block:: python

            import pandas as pd
            test_data = {'Daily_volumes':[1,2,3,2,1,2,3,2,1,2,3,2,1,2,3,2,1,2,3,2,1,2,3]}
            date_generated = pd.date_range(start='1/1/2022',end='1/23/2022')
            test_dataframe = pd.DataFrame(test_data, index=date_generated)
            auto_sarimax(test_dataframe)
    """

    # Create empty list to store search results
    order_aic_bic=[]

    # Loop over p values from 0-2
    for p1 in range(p):
    # Loop over q values from 0-2
        for q1 in range(q):
            # create and fit ARMA(p,q) model
            for P1 in range(P):
                for Q1 in range(Q):
                    model = SARIMAX(oc_train, order=(p1,1,q1), seasonal_order=(P1,1,Q1,7), freq="D") 
                    results = model.fit()  
            # Append order and results tuple
                    order_aic_bic.append((p1,q1,P1,Q1,results.aic, results.bic))
                    print("Calculated with p= ", p1," q= ",  q1, " P= ", P1, " Q= ", Q1)

    # Construct DataFrame from order_aic_bic
    order_df = pd.DataFrame(order_aic_bic, 
                            columns=['p','q','P','Q','AIC','BIC'])

    order_df.to_csv(f'Test_results_{date.today()}.csv'
```
### Post-processing

Once our forecast has been produced it is time to correct any undersired artifacts inherent to this forecasting method. One of them is the forecasting of negative values. In the specific context of a contact centre it does not make sense to have -100 calls in a day or -50 emails received. Usually extreme negative values in your forecast could indicate your are no correctly fitting your model, but small incursions into negative territory are just a normal effect of a model that takes into account the linear trend of your data.

The first process to fix this will be to trim any negative values, which we can do with a single line of code:

``` python
  predictions[predictions < 0] = 0
```

We will also make sure that any holidays or weekends in which our contact centre is closed our predictions will be zero. For that we will take into account the weekday and the holiday calendar, which we can find using the holidays package:

``` python
 import holidays
 def add_holidays(prediction: pd.DataFrame, country: str):
    """
    Zeroes any values that fall on holidays or weekends

    Args:
        prediction (pd.dataframe): Training dataset to be processed
        country (str): International country code in upper case

    Returns:
        prediction (pd.dataframe): Processed prediction

    Example:

        .. code-block:: python

            import pandas as pd
            test_data = {'Daily_volumes':[1,2,3,2,1,2,3,2,1,2,3,2,1,2,3,2,1,2,3,2,1,2,3]}
            date_generated = pd.date_range(start='4/8/2022',end='4/30/2022')
            test_dataframe = pd.DataFrame(test_data, index=date_generated)
            predict_X_plus_5(test_dataframe)
    """
    #Generate specific holidays calendar
    holiday = holidays.country_holidays(country)

    #Check whether there are any holidays or weekends in the generated prediction and zero those values
    for index, i in prediction.iterrows():
        if index in holiday or index.weekday() >= 5:
            prediction.loc[index, prediction.columns[0]] = 0


    return prediction
```
### Delivery and reporting

Once our historical data and forecasts are safely stored in our SQL database, it is time to make sure that the different business units within the organization can make use of this data.

The most straightforward way to share the data among non-extremely tech savvy professionals is to create a Power BI dataset, which can be easely used from any other applications than Power BI like the widely used MS Excel.

In the case of a Power BI dashboard, it is important to provide the end users with controls and slicers that make their lives easier in using and understanding the data, which will also have an impact on the amout of value they can extract from it. Consider this: an end user looking at some predictions made by a fancy technologic algorithm will have the feeling of peeking into the future. They will want to look further in time, but also check how well the system performed in the past. They will also play with the controls to find out whether their assumptions are right or if the system "agrees' with them on what the future awaits. And this is just the surface, this data will be used to take important decisions like granting time off, scheduling more agents on a given channel or even hiring.

<img src="https://raw.githubusercontent.com/Leonardojul/Forecasting-System/main/FC-dashboard1.png" width="75%" height="75%">

On the main page of our dashboard we can see a time-series with a daily granularity in which actuals (blue line) forecast from 4 weeks ago (orange line) and most recent forecast (purple line) are superimposed so it is easy to follow the systems' prediction and accuracy. It is also possible to "zoom in" or "out" on a given time period with the time slicers or to filter on specific channels, languages or a combination.

<img src="https://raw.githubusercontent.com/Leonardojul/Forecasting-System/main/FC-dashboard3.png" width="75%" height="75%">

The second screenshot shows a similar page, but in this case, aggregated weekly. Not all business units require the same level of detail for their operation and weekly or monthly results are more useful for evaluating both the system and the efficency of the contact centre. In this case we can also see a custom tooltip we built to give more clarity on the data displayed. This specific example shows how in week 17 an abnormally high volume of phone calls was received. This, of course, could not have been forecasted as it was the result of an unforseen issue. For the sake of our own peace of mind and that of our stakeholders, we can add information to this visualization that helps clarify the anomaly. In this case a mistake from another department explains the deviation from the forecasted values.

<img src="https://raw.githubusercontent.com/Leonardojul/Forecasting-System/main/FC-dashboard2.png" width="75%" height="75%">

Finally, and among other pages I will not be showcasing in this document, we need to clearly show our models' performance, which translates directly into how much it can be trusted. The choice of the right metric(s) is crucial in this case as it has to be both useful from a technical standpoint, as well as easy to understand to everybody. Measures such as RMSE, although useful and perhaps more scientific, are not easy to read, understand or explain to all stakeholders. For this reason we have chosen to implement the mean absolute percentage error (MAPE):

![](https://latex.codecogs.com/png.image?\dpi{110}&space;\bg_white&space;M.A.P.E&space;=&space;\frac{1}{n}\sum_{t&space;=&space;1}^{n}\left&#124;1-\frac{Ft}{At}&space;\right&#124;)

Along with the mean percentage error (MPE):

![](https://latex.codecogs.com/png.image?\dpi{110}&space;\bg_white&space;M.P.E&space;=&space;\frac{1}{n}\sum_{t&space;=&space;1}^{n}\1-\frac{Ft}{At}&space;)

Where Ft are our forecasted values and At are our actual values.

MAPE is easy to understand as it is nothing but an aggregation of all the errors in a correlection of forecasted and actual value pairs. It is useful to check whether the forecast is within the 10% error target. MPE, on the other hand, helps us complete the story by showing whether our errors are skewed in one direction and, if so, where.

### Conclussion

The above project has been designed and implemented over time, using different technologies and systems to tailor it to very specific needs and to what was possible within an organization. Every forecasting tool will have its own particularities and challenges, and that is the reason why it is important to be flexible during both the design and implementation phase. Listening to your stakeholders and making the most out of the tools and infrastructure available should be the priorities of any respectable data scientist.

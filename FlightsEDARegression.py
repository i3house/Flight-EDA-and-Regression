#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# prints all columns of a dataframe. Does not truncate them.
pd.set_option('display.max_columns', None)

master_data = pd.read_csv('flights.csv')

# EXPLORATORY DATA ANALYSIS

# Number of observations and features
print(master_data.shape[0])
print(master_data.shape[1])

# Different airlines and their counts
print(master_data['AIRLINE'].nunique())
print(master_data['AIRLINE'].value_counts())

# Create a copy of the master data to clean
df = master_data.copy()  

# Number of missing values in departure delay
print(df['DEPARTURE_DELAY'].isnull().sum())

# Number of missing values in arrival delay
print(df['ARRIVAL_DELAY'].isnull().sum())

# It seems that all missing departure delays were cancelled
print(df[df['DEPARTURE_DELAY'].isnull()]['CANCELLED'].value_counts())

# seems that all missing arrival delays where departure delay is not missing
# are either due to diverted or cancelled flights
print(df[(df['ARRIVAL_DELAY'].isnull()) & (df['DEPARTURE_DELAY'].notnull())])

# Let's drop these observations and reset the index
df.dropna(subset=['ARRIVAL_DELAY'], inplace=True)
df.reset_index(drop=True, inplace=True)

# Average and Median Departure delays
print(df['DEPARTURE_DELAY'].describe())

# Average and Median Arrival delays
print(df['ARRIVAL_DELAY'].describe())

# Looking at the above it seems that both departure delay and 
# arrival delay are positively skewed, which makes sense

# Lets graphically view the departure delays and arrival delays for
# each airline

# Box plots for departure delays for each airline
plt.figure(figsize=(15, 10), dpi=250)
sns.boxplot(x='DEPARTURE_DELAY',
            y='AIRLINE',
            fliersize=2,
            data=df)
plt.xlabel('Departure Delay (mins)')
plt.ylabel('Airline Code')

# Box plots for arrival delays for each airline
plt.figure(figsize=(15, 10), dpi=250)
sns.boxplot(x='ARRIVAL_DELAY',
            y='AIRLINE',
            fliersize=2,
            data=df)
plt.xlabel('Arrival Delay (mins)')
plt.ylabel('Airline Code')

# Lets see the 5 number summary (min, Q1, median, Q3, max) of
# departure delay and arrival delay for each airline.

# Five number summary of departure delay for each airline
print(df.groupby('AIRLINE')['DEPARTURE_DELAY'].describe().sort_values('50%', ascending=False))

# Five number summary of arrival delay for each airline
print(df.groupby('AIRLINE')['ARRIVAL_DELAY'].describe().sort_values('50%', ascending=False))

# Top 10 airports with highest average departure delay
print(df.groupby('ORIGIN_AIRPORT')['DEPARTURE_DELAY'].mean().sort_values(ascending=False).head(10))

# Checking how many flights depart from the top 10 airports with highest
# average departure delay. It appears only a handful of flights depart from 
# these airports. Hence they have the highest average Departure Delay
a = df.groupby('ORIGIN_AIRPORT')['ORIGIN_AIRPORT'].count()
print('Flights departing from FAR:', a['FAR'])
print('Flights departing from 12898:', a['12898'])
print('Flights departing from BMI:', a['BMI'])
print('Flights departing from ERI:', a['ERI'])
print('Flights departing from MYR:', a['MYR'])
print('Flights departing from 14576:', a['14576'])
print('Flights departing from 14696:', a['14696'])
print('Flights departing from 10157:', a['10157'])
print('Flights departing from 12992:', a['12992'])
print('Flights departing from 12206:', a['12206'])

# Lets check if the departure delay and arrival has 
# anything to do with distance of trip?

# Scatter Plot between Departure Delay and Distance
plt.scatter(df['DISTANCE'], df['DEPARTURE_DELAY'], color='r', marker='.')
plt.ylabel("Departure Delay (mins)")
plt.xlabel("Distance")
plt.show()

# Scatter Plot between Arrival Delay and Distance
plt.scatter(df['DISTANCE'], df['ARRIVAL_DELAY'], color = 'g', marker = '.')
plt.ylabel("Arrival Delay (mins)")
plt.xlabel("Distance")
plt.show()

# Correlation coefficients between Departure/Arrival Delay and Distance
print(df[['DISTANCE', 'DEPARTURE_DELAY','ARRIVAL_DELAY']].corr())

# Lets check the relation between departure delay and 
# DAY_OF_WEEK
print(df.groupby('DAY_OF_WEEK')['DEPARTURE_DELAY'].describe())
# It seems Wednesday and Saturday have a lesser avg. departure delay and
# lesser 3rd quantile departure delay.

# If there is a departure delay (i.e. positive values for departure delay),
# does distance have anything to do with arrival delay ?  

# Scatter plot of Arrival Delay vs Distance when Departure Delay > 0
plt.scatter(df[df['DEPARTURE_DELAY'] > 0]['DISTANCE'],
            df[df['DEPARTURE_DELAY'] > 0]['ARRIVAL_DELAY'], color='g', marker='.')
plt.ylabel("Arrival Delay (mins)")
plt.xlabel("Distance")
plt.show()
# But scatter plot does not really give us any hint. Hence we check the pearson
# correlation coefficient.

# Correlation coefficients between Arrival Delay and Distance Departure Delay > 0
print(df[df['DEPARTURE_DELAY'] > 0][['ARRIVAL_DELAY', 'DISTANCE']].corr())

# Ten Busiest Airports in terms of departing flights
# and their departure delay five number statistics.
print(df.groupby('ORIGIN_AIRPORT')['DEPARTURE_DELAY'].describe().sort_values('count', ascending=False).head(10))

# Ten Busiest Airports in terms of arriving flights
# and their arrival delay five number statistics
print(df.groupby('DESTINATION_AIRPORT')['ARRIVAL_DELAY'].describe().sort_values('count', ascending=False).head(10))

# Which months have the highest average departure delay and arrival delay.
print(df.groupby('MONTH')['DEPARTURE_DELAY'].describe().sort_values('50%', ascending=False))
# It seems that July (and even June) have the highest departure delay (Average, Median, 75 quantile)

print(df.groupby('MONTH')['ARRIVAL_DELAY'].describe().sort_values('50%', ascending=False))
# July (and even June) have highest arrival delays as well (Average, Median, 75 quantile)

# REGRESSION ANALYSIS

# Removing all rows that have missing values in the WEATHER_DELAY column
df.dropna(subset=['WEATHER_DELAY'], inplace=True)
df.reset_index(drop=True, inplace=True)
# Check each column for missing values. Only column cancellation feature has null values.
print(df.isnull().any())

# 2
# Regression Model using statsmodels
X = df[['LATE_AIRCRAFT_DELAY', 'AIR_SYSTEM_DELAY', 'WEATHER_DELAY',
        'DAY_OF_WEEK', 'DEPARTURE_TIME', 'DEPARTURE_DELAY', 'DISTANCE', 'AIRLINE']]

# It would be better if we convert DAY_OF_WEEK into a column with categorical values
# According to the dataset 1 is Monday, 2 is Tuesday and so on...
X['DAY_OF_WEEK'].replace([1, 2, 3, 4, 5, 6, 7], ['Monday', 'Tuesday',
                         'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], inplace=True)

# Creating dummy variables for all categorical variables (DAY_OF_WEEK and AIRLINE)
# The reference level for DAY_OF_WEEK is Friday and the reference level for AIRLINE is Airline AA
X = pd.get_dummies(data=X, drop_first=True)

# Adding intercept column
X_int = sm.add_constant(X)

y = df['ARRIVAL_DELAY']

linreg = sm.OLS(y, X_int).fit()
print(linreg.summary())

# Print the Model
print("Our Multiple Linear Regression Model is:")
print("ARRIVAL_DELAY = {:.4f}".format(linreg.params[0]), end="")
for i in range(1, len(linreg.params)):
    if linreg.params[i] < 0:
        print(
            " - {:.4f} *{}".format(abs(linreg.params[i]), linreg.params.index[i]), end="")
    else:
        print(
            " + {:.4f} *{}".format(linreg.params[i], linreg.params.index[i]), end="")

# Model Diagnostics

# Residual vs fitted value plot: The plot shows that the residual variance increases as the fitted values increase (Heteroscadasticity).
# The lowess line (to visualize non-linear patterns in a scatter plot) is curved
# Also, there are a lot of outliers.

lowess = sm.nonparametric.lowess(linreg.resid, linreg.fittedvalues)
plt.scatter(linreg.fittedvalues, linreg.resid)
plt.plot(lowess[:, 0], lowess[:, 1], c='r')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.show()

# Normality Test
# QQ Plot: The graph looks somewhat linear. The center follows a straight line but both the ends deviate quite a lot (heavy tails)
# The data is not precisely normally distributed, but its not too far off.
fig = sm.qqplot(linreg.resid, line='s')

# At a significance level of 5%, the following variables are significant: LATE_AIRCRAFT_DELAY, AIR_SYSTEM_DELAY,
# WEATHER_DELAY, DEPARTURE_TIME, DEPARTURE_DELAY, AIRLINE_UA, AIRLINE_WN

# R-squared: 93.89% of the variability of our model was captured by the proposed linear model.
print(linreg.rsquared)

# Interpretation for a few coefficients
# Fixing everything else, for every 1 minute delay due to air systems, the arrival delay of a flight increases by 0.3532 minutes
# Fixing everything else, for every 1 minute delay in the departure of a flight, its arrival delay increases by 0.8416 minutes. This makes sense because if a flight departs late, it can only catch up on time mid-air to a certain extent and will ultimately arrive late.
# Fixing everything else, Airline UA on average has a 6.95 min lesser arrival delay than Airline AA (our reference airline)
# Fixing everything else, Airline WN on average has a 4.02 min lesser arrival delay than Airline AA (our reference airline)

# Removing outliers
Q1 = df['ARRIVAL_DELAY'].quantile(0.25)
Q3 = df['ARRIVAL_DELAY'].quantile(0.75)
IQR = Q3 - Q1

df.drop(df[(df['ARRIVAL_DELAY'] > (Q3 + 1.5*IQR)) |
        (df['ARRIVAL_DELAY'] < (Q1 - 1.5*IQR))].index, inplace=True)
df.reset_index(drop=True, inplace=True)

# Refining regression model: Changing response variable to log(ARRIVAL_DELAY)
# and removing predictors with p-values more than 0.05

X_refined = df[['LATE_AIRCRAFT_DELAY', 'AIR_SYSTEM_DELAY', 'WEATHER_DELAY',
                'DAY_OF_WEEK', 'DEPARTURE_TIME', 'DEPARTURE_DELAY', 'DISTANCE', 'AIRLINE']]

# Convert DAY_OF_WEEK into a column with categorical values
X_refined['DAY_OF_WEEK'].replace([1, 2, 3, 4, 5, 6, 7],
                                 ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], inplace=True)

# Creating dummy variables for all categorical variables (DAY_OF_WEEK and AIRLINE)
# The reference level for DAY_OF_WEEK is Friday and the reference level for AIRLINE is Airline AA
X_refined = pd.get_dummies(data=X_refined, drop_first=True)

# Adding intercept column
X_refined_int = sm.add_constant(X_refined)

# Removing the columns we don't want (Removing Insignificant Predictors)
X_refined_int.drop(['DISTANCE', 'DAY_OF_WEEK_Monday', 'DAY_OF_WEEK_Saturday', 'DAY_OF_WEEK_Sunday', 
                    'DAY_OF_WEEK_Thursday', 'DAY_OF_WEEK_Tuesday', 'DAY_OF_WEEK_Wednesday', 'AIRLINE_AS',
                    'AIRLINE_B6', 'AIRLINE_DL', 'AIRLINE_EV', 'AIRLINE_F9', 'AIRLINE_HA', 'AIRLINE_MQ',
                    'AIRLINE_NK', 'AIRLINE_OO', 'AIRLINE_US', 'AIRLINE_VX'], axis=1, inplace=True)

# Changing the response variable. Adding 1 before taking the log is not 
# necessary because data does not contain 0 or negative values
y_refined = np.log(df['ARRIVAL_DELAY'])

linreg2 = sm.OLS(y_refined, X_refined_int).fit()
print(linreg2.summary())

# Print the Model
print("Our Refined Multiple Linear Regression Model is:")
print("ARRIVAL_DELAY = {:.4f}".format(linreg2.params[0]), end="")
for i in range(1, len(linreg2.params)):
    if linreg2.params[i] < 0:
        print(
            " - {:.4f} *{}".format(abs(linreg2.params[i]), linreg2.params.index[i]), end="")
    else:
        print(
            " + {:.4f} *{}".format(linreg2.params[i], linreg2.params.index[i]), end="")

# Refined Model Diagnostics
# Residual vs fitted value plot: This graph still looks heteroskedastic.
# The curvature of the scatterplot / lowess line indicate there are still 
# some non-linearities that the model is not accounting for. 

lowess2 = sm.nonparametric.lowess(linreg2.resid, linreg2.fittedvalues)
plt.scatter(linreg2.fittedvalues, linreg2.resid)
plt.plot(lowess2[:, 0], lowess2[:, 1], c='r')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.show()

# Normality Test
# QQ Plot : The graph looks somewhat linear. The distribution of the data negatively skewed (left skewed)
fig = sm.qqplot(linreg2.resid, line='s')

# At a significance level of 5%, the following variables are significant: LATE_AIRCRAFT_DELAY, AIR_SYSTEM_DELAY,
# WEATHER_DELAY, DEPARTURE_TIME, DEPARTURE_DELAY, AIRLINE_UA

# R-squared: 78.92% of the variability of our model was captured by the proposed linear model. It has decreased since we used subset of the variables we used earlier.
print(linreg2.rsquared)

# Interpretation for a few coefficients
# Fixing everything else, for every 1 minute delay due to air systems, the arrival delay of a flight will be multiplied by exp(0.0131) = 1.0132 times
# Fixing everything else, for every 1 minute delay in the departure of a flight, its arrival delay will be multiplied by exp(0.0135) = 1.0136 times
# Fixing everything else, Airline UA average arrival delay will be multiplied by exp(-0.0941) = 0.91 times to that of Airline AA (our reference airline)

# Suggestions to improve the model further
# 1) Since the relationship is still non-linear, perhaps some higher degree terms can be added to the model.
# 2) Interaction terms can also be added such as interactions between the various kinds of delays. 
# 3) The summary output suggests a strong colinearity between the independent variables. A correlation matrix between them
# indicated that the pearson correlation coefficient between LATE_AIRCRAFT_DELAY and DEPARTURE_DELAY was 0.6 (which is between moderate and strong)
# Hence, to tackle this issue, we can consider removing one of them from the model or address the multicollinearity in other ways. 
# 4) Moreover, the departure time variable does not make much sense because the model treats it like an float whereas its actually a time.
# Ideally, the predicted arrival delay when time is 0000 and when it is 2359 should be close, but it will not be close in this model. Perhaps it can be removed as well. 

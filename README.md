# Flight-EDA-and-Regression
The objective of this project is to conduct an in-depth exploratory data analysis on a subset of the flight delay dataset, with the aim of investigating the potential influence of a set of variables on flight arrival delay times.

The entire dataset can be found here: [2015 Flight Delays and Cancellations](https://www.kaggle.com/datasets/usdot/flight-delays)

I am working with a randomized subset of 'flights.csv' data file to perform my analysis and building a regression model. Attaching the CSV file, the python file as well as my approach and findings in a easy to understand report :)

The flights.csv file contains the following features along with some data definitions & values  
YEAR (2015)  
MONTH (1 through 12)  
DAY (1 through 31)  
DAY_OF_WEEK (1 throught 7)  
AIRLINE  
FLIGHT_NUMBER  
TAIL_NUMBER  
ORIGIN_AIRPORT  
DESTINATION_AIRPORT  
SCHEDULED_DEPARTURE (Time - in XXYY format)  
DEPARTURE_TIME (WHEELS_OFF - TAXI_OUT)  
DEPARTURE_DELAY (DEPARTURE_TIME - SCHEDULED_DEPARTURE)  
TAXI_OUT (Time - The time duration elapsed between departure from the origin airport gate and wheels off)  
WHEELS_OFF (Time - The time point that the aircraft's wheels leave the ground)  
SCHEDULED_TIME (Time - in XXYY format)  
ELAPSED_TIME (AIR_TIME + TAXI_IN + TAXI_OUT)  
AIR_TIME (The time duration between WHEELS_OFF and WHEELS_ON time)  
DISTANCE  
WHEELS_ON (Time - The time point that the aircraft's wheels touch on the ground)  
TAXI_IN (Time - The time duration elapsed between wheels-on and gate arrival at the destination airport)  
SCHEDULED_ARRIVAL (Time - in XXYY format)  
ARRIVAL_TIME (Time - in XXYY format) (WHEELS_ON + TAXI_IN)  
ARRIVAL_DELAY (ARRIVAL_TIME - SCHEDULED_TIME)  
DIVERTED (Yes/No - Boolean)  
CANCELLED (Yes/No - Boolean)  
CANCELLATION_REASON (Reason for Cancellation of flight: A - Airline/Carrier; B - Weather; C - National Air System; D - Security)  
AIR_SYSTEM_DELAY  
SECURITY_DELAY  
AIRLINE_DELAY  
LATE_AIRCRAFT_DELAY  
WEATHER_DELAY  

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
teams = pd.read_csv("teams.csv")
teams = teams [["team", "country", "year", "athletes", "age", "prev_medals", "medals"]]
#how to print the teams dataframe that pd converted it into 
#print(teams.to_string()) 
#how to check if the correlation between all the columns in teams with medals (see if there is a correlation with medals and athletes and if its worth exploring)
#print(teams.corr() ["medals"])
#how to plot a linear regression and display the graph using plt.show()
#sns.lmplot(x="athletes", y="medals", data = teams, fit_reg=True, ci=None)
#plt.show()
#plotting a histogram
#teams.plot.hist(y="medals")
#plt.show()
#find the rows with null values -- cant use to train model
teams[teams.isnull().any(axis=1)]
#drop the ones with null values from teams
teams = teams.dropna()
#print(teams.to_string )

#create two new arrays, one to train the model on and one with different data to test it on
train = teams[teams["year"] < 2012].copy()
test = teams[teams["year"] >= 2012].copy()

#print the rows and columns of the dataset 
#print(train.shape)
#print(test.shape)

#initializing linear regression class
reg = LinearRegression()
predictors = ["athletes", "prev_medals"]
target = "medals"

#use the fit method to actually fit the model 
reg.fit(train[predictors], train["medals"])
LinearRegression()

#test is by using the predict method, notice we use the 
#test array and we only pass in predictors, we don't 
#want to give it the target because it should come up w the answer itself
predictions = reg.predict(test[predictors])
#print it, here we notice it is not whole numbers and some are negative, 
#you cant have part of a medal or negative medals
#print(predictions)

#add a new column for the predictions in the table
test["predictions"] = predictions
#for any location in the predictions column that is <0, make it 0
test.loc[test["predictions"]< 0 , "predictions"] = 0

#round all the predictions to the nearest whole number
test["predictions"] = test["predictions"].round()
#print(test.to_string )

#get the mean absolute error between the actual medals won and the ML's prediction
error = mean_absolute_error(test["medals"], test["predictions"])
#the number it prints means we were on average within that many medals that the team actually won
#print(error)

#give me some data on the medals column, we can see our
#error is way less than the standard deviation it prints which means 
#we have good data 
#print(teams.describe()["medals"])

#find how much off our predictions were for each team
errors = (test["medals"] - test["predictions"]).abs()

#group it by country and display it
error_by_team = errors.groupby(test["team"]).mean()
#print(error_by_team)

#find average medals for each team
medals_by_team = test["medals"].groupby(test["team"]).mean()
error_ratio = error_by_team / medals_by_team
#print(error_ratio)

#get rid of the null values that result from divide by 0
error_ratio = error_ratio[~pd.isnull(error_ratio)]
#print(error_ratio)

#get rid of infinite values that result from 0/#
error_ratio = error_ratio[np.isfinite(error_ratio)]
print(error_ratio)


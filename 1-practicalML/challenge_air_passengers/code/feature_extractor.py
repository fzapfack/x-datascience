from __future__ import division
 
import pandas as pd
import os
from sklearn.preprocessing import scale, PolynomialFeatures,normalize
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.learning_curve import learning_curve
import math
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as ts
 
class FeatureExtractor(object):
 
    def __init__(self):
        self.x_columns=None
        self.useExtDataWeather=0
        self.useExtDistance=1
        self.useExtSeason=1
        self.usedateWeeklyStats=0
        self.userouteStatsDF=1
        self.useExtEnplanements=0
        self.useStd=0
        self.routeStatsDF=None
        self.dateWeeklyStatsDF=None
        path = os.path.dirname(__file__)  # use this in submission
        airport_info = pd.read_csv(os.path.join(path, "external_data.csv"))
        self.airport_info = airport_info.set_index(['IATA_FAA'])
 
 
    def fit(self, X_df, y_array):
        if(self.usedateWeeklyStats):
            self.dateWeeklyStatsDF=X_df.copy()
            self.dateWeeklyStatsDF["pax"]=y_array
            self.dateWeeklyStatsDF['DateOfDeparture'] = pd.to_datetime(self.dateWeeklyStatsDF['DateOfDeparture'])
            self.dateWeeklyStatsDF['week'] = self.dateWeeklyStatsDF['DateOfDeparture'].dt.week
            self.dateWeeklyStatsDF=self.dateWeeklyStatsDF.groupby('week').median()
            #print self.dateWeeklyStatsDF
        if(self.userouteStatsDF):
            self.routeStatsDF=X_df.copy()
 
            self.routeStatsDF["AirRoute"]=self.routeStatsDF.apply(self.createRoute, axis=1)
            self.routeStatsDF["pax"]=y_array
            self.routeStatsDF=self.routeStatsDF.drop('DateOfDeparture', axis=1)
 
            self.routeStatsDF=self.routeStatsDF.drop('Departure', axis=1)
            self.routeStatsDF=self.routeStatsDF.drop('Arrival', axis=1)
            self.routeStatsDF=self.routeStatsDF.drop('std_wtd', axis=1)
            self.routeStatsDF=self.routeStatsDF.groupby('AirRoute').median().rename(columns={'pax': 'mean'}).rename(columns={'WeeksToDeparture': 'avgWeeksToDepart'})
            #print  self.routeStatsDF
        #Use a .copy for actual training
        plotVisuals=0
        if(plotVisuals==1):
            X_df["log_PAX"]=y_array
            print X_df["WeeksToDeparture"].mean()
            print len(np.unique(y_array))
            X_df["AirRoute"]=X_df.apply(self.createRoute, axis=1)
            X_df["Season"] = X_df['DateOfDeparture'].map(self.calculateSeason)
 
            X_df['DateOfDeparture'] = pd.to_datetime(X_df['DateOfDeparture'])
            X_df['year'] = X_df['DateOfDeparture'].dt.year
            X_df['month'] = X_df['DateOfDeparture'].dt.month
            X_df['day'] = X_df['DateOfDeparture'].dt.day
            X_df['weekday'] = X_df['DateOfDeparture'].dt.weekday
            X_df['week'] = X_df['DateOfDeparture'].dt.week
            X_df['n_days'] = X_df['DateOfDeparture'].apply(lambda date: (date - pd.to_datetime("1970-01-01")).days)
 
            if(0):
                X_df = X_df.drop('DateOfDeparture', axis=1)
                X_df = X_df.join(pd.get_dummies(X_df['Departure'], prefix='d'))
                X_df = X_df.join(pd.get_dummies(X_df['Arrival'], prefix='a'))
                X_df = X_df.drop('Departure', axis=1)
                X_df = X_df.drop('Arrival', axis=1)
                X_df = X_df.drop('AirRoute', axis=1)
 
                gb = GradientBoostingRegressor(alpha=0.9, init=None,max_depth=3, learning_rate=0.2, loss='ls'
                                    ,max_features=None,min_samples_leaf=1, min_samples_split=2,min_weight_fraction_leaf=0.0
                                    ,n_estimators=100,presort='auto', random_state=None, subsample=1.0, verbose=0,warm_start=False)
                train_sizes, train_scores, valid_scores = learning_curve(gb, X_df.values, y_array, train_sizes=[50, 100, 1000, 7418], cv=10)
                plt.figure()
                plt.title("title")
 
                plt.xlabel("Training examples")
                plt.ylabel("Score")
                train_scores_mean = np.mean(train_scores, axis=1)
                train_scores_std = np.std(train_scores, axis=1)
                test_scores_mean = np.mean(valid_scores, axis=1)
                test_scores_std = np.std(valid_scores, axis=1)
                plt.grid()
 
                plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                                 train_scores_mean + train_scores_std, alpha=0.1,
                                 color="r")
                plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
                plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                         label="Training score")
                plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                         label="Cross-validation score")
 
                plt.legend(loc="best")
                plt.show()
 
            X_df["WeeksToDeparture"]=X_df["WeeksToDeparture"]/X_df["WeeksToDeparture"].max()-X_df["WeeksToDeparture"].mean()/X_df["WeeksToDeparture"].max()
            X_df["std_wtd"]=X_df["std_wtd"]/X_df["std_wtd"].max()-X_df["std_wtd"].mean()/X_df["std_wtd"].max()
            X_df["log_PAX"]=X_df["log_PAX"]/X_df["log_PAX"].max()-X_df["log_PAX"].mean()/X_df["log_PAX"].max()
            counter=0
            indx=X_df["week"].unique()
 
            X_df.boxplot(column='log_PAX', by="weekday", ax=None, fontsize=None, rot=0, grid=True, figsize=None, layout=None)
            #print frame.head(5)
            #g = sns.FacetGrid(X_df, col="week", sharex=False)
            #g.map(sns.boxplot, 'log_PAX')
            raw_input("Press Enter to continue...")
            for i in indx:
 
                frame=X_df.loc[X_df['week'] == i].sort('week')
                lag=0
                result = ts.adfuller(frame['log_PAX'],lag)
                print result[0]
                print result[4]['10%']
                if(result[0]<result[4]['5%']):
                    counter=counter+1
                #frame.plot(x='n_days',y=['log_PAX','WeeksToDeparture'])
 
            print counter
            print "end test"
    def transform(self, X_df):
        X_array=self.initFeatureSet(X_df)
        return X_array
 
 
    def initFeatureSet(self,X_df):
        X_encoded = X_df
 
        X_encoded["AirRoute"]=X_encoded.apply(self.createRoute, axis=1)
 
        X_encoded.join(pd.get_dummies(X_encoded['AirRoute'], prefix='r'))
        X_encoded["RouteAvg"], X_encoded["RouteAvgDepart"],X_encoded["Distance"] = zip(*X_encoded['AirRoute'].map(self.assignStats))
        X_encoded["Season"] = X_encoded['DateOfDeparture'].map(self.calculateSeason)
        #X_encoded["AprRouteAvg"] = X_encoded.groupby('AirRoute')
        X_encoded=X_encoded.drop('AirRoute', axis=1)
 
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Departure'], prefix='d'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Arrival'], prefix='a'))
        X_encoded['DateOfDeparture'] = pd.to_datetime(X_encoded['DateOfDeparture'])
        X_encoded['year'] = X_encoded['DateOfDeparture'].dt.year
        X_encoded['month'] = X_encoded['DateOfDeparture'].dt.month
        X_encoded['day'] = X_encoded['DateOfDeparture'].dt.day
        X_encoded['weekday'] = X_encoded['DateOfDeparture'].dt.weekday
        X_encoded['week'] = X_encoded['DateOfDeparture'].dt.week
        X_encoded['n_days'] = X_encoded['DateOfDeparture'].apply(lambda date: (date - pd.to_datetime("1970-01-01")).days)
        if(self.usedateWeeklyStats):
            X_encoded['weekAvg'] = X_encoded['week'].apply(lambda row: self.dateWeeklyStatsDF.loc[row]["pax"])
        #X_encoded = X_encoded.join(pd.get_dummies(X_encoded['year'], prefix='y'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['month'], prefix='m'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['day'], prefix='d'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['weekday'], prefix='wd'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['week'], prefix='w'))
        #X_encoded['currentAvgWeeks2Dep'] = X_encoded.ap
        X_encoded = X_encoded.drop('year', axis=1)
        X_encoded = X_encoded.drop('month', axis=1)
        X_encoded = X_encoded.drop('day', axis=1)
        X_encoded = X_encoded.drop('weekday', axis=1)
        X_encoded = X_encoded.drop('std_wtd', axis=1)
        X_encoded = X_encoded.drop('DateOfDeparture', axis=1)
        X_encoded = X_encoded.drop('Departure', axis=1)
        X_encoded = X_encoded.drop('Arrival', axis=1)
 
 
        self.x_columns=X_encoded.columns
        X_array = X_encoded.values
        #X_array = normalize(X_array, axis=1)
 
        return X_array
 
 
 
    def calculateSeason(self,date):
        season=0
        date=date.split("-")
        #SEASON
        #Highest: Mid December into the first week of January
        highest_start = [15,12]
        highest_end = [7,1]
        #High: Early January to mid April & mid June to mid August
        high_start_1 = [8,1]
        high_end_1 = [15,3]
        high_start_2 = [15,6]
        high_end_2 = [15,9]
        #Lowest: Mid April to mid June & mid August to mid December
        month=int(date[1])
        day=int(date[2])
        if (month==1 and day<8) or (month==12 and day>14):
                season=2
        elif (month==1 and day>7) or (month==3 and day<14)  or month==2 or month==7 or month==8 or (month==6 and day>14) or (month==9 and day<16):
                season=1
        return season
 
    def createRoute(self,row):
         airport2=str(row["Arrival"])
         airport1=str(row["Departure"])
         route=airport1+"_"+airport2
         return route
 
    def createDistance(self,row):
        airport1 = row["Arrival"]
        airport2 = row["Departure"]
        try:
            avg = self.routeStatsDF.loc[str(airport2) + "_" + str(airport1)]["mean"]
            avg2 = self.routeStatsDF.loc[str(airport2) + "_" + str(airport1)]["avgWeeksToDepart"]
        except:
            avg = self.routeStatsDF["mean"].mean()
            avg2 = self.routeStatsDF["avgWeeksToDepart"].mean()
 
    def assignStats(self,row):
        airport1 = row.split("_")[1]
        airport2 = row.split("_")[0]
        try:
            avg = self.routeStatsDF.loc[str(airport2) + "_" + str(airport1)]["mean"]
            avg2 = self.routeStatsDF.loc[str(airport2) + "_" + str(airport1)]["avgWeeksToDepart"]
        except:
            avg = self.routeStatsDF["mean"].mean()
            avg2 = self.routeStatsDF["avgWeeksToDepart"].mean()
        airport1_loc_info=(self.airport_info.loc[airport1]["Latitude"],self.airport_info.loc[airport1]["Longitude"])
        airport2_loc_info=(self.airport_info.loc[airport2]["Latitude"],self.airport_info.loc[airport2]["Longitude"])
        x1 = math.radians(airport1_loc_info[0])
        y1 = math.radians(airport1_loc_info[1])
        x2 = math.radians(airport2_loc_info[0])
        y2 = math.radians(airport2_loc_info[1])
        # Great circle distance in radians
        angle1 = math.acos(math.sin(x1) * math.sin(x2)  + math.cos(x1) * math.cos(x2) * math.cos(y1 - y2))
        # Convert back to degrees.
        angle1 = math.degrees(angle1)
        # Each degree on a great circle of Earth is 60 nautical miles.
        dist = 60.0 * angle1
        return  avg,avg2,dist

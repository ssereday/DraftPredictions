#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 09:10:49 2023
TODO: 

@author: t7610
"""

import pandas as pd
import numpy as np
from metaflow import FlowSpec, step, IncludeFile, Parameter
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature


# TODO:
#    1 ADD QUICK NOTES THROUGH WHOLE DOCUMENT
#    2 SAVE TO GET
#    3 INVESTIGATE NEXT STEPS

# add notes, type, etc. look for tips to improve coding
# add job_request.yaml, put fields unlikely to change at bottom
#add imputation flag
#regression
# metaflow, more modular
# APM, more leagues, more features, better features, better algorithms, more applications
# CBB, RGM
# MPG/SRS Prior, Peak HistAPM+WS48, Peak BPM, finally ResidPM
# convert OAPM, DAPM to %TmPts scale

#scrape data: https://github.com/roclark/sportsipy/blob/master/docs/ncaab.rst?plain=1 buggy
#explain time to complete, next steps, reasons for choosing what I chose.

#Preparation: 1 hour
#Viability phase: 1 hour
#Iteration setup: 3 hours

# Raw Data Source for features
nba_college_link = "http://www.nbastats.net/07Colleges/01Colleges-Statistics.xlsx"
# Raw Data Source for target
nba_career_link = "http://www.nbastats.net/01NbaAbaNblAbl/07-Players/03-PlayerCareerStats.xlsx"
# location to save
datafile = '../data/nbastatsnet_draftpred.csv'
# suffix to rename and differentiate college features
colsuffix = '_college'
# Used to combine data tables
# TODO: Find more robust idfields as data expands
idfields = ['FirstName','LastName']
# found fields by manual inspection, exclude .1's
aggcols = ['G','GS']
pgcols = ['MIN', 'FGM', 'FGA', '3FGM', '3FGA', 'FTM', 'FTA', 'OFF',
       'DEF', 'REB', 'AST', 'STL', 'BLK', 'TO', 'PF', 'PTS']
# rename fields in imported data, found by manual inspection
colname_map = {'Unnamed: 2':'LastName','Unnamed: 1':'LastName',"Name":"FirstName","Team(s)":"Team"}
# position dictionary to convert string to a feature
posdict = {'C':5,'C-F':4.6,'C-G':3.7,'F':3.5,'F-C':4,'F-G':2.7,'G':1.5,'G-F':2.3}
# location of intermediate files
collegefile = '../data/collegestats.csv'
nbacareerfile = '../data/nbacareerstats.csv'

# dictionary to save imported data
urlfiledict = {
        nba_college_link: collegefile,
        nba_career_link: nbacareerfile}


def season_int(dfcol):
    """
    converts string year to numeric
    :param dfcol: column name to be converted
    :returns: converted column
    """    
    return np.where(dfcol.str[0]=='(',dfcol.str[1:5],-1).astype(int)+1


def import_file(link, colname_map, outfile, skiprows=4):
    """
    
    :param link: 
    :param colname_map: 
    :param outfile: 
    :param skiprows: 
    :returns:
    """
    df = pd.read_excel(link ,skiprows=skiprows)
    df = df[[x for x in df.columns if '%' not in x]]
    df = df.rename(columns=colname_map)
    df = df[~df['LastName'].isnull()]
    df.to_csv(outfile)
    return df

def clean_yr(df, SeasonFields):
    """
    
    :param df: 
    :param SeasonFields: 
    :returns:
    """
    for s in SeasonFields:
        df[s] = season_int(df[s])
    return df

def pg_fields(df, aggcols, pgcols, suffix=''):
    """
    
    :param df: 
    :param aggcols: 
    :param pgcols: 
    :param suffix: 
    :returns:
    """
    statcols = aggcols + pgcols
    statcols = statcols + [x+'.1' for x in statcols]
    df.columns = [(x.replace('.1','PG')+suffix) if x in statcols else x for x in df.columns]
    return df

def choose_columns(df, idfields, aggcols, pgcols, suffix=''):
    """
    
    :param df: 
    :returns:
    """
    return df[idfields+[x+suffix for x in aggcols]+[x+'PG'+suffix for x in pgcols]]




def save_data_locally(urlfiledict, colname_map):
    for key, value in urlfiledict.items():
        import_file(key, colname_map, value, skiprows=4)
    return


class SimpleNBADraftPrediction(FlowSpec):
    """
    A flow to generate some statistics about the movie genres.

    The flow performs the following steps:
    1) Ingests a CSV into a dataframe.
    2) Fan-out over genre using Metaflow foreach.
    3) Compute quartiles for each genre.
    4) Save a dictionary of genre-specific statistics.

    """
    refresh = Parameter(
        "refresh", help="Should you reimport the files?", default=True
    )

    @step
    def start(self):
        """
        The start step:
        1) Loads the movie metadata into dataframe.
        2) Finds all the unique genres.
        3) Launches parallel statistics computation for each genre.

        """
#        collegefile = '/TB10/Basketball/Git/draft_predictions/'
        #### REFRESH NOT WORKING
        if self.refresh:
            self.cdf = import_file(nba_college_link,colname_map, collegefile, skiprows=4)
            self.ndf = import_file(nba_career_link,colname_map, nbacareerfile, skiprows=4)
            self.ndf = pg_fields(self.ndf, aggcols, pgcols, suffix='')
            self.ndf = self.ndf[self.ndf['League']=='NBA']
        else:
            self.ndf = pd.read_csv(collegefile)
            self.cdf = pd.read_csv(nbacareerfile)
        print(self.ndf.shape)
        self.next(self.college_data)

    @step
    def college_data(self):
        """
        
        """

        self.cdf = clean_yr(self.cdf, SeasonFields=['Season'])
        indcols = ['CC','FR','DNP']
        for i in indcols:
            self.cdf[i] = np.where(self.cdf[i]==i,1,0)
        self.cdf['Notes'] = self.cdf['Notes'].astype(str)
        self.cdf = pg_fields(self.cdf, aggcols, pgcols, suffix=colsuffix)
        self.plyryrs = self.cdf[self.cdf['Season']>0][['FirstName','LastName','Season','Team','Notes','CC', 'FR', 'DNP']].groupby(['FirstName','LastName']).agg(yrs_college=('Season', 'count'), first_yr_college=('Season', 'min'), last_yr_college=('Season', 'max'), last_college=('Team','last'), notes_college=('Notes','max'),CC_college=('CC','sum'),FR_college=('FR','sum'), DNP_college=('DNP','sum')).reset_index()
        self.college_career = self.cdf[self.cdf['Season']==0]
        self.college_career = choose_columns(self.college_career, idfields, aggcols, pgcols, suffix=colsuffix)
        self.college_career = pd.merge(self.college_career, self.plyryrs, on=['FirstName','LastName'])
        per36cols = [x for x in self.college_career.columns if ('PG' in x and x[:3]!='MIN')]
        self.college_career = self.college_career.replace('---','0')
        for x in per36cols:
            self.college_career[x.replace('PG','36')] = self.college_career[x].astype(float)/self.college_career['MINPG'+x.split('PG')[-1]].astype(float)*36
        self.ndf = pd.merge(self.ndf, self.college_career, on=idfields)
        self.ndf['yrs_since_college'] = self.ndf['From'] = self.ndf['last_yr_college']
        self.ndf = self.ndf.replace({'---': '0'}, regex=True)
        self.next(self.model)

    @step
    def model(self):
        """
        

        """

        self.target = 'MINPG'
        self.nullval = -99
        self.othfeats = ['last_yr_college','yrs_since_college','G_college','MINPG_college','yrs_college','CC_college','FR_college', 'DNP_college']
        self.col_features = self.othfeats + [x for x in self.ndf.columns if ('college' in x and '36' in x)]
        self.ndf = self.ndf[~self.ndf[self.target].isnull()]
        self.ndf['rand'] = np.random.rand(self.ndf.shape[0])
        self.ndf['train_ind'] = np.where(self.ndf['rand']<.6,"train",np.where(self.ndf['rand']>0.8,"test","valid"))
        self.train = self.ndf[self.ndf['train_ind']=='train']
        self.validate = self.ndf[self.ndf['train_ind']=='valid']
        self.test = self.ndf[self.ndf['train_ind']=='test']
        from sklearn.ensemble import RandomForestRegressor
        self.regr = RandomForestRegressor(max_depth=2, random_state=0)
        from sklearn.metrics import mean_squared_error
        self.mds = range(5,15)
        self.train.fillna(self.nullval).to_csv("../data/tmp.csv")
        for md in self.mds:
            regr = RandomForestRegressor(max_depth=md, random_state=0)
            regr.fit(self.train[self.col_features].fillna(self.nullval), self.train['MINPG'])
            self.test['predMPG'] = regr.predict(self.test[self.col_features].fillna(self.nullval))
            print(md)
            rmse=mean_squared_error(self.test['predMPG'], self.test['MINPG'])
            if md==self.mds[0]:
                self.best_md = md
                self.best_rmse = rmse
            else:
                if rmse<self.best_rmse:
                    self.best_md=md
                    self.best_rmse = rmse
            print(rmse)
        
        self.regr = RandomForestRegressor(max_depth=self.best_md, random_state=0)
        self.regr.fit(self.train[self.col_features].fillna(self.nullval), self.train['MINPG'])
        self.validate['predMPG'] = self.regr.predict(self.validate[self.col_features].fillna(self.nullval))
        self.ndf['predMPG'] = self.regr.predict(self.ndf[self.col_features].fillna(self.nullval))
        self.test['predMPG'] = self.regr.predict(self.test[self.col_features].fillna(self.nullval))
        self.train['predMPG'] = self.regr.predict(self.train[self.col_features].fillna(self.nullval))
        self.full = pd.concat([self.test,self.train,self.validate])
        self.full.to_csv("../data/nbastatsnet_draftpred.csv")
        self.next(self.validations)

    @step
    def validations(self):
        """
        
    
        """
        from sklearn.metrics import mean_squared_error
        # 
        #check for errors
#        self.error_records = self.ndf[self.ndf['yrs_since_college']][['FirstName','LastName']].drop_duplicates()
#        print(self.error_records)
        #eda
        print(self.ndf.describe())
        
        self.rmse=mean_squared_error(self.validate['predMPG'], self.validate['MINPG'])
        print(self.rmse)
        #Correlation
        print(np.corrcoef(x=self.validate['predMPG'].astype(float), y=self.validate['MINPG'].astype(float))[0][1])
        self.vimp_df = pd.DataFrame([self.col_features,self.regr.feature_importances_]).transpose()
        self.vimp_df.columns = ['column','importance']
        print(self.vimp_df.sort_values('importance',ascending=False))
        print(self.ndf[['FirstName','LastName','MINPG','predMPG','train_ind']].sort_values('predMPG',ascending=False).head(30))
        print(self.ndf[['FirstName','LastName','MINPG','predMPG','train_ind']].sort_values('predMPG',ascending=False).tail(30))
        print(self.validate[['FirstName','LastName','MINPG','predMPG']].sort_values('predMPG',ascending=False).head(30))
        print(self.validate[['FirstName','LastName','MINPG','predMPG']].sort_values('predMPG',ascending=False).tail(30))
        self.next(self.end)
    
    @step
    def end(self):
        """
        End the flow.

        """
        pass


if __name__ == "__main__":
    SimpleNBADraftPrediction()




#https://mlflow.org/docs/latest/tutorials-and-examples/tutorial.html
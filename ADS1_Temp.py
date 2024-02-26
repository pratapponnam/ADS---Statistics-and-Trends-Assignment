# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 22:41:56 2024

@author: 
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def process_data_pb(df_ESG,df_Temp):
    """Function to clean a data frame with required columns """
    

    #dropping unecessary data
    df_ESG = df_ESG.drop(['Country Name','Country Code','Time Code'
                          ,'Unnamed: 8','Unnamed: 9','Unnamed: 10'],axis = 1)

    #changing the column names to have relevant info
    df_ESG = df_ESG.rename(columns=
        {"CO2 emissions (metric tons per capita) [EN.ATM.CO2E.PC]":
                                    "CO2 emissions",
         "Forest area (% of land area) [AG.LND.FRST.ZS]": 
                                    "Forest Area",
         "Population density (people per sq. km of land area) [EN.POP.DNST]":
                                    "Population density",
         "Renewable energy consumption (% of total final energy consumption) [EG.FEC.RNEW.ZS]"
                                   :"Renewable Energy"})

    df_Temp = df_Temp.rename(columns={"Glob":"Global Temp",
                                      "NHem":"Northern Temp",
                                      "SHem":"Southern Temp"})

    #joining both dataframes 

    df = df_ESG.join([df_Temp['Global Temp'],
                      df_Temp['Northern Temp'],
                      df_Temp['Southern Temp']])

    # Filling all blanks with 0's to change the type of year from float to int.
    df.index = df.index.fillna(0)
    df.index = df.index.astype(int)
    return df


def plot_temp_line_pb(df):
    """
    Defining a function to create a Line plot 
    to identify the relation between Forest Area and Population Density
    """
    plt.figure(figsize=(7, 5))
    #plotting the global temp change data
    plt.plot( df['Population density'],df['Forest Area'], color='blue',marker ='o')
    
    #set the titles, labels and grid
    plt.title('Relation between Forest Area & Population Density')
    plt.xlabel('Population Density')
    plt.ylabel('Forest Area')
    plt.grid(True)
    plt.xlim(50,62)
    plt.xticks(rotation=45)
    # Save the plot
    plt.savefig('Linegraph.png')
    # Show the plot
    plt.show()
    return


def plot_temp_histogram_pb(*df):
    """
    Defining a function to create a histogram 
    to understand the frequency of temperature 
    anomalies for different zones across the years
    """
    # Array for displaying the labels
    x = ['Global','Northern Hemisphere','Southern Hemisphere']
    
     
    plt.figure(figsize=(7, 5))
    
    # plotting an overlapped histogram to observe the frequency of the temp.
    for i, df in enumerate(df):
        sns.histplot(df, kde=True, stat="density",bins=10, 
                     linewidth=0, label=x[i],alpha=0.5)
    
    #set the titles, legend, labels and grid 
    plt.title('Distribution of Temperature Anomalies')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Frequency')
    plt.xlim(0.4,1.4)
    plt.grid(axis='y')
    plt.legend()
    # Save the plot
    plt.savefig('histogram.png')
    # Show the plot
    plt.show()
    return

def plot_temp_correlation_pb(df):
    """
    Defining a function to create a Heatmap to plot
    correlation between different factors 
    """
    # creating an upper triangular data with 0's and 1's for masking
    mask = np.triu(np.ones_like(df.corr()))
    plt.figure(figsize=(7, 5))
    # plotting a heatmap
    sns.heatmap(df.corr(), annot=True,mask=mask,cmap='coolwarm', linewidths=.5)
    #set the title
    plt.title('Correlation between various factors')
    # Save the plot
    plt.savefig('Heatmap.png')
    # Show the plot
    plt.show()
    return


#dataframes to store the data from csv files 
df_Temp= pd.read_csv('C:/Users/anude/Downloads/Untitled Folder/zonann_temps.csv',
                     index_col = 'Year')
df_ESG = pd.read_csv('C:/Users/anude/Downloads/Untitled Folder/ESGData.csv',
                     index_col = 'Time')

#to clean the data
df = process_data_pb(df_ESG,df_Temp)

#Using describe function for mean, stanadrd deviation, min and max value.
print('Stats of the data', end='\n')
df.describe()

#basic statistics

print('Skewness of the data', end='\n')
print(df.skew() , end='\n\n')

print('Kurtosis of the data', end='\n')
print(df.kurtosis() , end='\n\n')

print('Correlation of the data', end='\n')
print(df.corr() , end='\n\n')

#Visualising the Line Graph
plot_temp_line_pb(df)

#Visualising the Histogram
plot_temp_histogram_pb(df['Global Temp'],
                       df['Northern Temp'],
                       df['Southern Temp'])

#Visualising the Heatmap 
plot_temp_correlation_pb(df)
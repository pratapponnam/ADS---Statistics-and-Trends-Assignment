# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 22:41:56 2024

@author: Pratap Ponnam
"""
#importing required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import requests


def download_data(url):
    """
    Function to downlaod the data from the url provided and 
    returns the data frame which has the downloaded data 

    """
    while(1):
        try:
        #fetch the data from url 
            response = requests.get(url)
            #check if we got a response from the url
            if response.status_code == 200:
            # Save the content of the response to a local CSV file
                with open("downloaded_data.csv", "wb") as f:
                    f.write(response.content)
                break
            else:
                print("Failed to download CSV file. Status code:", response.status_code)
        #if exception is raised,continuing the loop
        except requests.exceptions.HTTPError :
            continue
        except requests.exceptions.ConnectionError :
            continue
        except requests.exceptions.Timeout :
            continue
        except requests.exceptions.RequestException :
            continue
    #moving data to dataframe from the downladed data
    df = pd.read_csv("downloaded_data.csv")
    return df


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


def plot_line_graph(df):
    """
    Defining a function to create a Line plot 
    to identify the relation between Forest Area and Population Density
    """
    plt.figure(figsize=(7, 5))
    #plotting the global temp change data
    plt.plot( df['Population density'],df['Forest Area'], color='blue',marker ='o')
    
    #set the titles, labels, limits and grid values
    plt.title('Relation between Forest Area & Population Density')
    plt.xlabel('Population Density')
    plt.ylabel('Forest Area')
    plt.grid(True)
    plt.xlim(50,62)
    plt.xticks(rotation=45)
    # Save the plot as Linegraph.png
    plt.savefig('Linegraph.png')
    # Show the plot
    plt.show()
    return


def plot_temp_histogram(*df):
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
    # Save the plot as histogram.png
    plt.savefig('histogram.png')
    # Show the plot
    plt.show()
    return


def plot_heatmap_correlation(df):
    """
    Defining a function to create a Heatmap to plot
    correlation between different factors 
    """
    # creating an upper triangular data with 0's and 1's for masking
    mask = np.triu(np.ones_like(df.corr()))
    plt.figure(figsize=(7, 5))
    # plotting a heatmap
    sns.heatmap(df.corr(), annot=True, mask=mask,
                cmap='coolwarm', linewidths=.5)
    
    #set the title
    plt.title('Correlation between various factors')
    # Save the plot as Heatmap.png
    plt.savefig('Heatmap.png')
    # Show the plot
    plt.show()
    return


#storing the filelinks in variables
url1 = 'https://github.com/pratapponnam/ADS---Statistics-and-Trends-Assignment/blob/main/zonann_temps.csv?raw=True'
url2 = 'https://github.com/pratapponnam/ADS---Statistics-and-Trends-Assignment/blob/main/ESGData.csv?raw=True'

#dataframes to store the data from urls using the functions
df_Temp = download_data(url1)
df_ESG = download_data(url2)

#set the index to dataframes

df_Temp.set_index('Year', inplace = True)
df_ESG.set_index('Time', inplace = True)

#calling function to clean the data
df = process_data_pb(df_ESG,df_Temp)

#Using describe function for mean, stanadrd deviation, min and max value.
print('Stats of the data', end='\n')
df.describe()

#basic statistics of the data

print('Skewness of the data', end='\n')
print(df.skew() , end='\n\n')

print('Kurtosis of the data', end='\n')
print(df.kurtosis() , end='\n\n')

print('Correlation of the data', end='\n')
print(df.corr() , end='\n\n')

#Visualising the Line Graph
plot_line_graph(df)

#Visualising the Histogram
plot_temp_histogram(df['Global Temp'],
                       df['Northern Temp'],
                       df['Southern Temp'])

#Visualising the Heatmap 
plot_heatmap_correlation(df)
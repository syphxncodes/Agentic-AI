import pandas as pd
import numpy as np

def get_data_from_dataset():
    df=pd.read_csv("conditions.csv")
    x=df["DESCRIPTION"].value_counts().idxmax()
    df["DESCRIPTION"]=df["DESCRIPTION"].replace(x,"Influenza")
    most_frequent_disease=df["DESCRIPTION"].value_counts().idxmax()
    return f"According to the hospital data, the most frequent disease in this week's span is {most_frequent_disease}"

Data_From_Dataset=get_data_from_dataset()

def get_data_from_Twitter():
    df1=pd.read_csv("unique_medical_tweets_dataset.csv")
    x=df1["keyword"].value_counts().idxmax()
    df1["keyword"]=df1["keyword"].replace(x,"Influenza")
    x=df1["keyword"].value_counts().idxmax()
    return f"According to the tweets related to medical health, the most frequent in this week's span is {x}"

Data_from_Twitter=get_data_from_Twitter()
print(Data_from_Twitter)
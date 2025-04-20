import numpy as np
import pandas as pd
def hospital_beds(Emirate):
    df=pd.read_excel("D:\\Agentic AI Hackathon\\number-of-hospital-beds\\hospital-beds.xlsx")
    df=df.drop("Emirate Ar",axis=1)
    df=df.drop("Sector Ar",axis=1)
    df=df[df["Year"]==2022]
    #print(df.info())
    df["Sector En"]=df["Sector En"].replace("Government","Aster Care Hospital")
    df["Sector En"]=df["Sector En"].replace("Total","NMC Healthcare")
    df["Sector En"]= df["Sector En"].replace("Private","Prime Hospital")
    #print(df["Sector En"].unique())
    df["Total"]=df["Total"]//10
    df.rename(columns={"Sector En": "Hospital Name"}, inplace=True)
    df.rename(columns={"Total": "Number of Hospital Beds"}, inplace=True)
    beds_in_dubai = df[df["Emirate En"] == Emirate][["Hospital Name","Number of Hospital Beds"]]
    print(df.info())
    print(type(beds_in_dubai))
    table_text = beds_in_dubai.to_string(index=False)
    return table_text
    

hospital_beds("Sharjah")
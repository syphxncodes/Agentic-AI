import pandas as pd
import numpy as np
from langchain_community.llms import Ollama
from langgraph.graph import Graph
from langchain_core.prompts import PromptTemplate
from langchain_google_community import GmailToolkit
from langchain_google_community.gmail.send_message import GmailSendMessage
import regex as re
import json 
from langchain.output_parsers import StructuredOutputParser
from langchain.output_parsers.json import parse_json_markdown
import csv
import datetime
from io import StringIO
import smtplib
from email.message import EmailMessage
import os
from langchain_community.tools import DuckDuckGoSearchRun
  # Safely extracts JSON block from markdown

send_tool=GmailSendMessage()


MODEL = "gemma3:12b"
model = Ollama(model=MODEL)
most_frequent_disease="Influenza"
def get_data_from_dataset(state):
    try:
        
        state = state or {}
        
        
        try:
            df = pd.read_csv("conditions.csv")
            most_frequent_disease = df["DESCRIPTION"].value_counts().idxmax()
            # Only replace if needed for simulation
            if most_frequent_disease:
                df["DESCRIPTION"] = df["DESCRIPTION"].replace(most_frequent_disease, "Influenza")
                most_frequent_disease = "Influenza"
            
            hospital_summary = f"According to the hospital data, the most frequent disease in this week's span is {most_frequent_disease}"
            state["hospital_data"] = hospital_summary
        except FileNotFoundError:
            
            state["hospital_data"] = "No hospital data available - file not found"
        except Exception as e:
           
            state["hospital_data"] = f"Error processing hospital data: {str(e)}"
            
        return state
    except Exception as e:
        # Always return a valid state, even on error
        return {"hospital_data": f"Error: {str(e)}"}

def get_data_from_Twitter(state):
    try:
        # Ensure state exists
        state = state or {}
        
        try:
            df1 = pd.read_csv("unique_medical_tweets_dataset.csv")
            x = df1["keyword"].value_counts().idxmax()
            if x:
                df1["keyword"] = df1["keyword"].replace(x, "Influenza")
                x = "Influenza"
            
            twitter_data = f"According to the tweets related to medical health, the most frequent in this week's span is {x}"
            state["twitter_data"] = twitter_data
        except FileNotFoundError:
            state["twitter_data"] = "No Twitter data available - file not found"
        except Exception as e:
            
            state["twitter_data"] = f"Error processing Twitter data: {str(e)}"
            
        return state
    except Exception as e:
        # If state was None, initialize it
        if state is None:
            state = {}
        state["twitter_data"] = f"Error: {str(e)}"
        return state
msg=""
def analyze_with_llm(state):
    global msg
    try:
        # Ensure state exists
        state = state or {}
        df=pd.read_csv("synthetic_medical_supply_with_severity.csv")
        severity_avg=df["severity"].mean()
        people_avg_per_week=df["people_per_week"].mean()
        # Check if required data exists
        if "hospital_data" not in state:
            state["hospital_data"] = "No hospital data available"
        if "twitter_data" not in state:
            state["twitter_data"] = "No Twitter data available"
        
        prompt = PromptTemplate.from_template("""You are an intelligent agent tasked with analyzing the following data:  
- **Hospital Data**: {hospital}  
- **Twitter Data**: {twitter}  

Using this information:  
1. Provide a concise summary of the situation.  
2. If there is any indication of a disease outbreak, make sure to explicitly include the word **"outbreak"** in your response.  
3. Identify the disease, and list the medicines typically used to treat it.  
4. Provide a list of required medical equipment (just the names and quantities — no descriptions).  

Additional data for your consideration:  
- **Average severity**: {severity_avg}  
- **Average number of people per week**: {people_avg_per_week}  

Based on this, calculate the **total number of medicines required per hospital**, including the medicine names and quantities.  
Note: Since both the severity and the average number of people per week are high, don’t hesitate to recommend a **higher** amount of medical resources.  

In your final response, include:  
- Medicines (with names and quantities)[Include your own medicines, but also consider medicines from ['Ciprofloxacin' 'Atorvastatin' 'Wheelchair' 'Salbutamol' 'Doxycycline'
 'Scalpel' 'Pulse Oximeter' 'Lisinopril' 'Defibrillator' 'Fluconazole'
 'Paracetamol' 'Oxygen Cylinder' 'Hand Sanitizer' 'Amoxicillin'
 'Sterile Gauze' 'Disposable Gloves' 'IV Bag' 'Surgical Scissors'
 'Stethoscope' 'Pantoprazole' 'Levothyroxine' 'Blood Pressure Monitor'
 'Surgical Masks' 'Azithromycin' 'Thermometer' 'Omeprazole' 'Ventilator'
 'Ibuprofen' 'Metronidazole' 'Aspirin' 'Hydrochlorothiazide' 'Syringe Set'
 'Simvastatin' 'Bandage Roll' 'Metformin' 'Prednisolone' 'Drip Stand'
 'ECG Machine' 'Losartan' 'Insulin Pen']](Consider other medicines as well as required) - Consider the outbreak average as well and suggest(and give the number of medicines as a surplus to the hospital per week, give a good amount from 100-500), because you are suggesting less.
- Equipment (names and quantities).Give the reason behind such number.
Present all outputs in clear, in json format (with equipment:[name,quantity], and medicines:[name,quantity]). Give different quantities, that are believable.Don't give repeated.
 Always only return json for medicines and equipment, not for the summary. For the summary, return normal text.JSON IS VERY IMPORTANT.""")
        
        final_prompt = prompt.format(
            hospital=state["hospital_data"],
            twitter=state["twitter_data"],
            severity_avg=severity_avg,
            people_avg_per_week=people_avg_per_week
        )
        
        try:
            result = model.invoke(final_prompt)
            msg+=result
            state["model_analysis"] = result
        except Exception as e:
            
            state["model_analysis"] = f"Error analyzing data: {str(e)}"
            
        return state
    except Exception as e:
        
        return {"model_analysis": f"Error in analysis: {str(e)}", 
                "hospital_data": state.get("hospital_data", "Missing"), 
                "twitter_data": state.get("twitter_data", "Missing")}


graph = Graph()
graph.add_node("Hospital", get_data_from_dataset)
graph.add_node("Twitter", get_data_from_Twitter)
graph.add_node("analyze", analyze_with_llm)
graph.set_entry_point("Hospital")
graph.set_finish_point("analyze")  
graph.add_edge("Hospital", "Twitter")
graph.add_edge("Twitter", "analyze")


# Compile and invoke
app = graph.compile()
try:
    final_state = app.invoke({})
    if final_state is None:
        raise ValueError("Graph returned None - check finish point")
except Exception as e:
    print(f"Execution error: {e}")
    final_state = {}


flattened = {}
if isinstance(final_state, dict):
    for step_data in final_state.values() if all(isinstance(v, dict) for v in final_state.values()) else [final_state]:
        flattened.update(step_data)
        
# Output result
print(flattened.get("model_analysis", "Model analysis not found."))
#Sends active responses about the outbreak.
msg=flattened.get("model_analysis","")
outbreak_detected = bool(re.search(r'\boutbreak\b', msg, re.IGNORECASE))
print(outbreak_detected)
if (outbreak_detected):
    x=model.invoke("Give a list of 5 preventive measures for influenza, and give it in a formal way.")
    list1=["kodithyalasaiuday1234@gmail.com","f20230209@dubai.bits-pilani.ac.in","f20230208@dubai.bits-pilani.ac.in","f20230241@dubai.bits-pilani.ac.in"]
    for str11 in list1:
         #str1=f"hi guys. There is an outbreak of influenza.Be very careful. Savdhan rahe, sathark rahe."
         response=send_tool.invoke({
            "to":str11,
            "subject":"Savdhan Rahe, Sathark Rahe",
            "message": x })
         
data = parse_json_markdown(msg)
#def write_csv(data_list, filename):
    #if not data_list:
        #print(f"No data to write for {filename}.")
        #return
    #with open(filename, mode='w', newline='', encoding='utf-8') as file:
        #writer = csv.DictWriter(file, fieldnames=data_list[0].keys())
        #writer.writeheader()
        #writer.writerows(data_list)
    #print(f"Data successfully written to {filename}.")
#write_csv(data['medicines'], 'medicines.csv')
#write_csv(data['equipment'], 'equipment.csv')

#EMAIL_ADDRESS = "f20230254@dubai.bits-pilani.ac.in"
#EMAIL_PASSWORD = "zxje eoqp rnxa vrxu"  

#msg = EmailMessage()
#msg['Subject'] = 'Ordered Data CSV'
#msg['From'] = EMAIL_ADDRESS
#msg['To'] = 'kodithyalasaiuday1234@gmail.com'
#msg.set_content('The 2 attached csv files are an order list of the extra required medicines and equipment because of the particular outbreak.')


#filename = 'medicines.csv'
#with open(filename, 'rb') as f:
    #file_data = f.read()
    #msg.add_attachment(file_data, maintype='text', subtype='csv', filename=filename)

#with open("equipment.csv", "rb") as f:
    #file_data = f.read()
    #msg.add_attachment(file_data, maintype='text', subtype='csv', filename="equipment.csv")

#with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
    #smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
    #smtp.send_message(msg)

#print("Email sent successfully with attachment.")

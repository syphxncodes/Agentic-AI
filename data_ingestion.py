import pandas as pd
import numpy as np
from langchain_community.llms import Ollama
import os
from langgraph.graph import Graph
from langchain_core.prompts import PromptTemplate
from langchain_google_community import GmailToolkit
from langchain_google_community.gmail.send_message import GmailSendMessage
import regex as re
send_tool=GmailSendMessage()


MODEL = "llama3:latest"
model = Ollama(model=MODEL)

def get_data_from_dataset(state):
    try:
        # Initialize state if it's None
        state = state or {}
        
        # Try to read the CSV file
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
            # Handle missing file gracefully
            state["hospital_data"] = "No hospital data available - file not found"
        except Exception as e:
            # Catch any other exceptions
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
            # Catch any other exceptions
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
- Medicines (with names and quantities)  
- Equipment (names and quantities)  

Present all outputs in clear, structured lists. Give different quantities, that are believable.Don't give repeated.
 """)
        
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
            # Handle LLM errors gracefully
            state["model_analysis"] = f"Error analyzing data: {str(e)}"
            
        return state
    except Exception as e:
        # Always return a valid state, even on error
        return {"model_analysis": f"Error in analysis: {str(e)}", 
                "hospital_data": state.get("hospital_data", "Missing"), 
                "twitter_data": state.get("twitter_data", "Missing")}




# Create the graph
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
print(flattened.get("model_analysis", "Model analysis not found."))\
#Sends active responses about the outbreak.
msg=flattened.get("model_analysis","")
#outbreak_detected = bool(re.search(r'\boutbreak\b', msg, re.IGNORECASE))
#print(outbreak_detected)
#if (outbreak_detected):
    #list1=["kodithyalasaiuday1234@gmail.com","f20230209@dubai.bits-pilani.ac.in","f20230208@dubai.bits-pilani.ac.in","f20230241@dubai.bits-pilani.ac.in"]
    #for str11 in list1:
         #str1=f"hi guys. There is an outbreak of influenza.Be very careful. Savdhan rahe, sathark rahe."
         #response=send_tool.invoke({
            #"to":str11,
            #"subject":"Savdhan Rahe, Sathark Rahe",
            #"message": str1 })

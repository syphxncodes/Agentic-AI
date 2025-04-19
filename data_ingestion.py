import pandas as pd
import numpy as np
from langchain_community.llms import Ollama
import os
from langgraph.graph import Graph
from langchain_core.prompts import PromptTemplate


MODEL = "gemma2:latest"
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

def analyze_with_llm(state):
    try:
        # Ensure state exists
        state = state or {}
        
        # Check if required data exists
        if "hospital_data" not in state:
            state["hospital_data"] = "No hospital data available"
        if "twitter_data" not in state:
            state["twitter_data"] = "No Twitter data available"
        
        prompt = PromptTemplate.from_template("""
        You are an agent, which analyzes the data from                                   
        Hospital Data: {hospital}
        Twitter Data: {twitter}
        Based on this information, summarize.                                                                                                                
        """)
        
        final_prompt = prompt.format(
            hospital=state["hospital_data"],
            twitter=state["twitter_data"]
        )
        
        try:
            result = model.invoke(final_prompt)
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
print(flattened.get("model_analysis", "Model analysis not found."))
from langchain_community.tools import DuckDuckGoSearchRun
search = DuckDuckGoSearchRun()
disease="Influenza"
print(search.invoke(f'Give 5 points on preventive measures for the {disease} as a list'))
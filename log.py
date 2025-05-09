  import os
import re
import hashlib
import pandas as pd
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from fuzzywuzzy import process
from langchain.docstore.document import Document
from rich.console import Console
from rich.prompt import Prompt
from datetime import datetime
from io import StringIO
from tqdm import tqdm

# === Configuration ===
CSV_FILE = "supply_log.csv"
NEW_LOGS_FILE = "updated_medical_supply.csv"
PERSIST_PATH = "db/supply_vectordb"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"

# === Init Models ===
embedding_model = OllamaEmbeddings(model=EMBED_MODEL)
llm = ChatOllama(model=LLM_MODEL)
console = Console()

def normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', re.sub(r'\W+', ' ', text)).strip().lower()

def detect_query_type(query: str) -> str:
    prompt = f"""
You are a smart agent that classifies hospital supply queries.

Decide the type:
- Respond with "structured" if the query asks for specific data from known fields like hospital name, supply name, quantity, wastage, severity, etc.
- Respond with "semantic" if the query is vague, comparative, or requires interpretation or summarization.
- Respond with "hospital_check" **only if** the query is asking whether a specific hospital exists or not (e.g. "Is there a hospital named Apollo?").

Return exactly one of: structured, semantic, hospital_check.

Query: "{query}"

Answer:"""
    return llm.invoke(prompt).content.strip().lower()



def answer_structured(query: str) -> str:
    df = pd.read_csv(CSV_FILE)
    q = query.lower()
    if any(phrase in q for phrase in ["no of entries", "number of entries", "how many rows", "total entries"]):
        return f"📊 Total number of entries in the supply log: {len(df)}"

    # Prioritize "details of hospital" queries
    if "details" in q and "hospital" in q:
        hosp_match = re.search(r"(?:in|of|at|from)?\s*([\w\s]+(?:hospital))", q)
        if hosp_match:
            query_hosp = normalize(hosp_match.group(1))
            df["hospital_name_normalized"] = df["hospital_name"].apply(normalize)
            hosp_choices = df["hospital_name_normalized"].unique().tolist()
            best_match, score = process.extractOne(query_hosp, hosp_choices)
            if score > 80:
                df = df[df["hospital_name_normalized"] == best_match]
                return df[[ "hospital_name", "supply_name", "supplier_name",
                            "quantity_supplied", "weekly_wastage", "severity", "people_per_week"]].to_string(index=False)
            else:
                return f"❌ No hospital matched '{query_hosp}' (best: '{best_match}', score={score})"
        return "⚠️ No hospital name found for details request."

    # Regex match blocks
    hosp_match = re.search(r"(?:in|from|at|of|og|by)\s+([\w\s]+(?:hospital)?)", q)
    supply_match = re.search(r"(?:for|of|og)\s+([\w\s]+?)(?:\s+quantity|\s+greater|\s+in|\s+at|\s+from|$)", q)
    qty_match = re.search(r"(?:above|greater than|more than|over)\s+(\d+)", q)
    people_match = re.search(r"people(?:\s+per\s+week)?\s*(?:above|more than|greater than|over)?\s*(\d+)", q)
    wastage_match = re.search(r"wastage(?:\s+above|\s+more than|\s+greater than|\s+over)?\s*(\d+)", q)
    severity_match = re.search(r"severity\s*(high|medium|low|\d+)", q)

    # Supplier match (guarded)
    supplier_match = re.search(r"(?:supplied\s+by|from\s+supplier|supplier)\s+([\w\s]+)", q)
    if supplier_match:
        query_supplier = normalize(supplier_match.group(1))
        if "hospital" in query_supplier or "bought" in query_supplier or "received" in query_supplier:
            supplier_match = None
        else:
            df["supplier_name_normalized"] = df["supplier_name"].apply(normalize)
            supplier_choices = df["supplier_name_normalized"].unique().tolist()
            best_match, score = process.extractOne(query_supplier, supplier_choices)
            if score > 80:
                df = df[df["supplier_name_normalized"] == best_match]
            else:
                return f"❌ No supplier matched '{query_supplier}' (best: '{best_match}', score={score})"

    # Hospital match
    if hosp_match:
        query_hosp = normalize(hosp_match.group(1))
        df["hospital_name_normalized"] = df["hospital_name"].apply(normalize)
        hosp_choices = df["hospital_name_normalized"].unique().tolist()
        best_match, score = process.extractOne(query_hosp, hosp_choices)
        if score > 80:
            df = df[df["hospital_name_normalized"] == best_match]

            # ✅ Detect "last N transactions"
            last_n_match = re.search(r"(?:last|recent)\s+(\d+)", q)
            if last_n_match:
                n = int(last_n_match.group(1))
                df = df.tail(n)

        else:
            return f"❌ No hospital matched '{query_hosp}' (best: '{best_match}', score={score})"

    if supply_match:
        query_supply = normalize(supply_match.group(1))
        if query_supply not in ["supplies", "supply", "all supplies", "any supplies", "the supplies"]:
            supply_choices = df["supply_name"].str.lower().tolist()
            best_match, score = process.extractOne(query_supply, supply_choices)
            if score > 80:
                df = df[df["supply_name"].str.lower() == best_match]
            else:
                return f"❌ No supply matched '{query_supply}' (best: '{best_match}', score={score})"

    if qty_match:
        df = df[df["quantity_supplied"] >= int(qty_match.group(1))]

    if people_match:
        df = df[df["people_per_week"] >= int(people_match.group(1))]

    if wastage_match:
        df = df[df["weekly_wastage"] >= int(wastage_match.group(1))]

    if severity_match:
        level = severity_match.group(1).lower()
        df = df[df["severity"].astype(str).str.lower() == level]

    if df.empty:
        return "⚠️ No matching results. Try refining your query."

    return df[[ "hospital_name", "supply_name", "supplier_name",
                "quantity_supplied", "weekly_wastage", "severity", "people_per_week"]].to_string(index=False)




def append_to_vector_db(new_logs: pd.DataFrame, db: Chroma, batch_size: int = 5000):
    docs, ids = [], []
    for _, row in new_logs.iterrows():
        content = (
            f"Hospital: {row['hospital_name']}, Supply: {row['supply_name']}, "
            f"Supplier: {row['supplier_name']}, Quantity: {row['quantity_supplied']}, "
            f"Wastage: {row['weekly_wastage']}, Severity: {row['severity']}, "
            f"People/Week: {row['people_per_week']}"
        )
        uid = row['hashcode'] if 'hashcode' in row else hashlib.md5(content.encode()).hexdigest()
        docs.append(Document(page_content=content))
        ids.append(uid)
        for i in tqdm(range(0, len(docs), batch_size), desc="🔄 Indexing to vector DB"):
            db.add_documents(docs[i:i+batch_size], ids=ids[i:i+batch_size])

    # Split into chunks
    for i in range(0, len(docs), batch_size):
        db.add_documents(docs[i:i+batch_size], ids=ids[i:i+batch_size])

    console.print(f"✅ Appended {len(docs)} new logs to vector DB in batches of {batch_size}.")


def answer_semantic(query: str) -> str:
    db = Chroma(persist_directory=PERSIST_PATH, embedding_function=embedding_model)
    results = db.similarity_search_with_score(query, k=5)
    if not results:
        return "⚠️ No relevant documents found in semantic search."

    top_docs = [doc.page_content for doc, _ in results]

    prompt = f"""You are an intelligent hospital supply assistant. 
Using the below data, answer the user's question accurately and clearly.

Data:
{chr(10).join(top_docs)}

Question: {query}

Answer:"""

    return llm.invoke(prompt).content.strip()


def append_new_logs():
    if os.path.exists(NEW_LOGS_FILE) and os.path.getsize(NEW_LOGS_FILE) > 0:
        try:
            print(f"📄 Reading from file: {NEW_LOGS_FILE}")
            with open(NEW_LOGS_FILE, 'r', encoding='utf-8-sig') as f:
                lines = [line.strip() for line in f if line.strip()]
            print(f"📊 Non-empty lines read: {len(lines)}")

            content_buffer = StringIO("\n".join(lines))
            df_new_logs = pd.read_csv(content_buffer, header=None, names=[
                "supply_name", "supplier_name", "quantity_supplied",
                "hospital_name", "weekly_wastage", "severity", "people_per_week"
            ])

            def generate_hash(row):
                content = (
                    f"Hospital: {row['hospital_name']}, Supply: {row['supply_name']}, "
                    f"Supplier: {row['supplier_name']}, Quantity: {row['quantity_supplied']}, "
                    f"Wastage: {row['weekly_wastage']}, Severity: {row['severity']}, "
                    f"People/Week: {row['people_per_week']}"
                )
                return hashlib.md5(content.encode()).hexdigest()

            df_new_logs["hashcode"] = df_new_logs.apply(generate_hash, axis=1)

            try:
                df_existing = pd.read_csv(CSV_FILE)
            except pd.errors.EmptyDataError:
                df_existing = pd.DataFrame(columns=df_new_logs.columns)

            df_combined = pd.concat([df_existing, df_new_logs], ignore_index=True)
            df_combined.to_csv(CSV_FILE, index=False)

            print(f"✅ Appended {len(df_new_logs)} new logs to {CSV_FILE}")
            with open(NEW_LOGS_FILE, 'w') as f:
                f.truncate(0)

            db = Chroma(persist_directory=PERSIST_PATH, embedding_function=embedding_model)
            append_to_vector_db(df_new_logs, db)

        except Exception as e:
            print(f"❌ Error during append: {e}")
    else:
        print(f"⚠️ No new logs to process or {NEW_LOGS_FILE} is empty.")

def build_vector_db():
    df = pd.read_csv(CSV_FILE)
    if 'hashcode' not in df.columns:
        def generate_hash(row):
            content = (
                f"Hospital: {row['hospital_name']}, Supply: {row['supply_name']}, "
                f"Supplier: {row['supplier_name']}, Quantity: {row['quantity_supplied']}, "
                f"Wastage: {row['weekly_wastage']}, Severity: {row['severity']}, "
                f"People/Week: {row['people_per_week']}"
            )
            return hashlib.md5(content.encode()).hexdigest()
        df["hashcode"] = df.apply(generate_hash, axis=1)
        df.to_csv(CSV_FILE, index=False)
        print("🧾 Added hashcodes to supply_log.csv")

    docs, ids = [], []
    for _, row in df.iterrows():
        content = (
            f"Hospital: {row['hospital_name']}, Supply: {row['supply_name']}, "
            f"Supplier: {row['supplier_name']}, Quantity: {row['quantity_supplied']}, "
            f"Wastage: {row['weekly_wastage']}, Severity: {row['severity']}, "
            f"People/Week: {row['people_per_week']}"
        )
        docs.append(Document(page_content=content))
        ids.append(row['hashcode'])

    db = Chroma.from_documents(docs, embedding=embedding_model, persist_directory=PERSIST_PATH, ids=ids)
    print(f"✅ Indexed {len(docs)} documents into vector DB.")

def check_hospital_exists(query: str) -> str:
    df = pd.read_csv(CSV_FILE)
    q = query.lower()
    match = re.search(r"(?:is there\s+)?(?:a|the)?\s*([\w\s]+(?:hospital)?)", q)
    if match:
        query_hosp = normalize(match.group(1))
        df["hospital_name_normalized"] = df["hospital_name"].apply(normalize)
        hosp_choices = df["hospital_name_normalized"].unique().tolist()
        best_match, score = process.extractOne(query_hosp, hosp_choices)
        if score > 80:
            return f"✅ Hospital '{best_match}' found."
        else:
            return f"❌ No hospital matched '{query_hosp}' (best: '{best_match}', score={score})"
    return "⚠️ No hospital name found."

def main():
    if not os.path.exists(PERSIST_PATH):
        os.makedirs(PERSIST_PATH)
    if not os.path.exists(CSV_FILE):
        print(f"❌ Missing CSV file: {CSV_FILE}")
        return

    console.print("[bold green]🏥 Hospital Supply Query Assistant[/bold green]")

    

    append_new_logs()

    while True:
        query = Prompt.ask("\n🧠 Ask your query (or type 'exit')")
        if query.lower() == "exit":
            break

        query_type = detect_query_type(query)
        console.print(f"[yellow]🔍 Query Type: {query_type}[/yellow]")

        if query_type == "structured":
            result = answer_structured(query)
        elif query_type == "semantic":
            result = answer_semantic(query)
        elif query_type == "hospital_check":
            result = check_hospital_exists(query)
        else:
            result = "❌ Unable to determine query type."

        console.print(f"\n📋 [bold]Answer:[/bold]\n{result}")

if __name__ == "__main__":
    main()

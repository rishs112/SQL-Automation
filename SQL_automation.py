import os
import getpass
import time
from typing import TypedDict, List, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from sqlalchemy import create_engine, inspect, text

def _set_env(var: str):
    """Sets an environment variable if it's not already set."""
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Enter your {var}: ")

_set_env("GOOGLE_API_KEY")

DB_CONNECTION_STRING = "postgresql+psycopg2://sql_agent_user:N0cturn3112@localhost:5432/adventureworks"

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)

sanity_check_prompt_template = """
You are a gatekeeper for a database query agent. Your job is to determine if a user's question is a valid, good-faith question that can be answered by a database, or if it is nonsensical.

- A **valid** question asks for specific information. Examples: "How many customers are there?", "Who are the top salespeople?", "What are the products in the 'Bikes' category?"
- A **nonsensical** question is a random word, greeting, or statement that does not ask for data. Examples: "yes", "hello", "asdfghjkl", "the sky is blue"

User's question: "{question}"

Is this a valid question for a database agent? You MUST respond with only the single word: "valid" or "invalid".
"""
sanity_check_prompt = ChatPromptTemplate.from_template(sanity_check_prompt_template)

table_selection_prompt_template = """
You are an expert database administrator. Your task is to identify which database tables are relevant to a user's question.

Here is a list of all available tables in the database:
{table_list}

Here is the user's question:
"{question}"

Based on the user's question, which of the tables are necessary to form a SQL query? Please provide your answer as a single, comma-separated list of table names.

Example: person.person, sales.salesorderheader, sales.salesperson
"""
table_selection_prompt = ChatPromptTemplate.from_template(table_selection_prompt_template)

ambiguity_check_prompt_template = """
You are an expert at analyzing user questions for a database agent. Your task is to determine if a question is ambiguous or if it is clear enough to be answered with a single SQL query.

- An **ambiguous** question is one that could have multiple valid interpretations. For example, "Who are the best customers?" is ambiguous because 'best' could mean highest spending, most frequent, or something else.
- A **clear** question has only one logical interpretation. For example, "Which customers are from London?" is clear.
- A question that contains a clarification is also **clear**. For example, "Original question: 'Who are the best customers?'. User provided clarification: 'The ones who have spent the most money'" is clear.

Here is the user's question: "{question}"

Analyze the question. If it is clear, you MUST respond with only the single word: "clear".
If the question is ambiguous, you MUST respond with a clarifying question to ask the user.
"""
ambiguity_check_prompt = ChatPromptTemplate.from_template(ambiguity_check_prompt_template)

prompt_template = """
You are an expert PostgreSQL developer. Your task is to write a SQL query based on a user's question and the provided database schema.

**IMPORTANT RULE**: In your SELECT statement, you MUST include the columns the user is asking for. You MUST ALSO include any columns that are used in the WHERE clause for filtering, as this context is needed by a later step.

If you are correcting a previous query, ensure you fix the issues mentioned in the error message.

Here are some examples of questions and their correct SQL queries:
---
Question: "Who are the top 5 salespeople by total sales?" 
SQL: SELECT p.FirstName, p.LastName, SUM(soh.TotalDue) AS TotalSales FROM sales.salesperson sp 
JOIN person.person p ON sp.businessentityid = p.businessentityid 
JOIN sales.salesorderheader soh ON sp.businessentityid = soh.salespersonid 
GROUP BY p.FirstName, p.LastName 
ORDER BY TotalSales 
DESC LIMIT 5;
---
Question: How many products are in the 'Bikes' category? Return only the count.
SQL: SELECT COUNT(T1.productid) FROM production.product AS T1 
INNER JOIN production.productsubcategory AS T2 ON T1.productsubcategoryid = T2.productsubcategoryid 
INNER JOIN production.productcategory AS T3 ON T2.productcategoryid = T3.productcategoryid 
WHERE T3.name = 'Bikes';
---

Here is the history of the conversation so far:
{history}
Here is the database schema:
{schema}
Here is the user's question:
{question}

{error}

Please provide only the SQL query, with no additional text or explanation.
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

summarizer_prompt_template = """
You are a helpful AI assistant. Your task is to provide a clear, friendly, and concise answer to the user's question based on the data provided.

Here is the history of the conversation so far, for context:
{history}
Here was the user's original question:
"{question}"
Here is the data retrieved from the database:
{results}

Please provide a natural language response to the user's question. If the data is empty or an error, inform the user that you couldn't find an answer.
"""
summarizer_prompt = ChatPromptTemplate.from_template(summarizer_prompt_template)

class AgentState(TypedDict):
    """
        user_question: The natural language question from the user.
        db_schema: A string representation of the database schema.
        sql_query: The SQL query generated by the LLM.
        query_results: The results from executing the SQL query.
        error_message: Any error messages encountered during execution.
    """
    user_question: str
    db_schema: str
    sql_query: str
    query_results: List[Any]
    error_message: str
    final_response: str
    conversation_history: List[str]
    clarifying_question: str
    is_valid_question: bool

TABLE_LIST_CACHE = None

def user_input_node(state: AgentState):
    """Entry point: Gets the user's question and adds it to the state."""
    question = state.get('user_question')
    print("---Node: user_input_node---")
    print(f"Received user question: '{question}'")
    return {"user_question": question, 
            "conversation_history": state.get('conversation_history', []),
            "db_schema": "", 
            "sql_query": "", 
            "query_results": [],
            "error_message": "",
            "final_response": "",
            "clarifying_question": ""}

def sanity_check_node(state: AgentState):
    print("---Node: sanity_check_node (Checking for valid question...)---")
    time.sleep(2)
    prompt_text = sanity_check_prompt.format(question=state['user_question'])
    response = llm.invoke(prompt_text)
    content = response.content.strip().lower()
    if "valid" in content:
        print("---Decision: Question is valid.---")
        return {"is_valid_question": True}
    else:
        print("---Decision: Question is not valid.---")
        return {"is_valid_question": False}

def handle_invalid_question_node(state: AgentState):
    return {"final_response": "I'm sorry, that doesn't seem to be a question I can answer. Please ask a question about the database."}

def check_ambiguity(state: AgentState) -> str:
    """Checks if the user's question is ambiguous using the LLM."""
    time.sleep(2)
    response = llm.invoke(ambiguity_check_prompt.format(question = state['user_question']))
    content = response.content.strip()

    if content.lower() == "clear":
        print("---Decision: Question is clear.---")
        return {"clarifying_question": ""}
    else:
        print(f"---Decision: Question is ambiguous. Asking for clarification: {content}---")
        return {"clarifying_question": content}
    
def schema_explorer_node(state: AgentState):
    """Connects to the database and retrieves the schema of all tables."""
    global TABLE_LIST_CACHE
    print("---Node: schema_explorer_node (Dynamically fetching schema...)---")
    try:
        engine = create_engine(DB_CONNECTION_STRING)
        inspector = inspect(engine)
        
        if TABLE_LIST_CACHE is None:
            print("---(Cache miss! Fetching all table names from DB)---")
            all_tables = []
            all_schemas = inspector.get_schema_names()
            for schema in all_schemas:
                if not schema.startswith('pg_') and schema != 'information_schema':
                    for table_name in inspector.get_table_names(schema=schema):
                        all_tables.append(f"{schema}.{table_name}")
            TABLE_LIST_CACHE = all_tables
        else:
            print("---(Cache hit! Using stored table names)---")
            all_tables = TABLE_LIST_CACHE

        time.sleep(2) 
        table_list_str = "\n".join(all_tables)
        selection_prompt_text = table_selection_prompt.format(
            table_list=table_list_str,
            question=state['user_question']
        )
        response = llm.invoke(selection_prompt_text)
        relevant_tables_str = response.content.strip()
        relevant_tables = [table.strip() for table in relevant_tables_str.split(',')]
        print(f"Identified relevant tables: {relevant_tables}")

        schema_info = []
        for table_full_name in relevant_tables:
            if '.' in table_full_name:
                schema, table_name = table_full_name.split('.')
                columns = inspector.get_columns(table_name, schema=schema)
                column_details = ", ".join([f"{col['name']} ({col['type']})" for col in columns])
                schema_info.append(f"Table '{table_full_name}': [{column_details}]")
        
        formatted_schema = "\n".join(schema_info)
        return {"db_schema": formatted_schema}

    except Exception as e:
        return {"db_schema": f"Error during dynamic schema fetching: {e}"}

def query_generator_node(state: AgentState):
    """Generates an SQL query using the LLM."""
    print("---Node: query_generator_node---")
    time.sleep(2)
    error_info = state.get("error_message")
    error_prompt_section = ""
    if error_info:
        error_prompt_section = f"Your previous query failed with the following error:\n{error_info}\nPlease correct the query and try again."

    query_prompt_text = prompt.format(
        schema = state['db_schema'],
        question = state['user_question'],
        history = "\n".join(state['conversation_history']),
        error = error_prompt_section
    )
    response = llm.invoke(query_prompt_text)
    sql_query = response.content.strip().replace("```sql", "").replace("```", "")
    print(f"Generated SQL Query: \n{sql_query}")
    return {"sql_query": sql_query, "errore_message": ""}

def sql_executor_node(state: AgentState):
    """Executes the generated SQL query against the database."""
    print("---Node: sql_executor_node---")
    sql_query = state.get('sql_query')
    if not sql_query:
        return {"query_results": ["Error: No SQL query to execute."]}
    
    try:
        engine = create_engine(DB_CONNECTION_STRING)
        with engine.connect() as connection:
            result = connection.execute(text(sql_query))
            results_list = [dict(row._mapping) for row in result.fetchall()]
            print("✅ Query executed successfully.")
            return {"query_results": results_list, "error_message": ""}
            
    except Exception as e:
        print(f"❌ Error executing SQL query: {e}")
        return {"query_results": [], "error_message": str(e)}

def response_summarizer_node(state: AgentState):
    """Summarizes the query results into a final response."""
    print("---Node: response_summarizer_node---")
    time.sleep(2)
    summarizer_prompt_text = summarizer_prompt.format(
        question = state['user_question'],
        results = str(state['query_results']),
        history = "\n".join(state['conversation_history'])
    )
    response = llm.invoke(summarizer_prompt_text)
    return {"final_response": response.content}

def route_after_sanity_check(state: AgentState):
    if state.get("is_valid_question"):
        return "ambiguityCheck"
    else:
        return "handleInvalidQuestion"

def route_after_ambiguity_check(state: AgentState):
    """Routes to schema exploration or asks for clarification based on ambiguity check."""
    if state.get("clarifying_question"):
        print(f"---Decision: Asking user for clarification: {state['clarifying_question']}---")
        return "askUser"  
    else:
        print("---Decision: Question is clear, proceeding to schema exploration---")
        return "schemaExplorer"

def decide_next_node(state: AgentState):
    """Checks for errors to decide the next step."""
    if state.get("error_message"):
        print("---Decision: Error detected, looping back to query generator---")
        return "queryGenerator"
    else:
        print("---Decision: No error, proceeding to summarize results---")
        return "responseSummarizer"

workflow = StateGraph(AgentState)
workflow.add_node("userInput", user_input_node)
workflow.add_node("sanityCheck", sanity_check_node)
workflow.add_node("handleInvalidQuestion", handle_invalid_question_node)
workflow.add_node("ambiguityCheck", check_ambiguity)
workflow.add_node("schemaExplorer", schema_explorer_node)
workflow.add_node("queryGenerator", query_generator_node)
workflow.add_node("sqlExecutor", sql_executor_node)
workflow.add_node("responseSummarizer", response_summarizer_node)
workflow.add_node("askUser", lambda x: x) # Placeholder node to end the workflow when asking user

workflow.set_entry_point("userInput")
workflow.add_edge("userInput", "sanityCheck")
workflow.add_conditional_edges(
    "sanityCheck",
    route_after_sanity_check,
    {"ambiguityCheck": "ambiguityCheck", "handleInvalidQuestion": "handleInvalidQuestion"}
)
workflow.add_edge("handleInvalidQuestion", END)
workflow.add_conditional_edges(
    "ambiguityCheck",
    route_after_ambiguity_check,
    {"askUser": END, "schemaExplorer": "schemaExplorer"}
)
workflow.add_edge("schemaExplorer", "queryGenerator")
workflow.add_edge("queryGenerator", "sqlExecutor")
workflow.add_edge("responseSummarizer", END)
workflow.add_conditional_edges(
    "sqlExecutor",
    decide_next_node,
    {"queryGenerator": "queryGenerator", "responseSummarizer": "responseSummarizer"}
)

app = workflow.compile()

if __name__ == "__main__":
    conversation_history = []
    print("Welcome to the Intelligent SQL Agent!")
    print("Type 'new' to start a new conversation, or 'quit' to exit.")

    while True:
        user_question = input("\nYour question: ")

        if user_question.lower() in ['quit', 'exit', 'stop']:
            print("Goodbye!")
            break
        
        if user_question.lower() in ['new', 'reset', 'clear']:
            conversation_history = []
            print("Kicking off a new conversation!")
            continue

        current_question = user_question
        final_state = {}

        while True:
            initial_state = {
                "user_question": current_question,
                "conversation_history": conversation_history
            }
            print("---Invoking the agent---")
            final_state = app.invoke(initial_state)

            if not final_state.get("clarifying_question"):
                break  # Exit the loop if no clarification 

            print(f"Agent needs clarification: {final_state['clarifying_question']}")
            input_question = input("Your clarification: ")
            conversation_history.append(f"User: {current_question}")
            conversation_history.append(f"Agent: {final_state['clarifying_question']}")
            conversation_history.append(f"User: {input_question}") 
            
            current_question = f"Previous question: '{current_question}'. User provided clarification: '{input_question}'"
            print("\n---Re-invoking The Agent with new context---")

        final_response = final_state.get("final_response", "I'm sorry, I couldn't generate a response.")
        print("\n---Agent Response---")
        print(final_response)
        
        conversation_history.append(f"User: {user_question}")
        conversation_history.append(f"Agent: {final_response}")

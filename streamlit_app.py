import streamlit as st
import openai
import nest_asyncio
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.llms.openai import OpenAI
import os
from pydantic import BaseModel
from crewai import Agent, Task, Crew
from crewai_tools import (
  FileReadTool,
  ScrapeWebsiteTool,
  MDXSearchTool,
  SerperDevTool
)
import plotly.express as px
import pandas as pd
import json
from dotenv import load_dotenv

load_dotenv()

# Apply nest_asyncio to handle async in notebooks
nest_asyncio.apply()

openai_api_key = os.getenv("OPENAI_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")

# Check if the environment variables are set
if openai_api_key is None:
    st.error("OPENAI_API_KEY environment variable is not set.")
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

if serper_api_key is None:
    st.error("SERPER_API_KEY environment variable is not set.")
    raise ValueError("SERPER_API_KEY environment variable is not set.")


os.environ["OPENAI_API_KEY"] = openai_api_key
openai.api_key = openai_api_key
os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo-1106'
os.environ["SERPER_API_KEY"] = serper_api_key

search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

candidates = ['Ranil Wickramasinghe', 'Sajith Premadasa', 'Anura Kumara Dissanayake']

# Streamlit App Title
st.title("Presidential Election 2024")


# Initialize or retrieve session state
if 'history' not in st.session_state:
    st.session_state.history = []  # This stores previous questions/answers

# Load Documents (Manifestos)
@st.cache_data
def load_manifestos():
    # Assuming you're loading PDFs or text files stored in directories
    manifesto_rw = SimpleDirectoryReader(input_dir='./data/manifesto_rw').load_data()
    manifesto_sp = SimpleDirectoryReader(input_dir='./data/manifesto_sp').load_data()
    manifesto_akd = SimpleDirectoryReader(input_dir='./data/manifesto_akd').load_data()
    return manifesto_rw, manifesto_sp, manifesto_akd

manifesto_rw, manifesto_sp, manifesto_akd = load_manifestos()

# Set up the LLM and VectorStoreIndex for querying the manifestos
llm = OpenAI(temperature=0, model="gpt-3.5-turbo-1106", max_tokens=4096)

# Create VectorStoreIndexes for each manifesto
manifesto_rw_index = VectorStoreIndex.from_documents(manifesto_rw)
manifesto_sp_index = VectorStoreIndex.from_documents(manifesto_sp)
manifesto_akd_index = VectorStoreIndex.from_documents(manifesto_akd)

# Create Query Engines for each manifesto
manifesto_rw_engine = manifesto_rw_index.as_query_engine(similarity_top_k=2)
manifesto_sp_engine = manifesto_sp_index.as_query_engine(similarity_top_k=2)
manifesto_akd_engine = manifesto_akd_index.as_query_engine(similarity_top_k=2)

# Combine engines for multi-document querying
query_engine_tools = [
    QueryEngineTool(query_engine=manifesto_sp_engine, metadata=ToolMetadata(name="manifesto_sp", description="Sajith Premadasa's manifesto")),
    QueryEngineTool(query_engine=manifesto_akd_engine, metadata=ToolMetadata(name="manifesto_akd", description="Anura Kumara Dissanayake's manifesto")),
    QueryEngineTool(query_engine=manifesto_rw_engine, metadata=ToolMetadata(name="manifesto_rw", description="Ranil Wickramasinghe's manifesto")),
]

# Set up a sub-query engine to allow for comparisons between candidates
sub_query_engine = SubQuestionQueryEngine.from_defaults(query_engine_tools=query_engine_tools)

# Define models for Sentiments and Results (for election prediction)
class Sentiments(BaseModel):
    tweet: str
    pred_label: str

class Results(BaseModel):
    rw: float  # Ranil Wickramasinghe's chance of winning
    sp: float  # Sajith Premadasa's chance of winning
    akd: float  # Anura Kumara Dissanayake's chance of winning

scraper = Agent(
    role="Find out election related Social View",
    goal="Get the peoples thoughts on leading presidential candidates from internet",
    tools = [scrape_tool, search_tool],
    backstory="You're working on gathering what the people of sri lanka"    
              "thinks of each of the leading presidential candiates: {candidates}"
              "You collect information that helps "
              "our company learn how people think "
              "and make informed predictions on the upcoming election. "
              "Your work is the basis for "
              "the Tweeter Sentiment Analyst to predict the sentiment for each candidate",
    allow_delegation=False,
	verbose=True
)

analyst = Agent(
    role="Tweeter Sentiment Analyst",
    goal="Help Predict the election outcome by gicing the sentiment on various "
         "tweets from internet on the {candidates}.",
    backstory="You're working on analysing the upcoming"
              "presidential election of Sri Lanka"
              "You have to use the data from the Scraper "
              "to predict the sentiment of the content  "
              "and classify presidential election tweets into: Positive(label=1), and Negative(label=0) "
              "for each of the candidates {candidates}."
              "Your work is the basis for "
              "the Win Predictor to predict the winner of the election",
    allow_delegation=False,
	verbose=True
)

predictor = Agent(
    role="Predict the Election Outcome",
    goal="Predict the election outcome based on tweet sentiments, "
         "poling data, social media comments and survay data",
    backstory="You're working on predicting the results of the upcoming"
              "presidential election of Sri Lanka"
              "You have to use the data from the Tweeter Sentiment Analyst and poling data {poling_data}"
              "to predict the likely result of the election "
              "and give a chance of winning score as a percentage "
              "for each of the leading candidates. {candidates}"
              "You can use a weighting system to give weight to each of the "
              "criteria that will affect the winning chances score. Keep that consistant everyday."
              "Sentiment should have 0.4, poling data should have 0.6",
    allow_delegation=False,
	verbose=True
)

scrape_task = Task(
    description=(
        "Prioritize the leading 3 candidates "
        f"{candidates} when finding tweets and comments\n"
        "Use the tools to gather content and identify "
        "and arrage them."
    ),
    expected_output=f"An structured array of tweets and comments from Sri Lankan community "
        "on the presidential candidates {candidates} and the up coming "
        "presidential election",
    agent=scraper,
    # output_file="tweets.csv",
    human_input=True, # get the human feedback to see if you like or not
    # async_execution=True
)

analysis_task = Task(
    description=(
        "Get a sentiment label for all the tweets from the internet"
    ),
    expected_output=f"The sentiments for all the tweets scraped by the Scaper "
        "and classify presidential election tweets into: Positive(label=1), and Negative(label=0)",
    human_input=True,
    # output_json=Sentiments, 
      # Outputs the tweets and sentiments as a JSON file'
    agent=analyst
)

prediction_task = Task(
    description=(
        "Give a percentage chance of winning for each of the 3 candidates and "
        "the sum of the percentages should be eqal to 100%. Use the given weights for the "
        "final score considering both poling data and the sentiments."
    ),
    expected_output=f"A JSON file that has the percentage chance of winning for each of "
        "the 3 leading presidential candidates. "
        "rw: Ranil Wickramasinghe, sp: Sajith Premadasa, akd: Anura Kumara Dissanayake",
    output_json=Results, 
    agent=predictor
)

# Define the crew with agents and tasks
presidential_election_crew = Crew(
    agents=[scraper, 
            analyst, 
            predictor],
    
    tasks=[scrape_task, 
           analysis_task, 
           prediction_task],
    
    verbose=True
)

# Function to run CrewAI and get election predictions
def get_election_predictions(polling_data_json):
    input_data = {
        "poling_data": polling_data_json,
        "candidates": candidates
    }
     
    # Define the crew with agents and tasks
    presidential_election_crew = Crew(
        agents=[scraper, 
                analyst, 
                predictor],
        
        tasks=[scrape_task, 
            analysis_task, 
            prediction_task],
        
        verbose=True
    )
    
    result = presidential_election_crew.kickoff(inputs=input_data)
    st.write(result)
    predicted_results = result
    return predicted_results

# Function for RAG-based question answering on selected candidate's manifesto
def rag_model_answer(candidate, question):
    if candidate == "Ranil Wickramasinghe":
        engine = manifesto_rw_engine
    elif candidate == "Sajith Premadasa":
        engine = manifesto_sp_engine
    elif candidate == "Anura Kumara Dissanayake":
        engine = manifesto_akd_engine
    else:
        return "Invalid candidate selected."
    
    response = engine.query(question)
    return response

# Streamlit App with Tabs
tab1, tab2, tab3 = st.tabs(["Manifesto Comparison", "Election Predictions", "RAG Manifesto Q&A"])

# # Chatbot Input
# user_input = st.text_input("What do you want to compair in the manifestos:")

# # Generate response and update conversation history
# if user_input:
#     # Query the sub-query engine
#     response = sub_query_engine.query(user_input)
    
#     # Append question and response to session state
#     st.session_state.history.append((user_input, str(response)))

# # Display conversation history
# if st.session_state.history:
#     st.subheader("Your Questions and Answers:")
#     for i, (question, answer) in enumerate(st.session_state.history):
#         st.markdown(f"**Q{i+1}: {question}**")
#         st.markdown(f"*A{i+1}: {answer}*")

# Manifesto Comparison Tab
with tab1:
    st.header("Compare Candidates' Manifestos")
    
    # Input for comparing manifestos
    user_input = st.text_input("What do you want to compare in the manifestos?")
    
    if user_input:
        # Query the sub-query engine
        response = sub_query_engine.query(user_input)
        
        # Append question and response to session state
        st.session_state.history.append((user_input, str(response)))
    
    # Display chatbot conversation history
    if st.session_state.history:
        st.subheader("Your Questions and Answers:")
        for i, (question, answer) in enumerate(st.session_state.history):
            st.markdown(f"**Q{i+1}: {question}**")
            st.markdown(f"*A{i+1}: {answer}*")


# Election Predictions Tab
with tab2:
    st.header("Presidential Election Predictions")
    
    # Inputs for polling data
    rw_polling = st.number_input("Ranil Wickramasinghe Polling Data (%)", value=40)
    sp_polling = st.number_input("Sajith Premadasa Polling Data (%)", value=25)
    akd_polling = st.number_input("Anura Kumara Dissanayake Polling Data (%)", value=35)

    # Convert polling data to JSON format
    polling_data = {
        "rw": rw_polling,
        "sp": sp_polling,
        "akd": akd_polling
    }
    
    polling_data_json = json.dumps(polling_data)

    if st.button("Get Election Predictions"):
        st.write("Fetching prediction data...")
        predictions = get_election_predictions(polling_data_json)
        st.success("Predictions successfully fetched!")
        
        # Prepare data for pie chart
        data = {
            "Candidates": ["Ranil Wickramasinghe", "Sajith Premadasa", "Anura Kumara Dissanayake"],
            "Chances": [predictions["rw"], predictions["sp"], predictions["akd"]]
        }
        df = pd.DataFrame(data)
        
        # Display the prediction results as a pie chart
        fig = px.pie(df, names='Candidates', values='Chances', title='Election Outcome Prediction')
        st.plotly_chart(fig)
        
        # Display the raw data in tabular form
        st.write("Prediction Data:", df)


# RAG Model Q&A Tab
with tab3:
    st.header("Ask Questions About a Candidate's Manifesto")

    # Dropdown to select a candidate
    candidate = st.selectbox("Choose a candidate", ["Ranil Wickramasinghe", "Sajith Premadasa", "Anura Kumara Dissanayake"])
    
    # Input field for the user's question
    question = st.text_input(f"Ask a question about {candidate}'s manifesto:")
    
    # Generate answer when user clicks the button
    if st.button("Get Answer"):
        answer = rag_model_answer(candidate, question)
        st.write(f"Answer: {answer}")
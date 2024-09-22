# Presidential Election Dashboard & RAG Model

This repository contains a Streamlit dashboard that allows users to:
1. **Compare presidential candidates' manifestos** using a question-answer chatbot interface.
2. **Predict the outcome of the presidential election** using polling data and scraped sentiment data from Twitter.
3. **Ask questions about a specific candidate's manifesto** using a RAG (Retrieval-Augmented Generation) model.
4. **Web scraping and data preprocessing scripts** to collect and clean relevant data.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Scripts](#scripts)
- [Technologies](#technologies)
- [Contributing](#contributing)
- [License](#license)

## Overview

The goal of this project is to provide a unified platform for comparing, analyzing, and predicting outcomes in the presidential election based on candidates' manifestos and sentiment data. The system uses both LLM-based retrieval systems and custom models to predict the potential winner of the election.

## Features

- **Manifesto Comparison**: Compare different candidates' positions by asking questions about their manifestos.
- **Election Predictions**: Predict the election outcome based on polling data and scraped sentiment analysis from social media.
- **RAG-based Q&A**: Ask questions about a candidate's manifesto, using a Retrieval-Augmented Generation model.
- **Data Processing & Scraping**: Web scraping and preprocessing of polling data and social media sentiment.
  
## Installation

Follow these steps to set up the project locally:

### Prerequisites

Ensure you have the following installed:

- Python 3.8+
- [pip](https://pip.pypa.io/en/stable/)

### Clone the Repository

```bash
git clone https://github.com/your-username/presidential-election-dashboard.git
cd presidential-election-dashboard
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` includes:

- Streamlit
- LlamaIndex
- Plotly
- pandas
- CrewAI
- pydantic
- OpenAI
- crewai_tools (for scraping)
- langchain_community

### API Keys Setup

You need to set the following API keys:

1. OpenAI API Key: Get it from OpenAI.
2. Serper API Key: Get it from Serper.dev for web scraping.

To set these keys, create a .env file in the root directory with the following content:
```
OPENAI_API_KEY=your-openai-api-key
SERPER_API_KEY=your-serper-api-key
```

## Usage

### Running the Streamlit App
To start the Streamlit app locally, run the following command:
```bash
streamlit run app.py
```
This will open the app in your default browser.

## Features in the App
1. Manifesto Comparison: Navigate to the "Manifesto Comparison" tab to ask questions and compare the candidates' positions.
2. Election Predictions: Go to the "Election Predictions" tab, input polling data, and view the prediction results in a pie chart.
3. RAG Q&A: In the "RAG Manifesto Q&A" tab, select a candidate and ask specific questions about their manifesto.

## Datasets

This repository includes the following datasets:

1. **Candidate Manifestos**:
   - `data/manifesto_rw/`: Ranil Wickramasinghe's manifesto.
   - `data/manifesto_sp/`: Sajith Premadasa's manifesto.
   - `data/manifesto_akd/`: Anura Kumara Dissanayake's manifesto.

2. **Twitter Scraped Data**:
   - `data/tweets/`: Custom scraped Twitter sentiment data on the candidates using the `web_scraper.py` script.
   - This data is used in sentiment analysis and as input for the election prediction model.

3. **Polling Data**:
   - `data/polling_data.json`: Contains sample polling data for each candidate.

### Custom Datasets

- **Scraped Tweets**: Custom scraped tweets with sentiment labels using the `web_scraper.py` script.
- **Polling Data**: Preprocessed polling data in JSON format for election prediction.


## Scripts

This repository includes various utility scripts:

### 1. `web_scraper.py`
This script scrapes Twitter for relevant tweets about the presidential candidates and stores them in a CSV file.

#### Usage

```bash
python web_scraper.py
```

### 2. data_preprocessing.py

This script preprocesses raw tweets and polling data for model input, including tasks such as cleaning, tokenization, and formatting data for sentiment analysis or prediction.

#### Usage

```bash
python data_preprocessing.py
```

### 3. sentiment_analysis.py
This script performs sentiment analysis on the scraped tweets to classify them as positive or negative using a predefined sentiment analysis model.

#### Usage
```bash
python sentiment_analysis.py
```

### 4. rag_retrieval.py
This script is responsible for the Retrieval-Augmented Generation (RAG) model that takes in questions related to candidates' manifestos, retrieves relevant data from the PDFs, and generates answers.

#### Usage
```bash
python rag_retrieval.py
```

### 5. election_prediction.py
This script runs the election prediction model by combining polling data and sentiment analysis results, and predicts the percentage chances for each candidate.

#### Usage
```bash
python election_prediction.py
```

Each of these scripts contributes to the various functionalities of the project, from data collection and preprocessing to final prediction.

## Technologies
The following technologies are used in this project:

- Python: Programming language used for the entire project.
- Streamlit: Web framework to create the dashboard.
- OpenAI GPT-3.5: For natural language generation and processing.
- CrewAI: For agent-based models and scraping.
- pandas: For data manipulation and analysis.
- Plotly: For data visualization.
- LlamaIndex: For retrieval-based QA from PDFs.

## Contributing
If you'd like to contribute to this project, please fork the repository and use a feature branch. Pull requests are warmly welcome.

1. Fork the repository (`git clone https://github.com/your-username/presidential-election-dashboard.git`).
2. Create your feature branch (`git checkout -b feature/your-feature-name`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature-name`).
5. Create a new Pull Request.

## License


## Additional Notes
- *Customization*: If you want to add more candidates or new datasets, simply update the data directory with the relevant PDFs or JSON files.
- *Testing*: All approaches, datasets, and scripts have been tested on real datasets to ensure smooth integration. You can further customize and scale the model as needed.
Let me know if you have any questions or if anything is unclear!


## Directory Structure

```bash
.
├── app.py                   
├── requirements.txt          
├── data/                    
│   ├── manifesto_rw/         # Ranil Wickramasinghe's manifesto PDF
│   ├── manifesto_sp/         # Sajith Premadasa's manifesto PDF
│   └── manifesto_akd/        # Anura Kumara Dissanayake's manifesto PDF           
├── web_scraper.py
├── tweets.csv
├── data_preprocessing.py     
├── sentiment_analysis.py
├── polling_data.json    
└── README.md             
```


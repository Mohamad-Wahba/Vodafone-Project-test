# Vodafone-Chatbot with open-source models

## Project Overview

![visual app](https://github.com/user-attachments/assets/9ddf0705-98d0-449d-87f4-0f62a7971412)

Vodafone Chatbot is a managerial-level simulation designed to streamline Vodafone database interactions using Large Language Models (LLMs) and LangChain. This project leverages MySQL for data storage and retrieval and utilizes LLMs to generate SQL queries based on user questions. The resulting data is then processed and, if it represents a time series, visualized using graphs to facilitate easy interpretation.

**Authors:** Mohamed Bassiony, Mohamad Wahba, Beshoy Ashraf Samir, Mohamad Sharqawi  

![Project pipeline with langchain and LLM](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS-0YaK8SHX-0w-lvZFDNCnegesg0tYdBMOgw&s)



## Download Test Database chinook
First of all, download the [Chinook database](https://github.com/lerocha/chinook-database.git). This is a sample database that represents a digital media store, including tables for artists, albums, media tracks, invoices, and customers. We will use this database to test our chatbot.


## Install MySQL

1) Download MySQL From [Ofiicial site](https://dev.mysql.com/downloads/installer/)
2) Also you cloud download MySQL Workbench From [here](https://dev.mysql.com/downloads/workbench/)


## Connect chinook Database

1) Go to your terminal and log in to MySQL using the following command:
```bash
$ mysql -u root -p
```

2) You will be prompted to enter your password. Once you are logged in, you can create a new database using the following commands:
```bash
$ CREATE DATABASE chinook;
$ USE chinook;
$ SOURCE chinook.sql;
```


## Requirements

- Python 3.8 or later 

### Install Python using MiniConda

1) Download and install MiniConda From [here](https://docs.anaconda.com/free/miniconda/#quick-command-line-install)
2) Create a new environment using the following command:
```bash
$ conda create -n vodafone-chatbot python=3.10
```

Another way to create a new environment is by using the following command:
```bash
$ conda create --name vodafone-chatbot python=3.10
```

3) Activate the environment:
```bash
$ conda activate vodafone-chatbot
```

### (Optional) work on powershell environment

1) Install Virtualenv
```bash
$ pip install virtualenv
```

2) Create a new environment using the following command:
```bash
$ python -m venv vodafone-chatbot
```

3) Activate the environment:
```bash
$ .\vodafone-chatbot\Scripts\Activate.ps1
```

### (Optional) Setup your command line for better readability if you using WSL
```bash
export PS1=export PS1="\[\033[01;32m\]\u@\h:\w\n\[\033[00m\]\$ "
```

## Installation

### Install the required packages in src directory

```bash
$ pip install -r requirements.txt
```

### Setup the environment variables

```bash
$ cp .env.example .env
```

Set your environment variables in the `.env` file. Like `API_KEYS` value.


## Open-source models

Create a private key for free and choose suitable open-source model as API from [here](https://console.groq.com/docs/models)

## Run app.py in src directory using streamlit

```bash
$ streamlit run app.py
```

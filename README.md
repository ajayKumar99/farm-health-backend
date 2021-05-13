# Major Project 2 Final Application
This is a web application with backend written in flask(python) and frontend written in React(Javascript).
To run the application without any errors, you will have to setup both backend and frontend and run them parallely. Below is how you can do it.

## Requirements
* Python should be installed in your system
* Latest version of node should be installed with npm
* Windows Terminal / Powershell / Any terminal application

## Setting up the backend server
* Open a terminal and cd into farm-health-backend folder
  
  ```
  cd farm-health-backend
  ```
  
* Setup a virtual environment for python using the below command
  
  ```
  python -m venv env
  ```
  
* Activate the virtual environment by using below command
  
  ### For Windows

  ```
  env/Scripts/Activate
  ```
  
  ### For Linux/MacOS
  
  ```
  source env/bin/activate
  ```
  
* Install all the requirements from requirements.txt
  
  ```
  pip install -r requirements.txt
  ```
  
* Start the backend server

  ```
  python app.py
  ```
  
## Setting up the frontend application
* Open a terminal and cd into farm-health-frontend folder
  
  ```
  cd farm-health-frontend
  ```
  
* Install all node dependencies

  ```
  npm install
  ```
  
* Start the react application

  ```
  npm start
  ```
  
* You can start using the application at http://localhost:3000/.

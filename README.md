# Final Project — Bank Term Deposit Prediction (DevOps / MLOps)

we train a model , it is a prediction about if a client would agree to describe the fixed deposit after calling this client. 
This project use git ,docker ,CICD ,mlflow ,dvc


1. Data
Input Data:
age,professions,marriage,debt,education level,default record
contact info,when contacting,how many times to contact
last time activity,previous contact times,last result
jobs change rate,consumer confidence index,Euribor rate,hired num

2.directory
    .dvc
    .github
        workflows
            cd.yml
            ci.yml
├── app/                  # FastAPI service 
        main.py
        model_loader.py
        schemas.py
├── ml/
│   ├── train.py          # training script (calls MLflow)
│   ├── data_pipeline.py  # preprocessing / feature engineering (shared by train & serve)
│   ├── make_v2_data.py   # builds dataset v2 out of v1
│   ├── registry/         # current_model.md + model.pkl (pkl not in git)
│   ├── README_data.md    # dataset versions (v1 / v2)
│   ├── README_experiments.md
│   └── PROVENANCE.md
├── data/                 # DVC-tracked CSVs, git only has the .dvc pointers
├── tests/                # pytest: input validation / happy path / edge cases
├── requirements.txt      # single source of truth for deps
├── Dockerfile / .dockerignore
├── .github/workflows/    # ci.yml + cd.yml
├── DEPLOYMENT.md
└── README.md


4. CI/CD

CI-Branch(pr/push): feature->dev->staging->main
CD-Branch(push): dev->staging and staging->main

render staging `.env` from GitHub Secrets -> build staging image
render prod `.env` from GitHub Secrets -> build prod image

5. MLOps

To dvc:csv,pkl
To git:.dvc,code,.yml
remote dvc registry :Azure Blob (`myazure` -> `azure://dvc`, account `wut2026`), both dataset versions are already pushed.
Two versions data: 
**v1** (raw download) 
**v2** (education merged / outlier rows dropped / dupes removed, built by [ml/make_v2_data.py](ml/make_v2_data.py)).


Experiment tracking (MLflow)

The training script logs automatically: git commit, dataset version/md5 (tags), all hyperparams (params),
  accuracy / F1 / ROC-AUC (metrics), confusion matrix PNGs and the final model.pkl (artifacts).
4 runs so far: 2 models x 2 dataset versions. Locally self-hosted: metadata `sqlite:///mlflow.db`, artifacts in `mlruns/`; just run `mlflow ui`.
Conclusions and prod-model choice: [ml/README_experiments.md](ml/README_experiments.md).

6. PyTest

47 pytest cases 
Unit test:
test_schema: missing fields / wrong types / out-of-range / blank categories / extra fields -> 422; boundary values pass;
test_score_semantics: contract shape, valid categories, score range, inference determinism, OpenAPI;

Integration test:
test_edge_cases: unknown categories don't crash, unicode / extreme numerics, inference failure -> 500, fallback placeholder model.
test_predict: POST /predict, valid input -> 200, invalid input -> 422, fallback placeholder model -> 500.


7.How to test all functions
<!-- to see dvc remote list -->
dvc remote list   

<!-- see local dvc file-->
Get-Content data\bank-additional-full.csv.dvc  
Get-Content data\bank-additional-v2.csv.dvc

<!-- see local md5 file -->
Get-ChildItem .dvc\cache\files\md5 -Recurse -Name

<!-- pull data from remote -->
dvc pull data\bank-additional-full.csv.dvc
dvc pull data\bank-additional-v2.csv.dvc


<!-- push data to remote -->
dvc push

<!-- dvc status -->
dvc status

<!-- check mlflow ui ， http://127.0.0.1:5000，start a new powershell!-->
mlflow ui --backend-store-uri sqlite:///mlflow.db

<!-- lint -->
ruff check .

<!-- format -->
ruff format .
ruff format --check .

<!-- run pytest cases -->
pytest -v   

<!--FastAPI http://127.0.0.1:8000/docs，start a new powershell!-->
uvicorn app.main:app --port 8000

Invoke-RestMethod http://127.0.0.1:8000/health

$body = @{
  age = 35; job = "admin."; marital = "married"; education = "university.degree"
  default = "no"; housing = "yes"; loan = "no"; contact = "cellular"
  month = "may"; day_of_week = "mon"; campaign = 2; pdays = 999
  previous = 0; poutcome = "nonexistent"; emp_var_rate = -1.8
  cons_price_idx = 92.893; cons_conf_idx = -46.2; euribor3m = 1.3; nr_employed = 5099.1
} | ConvertTo-Json


Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -Body $jsonData -ContentType "application/json"

<!--Docker -->
docker build -t bank-marketing-api:local .

docker images bank-marketing-api

docker run -d --name bank-api -p 8001:8000 bank-marketing-api:local

Start-Sleep -Seconds 30

docker inspect --format "{{.State.Health.Status}}" bank-api

<!--E2E -->
Invoke-RestMethod http://127.0.0.1:8001/health
Invoke-RestMethod -Uri http://127.0.0.1:8001/predict -Method Post -Body $body -ContentType "application/json"

docker rm -f bank-api


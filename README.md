# Indian Crop Yield Predictor

Local-first Streamlit app with Dev/Test/Prod via Docker + Minikube and CI/CD via GitHub Actions.

## Quickstart

- Create venvs (optional): python -m venv .venv-dev && .\.venv-dev\Scripts\Activate.ps1 && pip install -r requirements.txt && streamlit run app/app.py
- Minikube (PowerShell): & minikube -p minikube docker-env --shell powershell | Invoke-Expression
- Build image: docker build -t indian-crop-yield:dev .
- Tag: docker tag indian-crop-yield:dev indian-crop-yield:test && docker tag indian-crop-yield:dev indian-crop-yield:prod
- Deploy dev: kubectl apply -f k8s/dev/namespace.yaml && kubectl apply -f k8s/dev/deployment.yaml && kubectl apply -f k8s/dev/service.yaml
- Open: minikube service -n crop-dev crop-yield-svc

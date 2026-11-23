# FastAPI Service, Tests and Git

## Setup (PowerShell)

```powershell
cd "C:\Users\Admin\Desktop\PYTHONTRAIN\assessment\FastAPI Service, Tests and Git"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## Run the API

```powershell
python -m uvicorn src.main:app --reload
```

Open <http://127.0.0.1:8000/docs>.

## Run tests and coverage

```powershell
pytest --cov=src --cov-report=term-missing
```

The test suite includes health, employee list, employee lookup success and 404,
successful creation, invalid salary, and empty-name validation.

## Install and run pre-commit

```powershell
pre-commit install
pre-commit run --all-files
```

## Git branch, commits, push, and pull request

Run these from the project folder after configuring your Git remote:

```powershell
git init
git switch -c feature/employee-service
git add pyproject.toml .gitignore .env.example README.md
git commit -m "chore: initialize employee service project"
git add src config docs scripts
git commit -m "feat: add FastAPI employee endpoints"
git add tests
git commit -m "test: cover employee API and validation"
git add .pre-commit-config.yaml
git commit -m "chore: configure black and ruff pre-commit hooks"
pre-commit run --all-files
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin feature/employee-service
```

Then open GitHub, select the pushed branch, and choose **Compare & pull request**.
Replace the remote URL placeholder with your actual repository URL. If this project
is being added to an existing repository, do not run `git init` or replace its remote;
create the branch from the existing repository instead.

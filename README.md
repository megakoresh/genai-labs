# GenAI labs

Repository contains experimental code to study generative models

## Dependencies

Python 3.12 and up required

Install all GPU drivers first
Add virtualenv

```
python -m venv .venv
```

Install cuda SDK corresponding to cuda version in requirements file
Activate virtual env and install pytorch

```
pip install -r requirements-pytorch
```

To update cuda version, update requirements, wipe and re-install virtual env.

Install other dependencies

```
pip install -r requirements-other.txt
```

Create `data` folda

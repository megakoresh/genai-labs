# Custom model building and fine-tuning experiments

Custom model code based on "Build a Large Language Model (From Scratch)" book by Sebastian Raschka

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

## Running

Most files can be run to invoke or test local functionality. All book chapter code should be runnable. Run as modules

```
python -m book_chapters_code.chapter7_finetuning_instruction --url https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json
```

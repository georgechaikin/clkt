# clkt
Continual Learning with knowledge tracing and sequential data

## Installation
Requirements:
- Python (>=3.10)
- Poetry
```bash
git clone https://github.com/georgechaikin/clkt.git
cd clkt
poetry install
```
## Examples:
* ```run_cl```: Runs experiments using SAKT model, 2009-2010 ASSISTment data and some CL strategies.
```bash
poetry run run_cl --data-path path/to/assist2009/skill_builder_data.csv
```
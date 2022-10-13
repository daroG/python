# Developer information

This file contains developer information for the repository.

## Python

Python part of the code shall follow those base guidelines:
* is compatible with Python version 3.10,
* all code shall pass configured static code checks,
* all defined test shall pass,
* all code shall be covered by tests.

### Library dependencies

Project is using [Poetry](https://python-poetry.org/) to manage dependencies and building package.

To install Poetry on your system follow the guide here: [Poetry Installation](https://python-poetry.org/docs/#installation)

If you want to have virtualenv created in `.venv` folder in repository root, configure Poetry with following command:
```
poetry config virtualenvs.in-project true
```

To create a venv and install all dependencies:
```
poetry install
```

To add a production dependency:
```
poetry add <dependency name>
```

To add a development dependency (like one used by tests):
```
poetry add -D <dependency name>
```

To build new package:
```
poetry build
```

### Tests

Tests shall be written using pytest and put in the `tests` package. For dependencies handling see [Dependencies section](#dependencies).

To run all the tests use the `pytest` command without any parameters in the main folder of the repo. This will start all the tests from the `tests` package and count coverage for every python file in the project package.

Results of tests will be written to standard output. Coverage can be found in the `.coverage` file and the HTML version of the report will be put in the `coverage-report` directory.

Pytest configuration is in the `pyproject.toml` file.

### Static code checkers

Static checkers are being run on the repository by the PR checker and therefore they shall be executed before submitting PR.

#### Linter

`flake8` with the following plugins:

* `flake8-isort` - checks imports order grouping;
* `darglint` - checks content of docstrings (arguments, returns, etc.);
* `flake8-pydocstyle` - checks presence of docstrings in required places.

To lint all python code use the `flake8` command without any parameters in the main folder of the repo.

Full linting report in HTML form will be available in `flake8-report` directory. Summary will be written to standard output.

`flake8` and `darglint` configuration is in the `setup.cfg` file.

`isort` and `pydocstyle` configuration is in the `pyproject.toml` file.

#### Typing checks

`mypy` is the plugin of choice for typing checks on this repo.

To check all python code use the `mypy` command without any parameters in the main folder of the repo.

`mypy` configuration is in the `pyproject.toml` file.

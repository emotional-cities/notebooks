# notebooks

Data analysis examples for the eMOTIONAL Cities project. Data used in the notebooks has been made publicly available in Amazon Simple Storage Service (S3) buckets.

More information about sample data sharing can be found in the [eMOTIONAL Cities data-share repository](https://github.com/emotional-cities/data-share).

## Compatible Environments

[Visual Studio Code](https://code.visualstudio.com/): All notebooks have been tested in Visual Studio Code on a Windows platform. Tests in other platforms and environments are forthcoming and will be added here.

## How to build

1. Open project folder in VS Code
2. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) (Python 3.9)
3. Install [Python Extension for VS Code](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
4. Create environment from VS Code:
  - `Ctrl+Shift+P` > `Create Environment`
  - Select `.conda` environment
5. Make sure correct environment is selected in the notebook

The current notebook requires Python 3.9+ to run successfully. The file `environment.yml` contains the list of minimal package dependencies required.

## How to export a dataset to OGC API records

Before trying to export datasets, make sure you can run the notebooks on an example dataset to validate all dependencies and required environment configuration is valid.

1. Open a python activated command line in the folder `src/ingestion`.
2. Run the `export.py` module:

```
python -m export <DATA_ROOT_PATH> --contacts contacts.json
```

The `contacts.json` file provides metadata about institutional contacts that should be attached to the data export. An example file is provided below:

```json
{
    "contacts": [
        {
            "name": "FirstName LastName",
            "institution": "Contoso",
            "email": "name1@example.com"
        },
        {
            "name": "AnotherName AnotherLast",
            "institution": "Contoso",
            "email": "name2@example.com"
        }
    ]
}
```

> [!NOTE]
> If your dataset contains missing UBX synchronization signals you can provide fallback schemas to be used in case of failure, e.g.
> ```
> python -m export <DATA_ROOT_PATH> --contacts contacts.json --schema outdoor missing_sync
> ```
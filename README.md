# notebooks

Data analysis examples and export tool for the eMOTIONAL Cities project. Data used in the notebooks has been made publicly available in Amazon Simple Storage Service (S3) buckets.

More information about sample data sharing can be found in the [eMOTIONAL Cities data-share repository](https://github.com/emotional-cities/data-share).

## Compatible Environments

[Visual Studio Code](https://code.visualstudio.com/): All notebooks have been tested in Visual Studio Code on a Windows platform. Tests in other platforms and environments are forthcoming and will be added here.

## Set-up Instructions

We recommend [uv](https://docs.astral.sh/uv/) for python version, environment, and package dependency management. However, any other tool compatible with the `pyproject.toml` standard should work.

### Prerequisites

1. Install [uv](https://docs.astral.sh/uv/)
2. Install [Python Extension for VS Code](https://marketplace.visualstudio.com/items?itemName=ms-python.python)

### Install from source

```
git clone https://github.com/emotional-cities/notebooks.git
cd notebooks
uv sync
```

Open the notebook and select the `.venv` environment kernel.

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

This repository was developed for the eMOTIONAL CITIES Project, which received funding from European Union’s Horizon 2020 research and innovation programme, under the grant agreement No 945307. The eMOTIONAL CITIES Project is a consortium of 12 partners co-coordinated by IGOT and FMUL, taking place between 2021 and 2025. More information at https://emotionalcities-h2020.eu/
import argparse
import json
import logging
from modules import *
from pluma.schema.outdoor import build_schema as schema_outdoor
from missing_sync import build_schema as schema_missing_sync

supported_schemas = {
    "outdoor": schema_outdoor,
    "missing_sync": schema_missing_sync
}

def to_ogcapi_records(dataset, geodata, city: str, region: str, id: str, contacts: list[Contact] = None):
    return DatasetRecord(dataset, geodata, properties=RecordProperties(
        title=f'{city} Outdoor Walk: {region} Subject {id}',
        description='Outdoor walk data collection',
        license='CC BY-NC 4.0',
        tool='Bonsai',
        keywords=[f'{city}', 'Outdoor', 'Walk', 'Microclimate', 'Biosignals'],
        contacts=contacts or [],
        themes=[]
    ))

def read_contacts(path):
    contacts = []
    with open(path, encoding="utf8") as fc:
        json_contents = json.load(fc)
        for contact_data in json_contents["contacts"]:
            contacts.append(Contact(
                name=contact_data["name"],
                institution=contact_data["institution"],
                email=contact_data["email"]
            ))
    return contacts

if __name__ == "__main__":
    logging.basicConfig(filename="export.log",
                        filemode="a",
                        format='%(asctime)s %(name)s %(levelname)s %(message)s',
                        level=logging.DEBUG)
    parser = argparse.ArgumentParser(description="Exporter of standardized outdoor walks to OGC API records and GeoJSON datasets.")
    parser.add_argument('root', type=str, help="The path to the root folder containing datasets.")
    parser.add_argument('--output', default="", type=str, help="Specify the optional path to the folder containing the exported datasets.")
    parser.add_argument('--contacts', type=str, help="The path to the JSON file specifying the list of contacts to include in the dataset.")
    parser.add_argument('--schema', nargs='*', type=str, help=
                        "Priority list of schemas to use for loading each dataset. If a dataset fails to load with the first schema," +
                        "a new attempt is made with the following schema, etc, until the end of the list of schemas.")
    args = parser.parse_args()
    contacts = None
    if args.contacts is not None:
        try:
            contacts = read_contacts(args.contacts)
        except:
            print(f"Error parsing contacts file: {args.contacts}")
            raise

    schemas = [(name, supported_schemas[name]) for name in args.schema or ["outdoor"]]

    root = Path(args.root)
    output = Path(args.output)
    if not output.exists():
        output.mkdir(parents=True)

    for match in root.glob("**/Streams_32"):
        for (schema_name, schema) in schemas:
            try:
                path = match.parent
                path_attributes = path.name.split('_')
                city = path_attributes[0]
                region = path_attributes[1]
                subject_id = path_attributes[2].split('-')[-1]

                print(f"Loading dataset: {path.parent.name}/{path.name}..." )
                dataset = load_dataset(path, ubx=True, unity=False, calibrate_ubx_to_harp=schema_name != "missing_sync", schema=schema)
                print(f"Dataset: {dataset} loaded successfully, and {'not' if not dataset.has_calibration else 'sucessfully'} calibrated.")
                geodata = dataset.to_geoframe()

                record = to_ogcapi_records(dataset, geodata, city, region, subject_id, contacts=contacts)
                rpath = output.joinpath(record.id)
                export_geoframe_to_geojson(geodata, rpath.with_suffix('.geojson'))
                with open(rpath.with_suffix('.json'), 'w', encoding="utf8") as f:
                    f.write(record.to_json())
                break
            except Exception as error:
                error_msg = f"Error exporting dataset: {path} with schema {schema_name}"
                print(error_msg)
                logging.exception(error_msg)
                continue
    
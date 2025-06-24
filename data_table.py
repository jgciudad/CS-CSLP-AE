import tables

COLUMN_SUBJECT_ID = 'subject_id'
COLUMN_LABEL = 'label'
COLUMN_SPECIES= 'species'
COLUMN_DATASET = 'dataset'

def create_table_description(config: dict):
    """ creates the description for the pytables table used for dataloading """
    n_sample_values = int(128)

    table_description = {
        COLUMN_SUBJECT_ID: tables.StringCol(20),
        COLUMN_LABEL: tables.StringCol(5),
        COLUMN_SPECIES: tables.StringCol(10),
        COLUMN_DATASET: tables.StringCol(20)
    }
    for c in config.CHANNELS:
        if 'EEG' in c:
            table_description[c] = tables.Float32Col(shape=(1, n_sample_values))
        
        table_description[c+"_rms"] = tables.Float32Col()
        
    return table_description

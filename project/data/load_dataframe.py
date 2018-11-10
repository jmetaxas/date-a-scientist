from pathlib import Path
import pandas as pd
from data.transform_data import clean_income, prepare_data
import os.path

pd.set_option('display.max_columns', None)


def load_profiles(income=False):
    # current_dir = 'csv/'
    current_path = os.path.abspath(os.path.dirname(__file__))
    file_name = 'new_profiles_full.csv'
    main_columns = ['age', 'height']

    if income:
        file_name = 'new_profiles_with_income.csv'
        main_columns += ['income']

    path = os.path.join(current_path, "csv/"+file_name)

    my_file = Path(path)
    if my_file.exists():
        print("from "+file_name)
        return pd.read_csv(path)
    else:
        print("from profiles.csv")
        profiles = pd.read_csv(os.path.join(current_path, "csv/profiles.csv"))
        profiles = profiles.dropna(subset=["height"])

        if income:
            profiles = clean_income(profiles)

        profiles, columns = prepare_data(profiles, ['sex', 'age_code', 'orientation', 'ethnicity', 'religion', 'income',
                                                    'career', 'academic_degree', 'body_type', 'diet',
                                                    'drinks_drugs_smokes', 'offspring', 'pets', 'essay'])
        columns = main_columns + columns
        profiles = profiles.loc[:, columns]
        profiles.to_csv(path, index=False)

        return profiles

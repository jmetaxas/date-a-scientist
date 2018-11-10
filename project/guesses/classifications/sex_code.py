import time
from data.load_dataframe import load_profiles
from utils.models_utils import svc, knc, nbc

start_time = time.time()
profiles = load_profiles(income=False)

"""
Sex Code (Classification)
"""
# print(profiles.corrwith(profiles['sex_code']))

current_guess = ['sex_code']
all_features = [
    'height',
    'has_fit_body_type', 'has_chubby_body_type', 'has_thin_body_type', 'has_average_body_type',
    'eats_anything', 'eats_vegetarian', 'eats_vegan',
    'stem_career', 'health_career', 'education_career',
]
mclasses=['0 (male)', '1 (female)']
svc(profiles, all_features, current_guess, vector_kernel='linear', vector_c=4, vector_gamma=8, show_report=True, show_matrix=True, matrix_classes=mclasses)
# knc(profiles, all_features, current_guess, 28, show_report=True, plot_best_k=1)
# nbc(profiles, all_features, current_guess)


print('')
print("%s seconds" % (time.time() - start_time))

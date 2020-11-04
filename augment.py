import pandas as pd 
import matplotlib.pyplot as plt 
from augment_funcs import * 
from snorkel.augmentation import RandomPolicy, MeanFieldPolicy, PandasTFApplier
import warnings 
import argparse 

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

parser = argparse.ArgumentParser()
parser.add_argument(dest='input_x_train', type=str, help="input features")
parser.add_argument(dest='input_y_train', type=str, help="input labels")
parser.add_argument(dest='output_x_train', type=str, help="output augmented features")
parser.add_argument(dest='output_y_train', type=str, help="output augmented labels")
args = parser.parse_args()

tfs = [swap_adjectives, replace_verb_with_synonym, replace_noun_with_synonym, 
        replace_adjective_with_synonym]

random_policy = RandomPolicy(
    len(tfs), sequence_length=2, n_per_original=2, keep_original=True)

X_train = pd.read_csv(args.input_x_train)
y_train = pd.read_csv(args.input_y_train)

print(preview_tfs(X_train,tfs))

mean_field_policy = MeanFieldPolicy(
    len(tfs),
    sequence_length=2,
    n_per_original=2,
    keep_original=True,
    p=[0.25, 0.25, 0.25, 0.25],
)

X_y_combined = pd.concat([X_train, y_train], axis=1)
tf_applier = PandasTFApplier(tfs, mean_field_policy)
aug = tf_applier.apply(X_y_combined)
df_train_aug = aug.drop('category_id', axis=1)
y_train_aug= aug["category_id"]

print(f'Original training set size: {len(X_train)}')
print(f'Augmented training set size {len(df_train_aug)}')

df_train_aug.to_csv(args.output_x_train, index=False)
y_train_aug.to_csv(args.output_y_train, index=False)
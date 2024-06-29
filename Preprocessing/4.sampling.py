from imblearn.over_sampling import SMOTEN
from imblearn.under_sampling import OneSidedSelection, RandomUnderSampler
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

file = "processed/firstPhase/modbusDataset.csv"

df_save = "processed/sampled/modbusDataset.csv"

separator = ','


def oss_undersampling(df):
    x = df.drop(columns=['attack_type'], axis=1)
    y = df['attack_type']

    print("------------------------------------------------------------")
    print("Counts of values before oss undersampling:")
    print(y.value_counts())

    oss = OneSidedSelection(sampling_strategy='majority', n_neighbors=1, n_seeds_S=200, random_state=42)
    x_res, y_res = oss.fit_resample(x, y)

    print("Counts of values after oss undersampling:")
    print(y_res.value_counts())
    print("------------------------------------------------------------")

    return pd.concat([x_res, y_res], axis=1)


def smoten_oversampling(df):
    x = df.drop(columns=['attack_type'], axis=1)
    y = df['attack_type']

    print("------------------------------------------------------------")
    print("Counts of values before smoten oversampling:")
    print(y.value_counts())

    smoten = SMOTEN(sampling_strategy='not majority', random_state=42)
    x_res, y_res = smoten.fit_resample(x, y)

    print("Counts of values after smoten oversampling:")
    print(y_res.value_counts())
    print("------------------------------------------------------------")

    return pd.concat([x_res, y_res], axis=1)


def random_undersampling(df):
    x = df.drop(columns=['data_type'], axis=1)
    y = df['data_type']

    print("------------------------------------------------------------")
    print("Counts of values before random undersampling:")
    print(y.value_counts())

    rund = RandomUnderSampler(sampling_strategy='majority', random_state=42, replacement=False)
    x_res, y_res = rund.fit_resample(x, y)

    print("Counts of values after random undersampling:")
    print(y_res.value_counts())
    print("------------------------------------------------------------")

    return pd.concat([x_res, y_res], axis=1)


def print_uniques(df, operation):
    string = ""
    for column in df.columns:
        string += str(df[column].nunique())
        string += "\n"
        string += str(df[column].isna().values.any())
        string += "\n"
    with open("processed/{}.txt".format(operation), "w") as text_file:
        print(string, file=text_file)


def main():
    df = pd.read_csv(file, sep=separator, header=0, index_col=None)
    df_attack = df[df['attack_type'] != 1]
    df_benign = df[df['attack_type'] == 1]

    print_uniques(df_attack, "beforeAll")

    df_attack = oss_undersampling(df_attack)
    df_attack = oss_undersampling(df_attack)
    df_attack = oss_undersampling(df_attack)
    print_uniques(df_attack, "afterOss")

    df_attack = smoten_oversampling(df_attack)
    print_uniques(df_attack, "afterSmoten")

    df = pd.concat([df_attack, df_benign], ignore_index=True)

    print_uniques(df, "beforeRandom")
    df = random_undersampling(df)
    print_uniques(df, "afterRandom")

    correlation_matrix = df.corr()
    plt.figure(figsize=(50, 50))
    sns.set(font_scale=1)
    sns.heatmap(correlation_matrix, cmap="YlGnBu", annot=True)
    plt.savefig("processed/sampled/corr/modbusDataset3oss.png")
    plt.close()
    print("Saved correlation matrix.")

    df.to_csv(df_save, index=False)
    del df_attack
    del df_benign
    del df


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np

def generate_random_csv(input_csv_path, output_csv_path, num_rows):
    """
    create new csv file with reasonable values for each feature

    Args:
        input_csv_path (str): path to the input csv file
        output_csv_path (str): path to save the generated csv file
        num_rows (int): number of data points to generate
    """
    df = pd.read_csv(input_csv_path)
    df = df.iloc[:, :-1] # drop the last column (target)

    new_data = {}

    for column in df.columns:
        col_min = df[column].min()
        col_max = df[column].max()

        # additional gap to represent possible real world data
        min_span = 0.1 * col_min
        max_span = 0.1 * col_max

        # generate random values within the extended range
        new_data[column] = np.random.uniform(col_min - min_span, col_max + max_span, num_rows)

    new_df = pd.DataFrame(new_data)
    new_df.to_csv(output_csv_path, index=False)

    # print(f"Random CSV file generated and saved to {output_csv_path}")

generate_random_csv("../data/processed/training_boston.csv", "random_data.csv", 200)
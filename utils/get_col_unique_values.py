import pandas as pd

df = pd.read_csv('case_study/mtr/results/MTR_demand_tuple_classified.csv')

col = ['classification']

def get_col_unique_values(df, col):
    """
    Get unique values from a specified column in a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to search.
    col (list): List containing the column name(s) to extract unique values from.

    Returns:
    list: A list of unique values from the specified column.
    """
    data = df[col].values.tolist()
    values = set()
    for row in data:
        if isinstance(row, list):
            values.update(row)
        else:
            values.add(row)
    return list(values)        

if __name__ == "__main__":
    unique_values = get_col_unique_values(df, col)
    print(unique_values)
    print(len(unique_values))
    print(type(unique_values))
    print(unique_values)  # Print first 10 unique values for brevity
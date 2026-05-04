if __name__ == "__main__":
    import argparse
    import pandas as pd
    import numpy as np

    # argument parser to receive the input file path and the output file path
    parser = argparse.ArgumentParser(description="Clean columns in H5 file")
    parser.add_argument("--input", required=True, help="Input H5 file path")
    parser.add_argument("--output", required=True, help="Output H5 file path")
    args = parser.parse_args()

    # Read H5 file and filter a list of hardn0coded columns
    persons_df = pd.read_hdf(args.input, key="persons")
    persons_required_columns = [
        "age",
        "sex",
        "person_sex",
        "edu",
        "student",
        "worker",
        "earning",
        "MAR",
        "relate",
        "race_id",
        "hispanic",
        "household_id",
    ]
    persons_df = persons_df[persons_required_columns]
    # Save the filtered DataFrame to a new H5 file
    persons_df.to_hdf(args.output, key="persons", mode="w")

    # Do the same for the "households" table
    households_df = pd.read_hdf(args.input, key="households")
    households_required_columns = [
        "income",
        "lcm_county_id",
    ]
    households_df = households_df[households_required_columns]

    # Add random values between 1 and 4 to the "job_industry" column and "job_occupation" column
    households_df["job_industry"] = pd.Series(
        np.random.choice([1, 2, 3, 4], size=len(households_df)),
        index=households_df.index,
    )
    households_df["job_occupation"] = pd.Series(
        np.random.choice([1, 2, 3, 4], size=len(households_df)),
        index=households_df.index,
    )
    households_df.to_hdf(args.output, key="households", mode="a")

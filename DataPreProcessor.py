import os
import pandas as pd

def rename_files(base_dir):
    for root, _, files in os.walk(base_dir):
        for file in files:
            file_path = os.path.join(root, file)

            try:
                # Read last line of the file
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if not lines:
                        continue
                    last_line = lines[-1].strip()

                # Determine new filename
                prefix = "1_" if last_line.endswith("#1#") else "0_"
                new_file_name = prefix + file
                new_file_path = os.path.join(root, new_file_name)

                # Rename file if necessary
                if file != new_file_name:
                    os.rename(file_path, new_file_path)
                    print(f"Renamed: {file_path} -> {new_file_path}")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

def load_files_to_dataframe(base_dir):
    df_list = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.startswith("1_") and file.endswith(".pkl"):
                file_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(file_path)
                    df_list.append(df)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return df_list
    # if df_list:
    #     final_df = pd.concat(df_list, ignore_index=True)
    #     print("Dataframe loaded successfully!")
    #     return final_df
    # else:
    #     print("No valid files found.")
    #     return pd.DataFrame()

# Example usage
base_directory = "logs"
#rename_files(base_directory)
df = load_files_to_dataframe(base_directory)

# Display the loaded DataFrame
print(df)

import pandas as pd
from os.path import join as osp
import re
import json
from tqdm import tqdm

import random
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def extract_prefix(data):
    data_list = data.split(" ")
    if len(data_list) > 1:
        return data_list[1].strip()
    else:
        return None


def find_labeled_entities(input_string, labels_dict):
    # Split the input string into words
    words = input_string.split()

    # Initialize the result list
    result = []

    # Iterate through the words
    for i in range(len(words)):
        # Check for single-word labels
        if words[i] in labels_dict:
            result.append([i, i + 1, labels_dict[words[i]]])
        else:
            # Check for multi-word labels
            for j in range(i, len(words)):
                phrase = " ".join(words[i : j + 1])
                if phrase in labels_dict:
                    result.append([i, j + 1, labels_dict[phrase]])
                    break  # Stop extending once a match is found
    return result


def read_file(src_file_path):
    print(f"Reading file : {src_file_path}")
    with open(src_file_path, "r", encoding="utf-8") as f:
        raw_data = f.read()
    data_lines = raw_data.split("\n\n")
    return data_lines


def read_whois_file(src_file_path):
    data_lines = read_file(src_file_path)
    data_lines = [x for x in data_lines if x.startswith("inet6num")]
    return data_lines


def read_label_file(src_file_path):
    print(f"Reading file : {src_file_path}")
    df = pd.read_parquet(src_file_path)
    df = df.reset_index()
    df["changed"] = df["changed"].dt.strftime("%Y%m%d")
    df = df[["prefix", "OrgName", "country", "OrgID", "NetType", "changed"]]
    for col in df.columns:
        df[col] = df[col].str.strip()
    return df


def create_labelled_data(whois_data_lines, df_label, file_path):
    res_list = []
    for i in tqdm(range(len(whois_data_lines))):
        text = whois_data_lines[i]
        text = re.sub(r"\s+", " ", text)

        pfx = extract_prefix(text)
        if pfx is None:
            print(f"No prefix found : {text}")
            continue

        df_curr = df_label[df_label.prefix == pfx]
        if df_curr.empty:
            print(f"No match found for prefix : {pfx}")
            continue

        labels_dict = df_label[df_label.prefix == pfx].to_dict(orient="records")[0]
        labels_dict = {v: k.upper() for k, v in labels_dict.items() if v is not None}
        label = find_labeled_entities(text, labels_dict)
        res_list.append({"text": text.split(" "), "label": label})

    with open(file_path, "w") as f:
        json.dump(res_list, f, indent=4)


def create_train_test_split(input_file_path, output_dir, sample_size=None):
    with open(input_file_path, "r") as f:
        data_list = json.load(f)
    sample_data = random.sample(data_list, sample_size)
    train_data, test_data = train_test_split(
        sample_data, test_size=0.3, random_state=RANDOM_SEED
    )
    test_data, valid_data = train_test_split(
        test_data, test_size=0.5, random_state=RANDOM_SEED
    )
    with open(osp(output_dir, "train.json"), "w") as f:
        json.dump(train_data, f, indent=4)

    with open(osp(output_dir, "valid.json"), "w") as f:
        json.dump(valid_data, f, indent=4)

    with open(osp(output_dir, "test.json"), "w") as f:
        json.dump(test_data, f, indent=4)


if __name__ == "__main__":
    file_name = "2024-06-30_AFRINIC_v6.parquet"
    file_path = osp("whois-data", "Raw", file_name)
    df_label = read_label_file(file_path)

    file_name = "240807_afrinic.db"
    file_path = osp("whois-data", "Raw", file_name)
    whois_data_lines = read_whois_file(file_path)

    file_name = "2024-06-30_AFRINIC_v6.json"
    file_path = osp("whois-data", file_name)
    create_labelled_data(whois_data_lines, df_label, file_path=file_path)

    file_name = "2024-06-30_AFRINIC_v6.json"
    input_file_path = osp("whois-data", file_name)
    output_dir = "data"
    sample_size = 3000
    create_train_test_split(input_file_path, output_dir, sample_size=sample_size)

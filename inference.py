import os
import boto3
import random
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import pickle
from tensorflow.keras.layers import TextVectorization
from tensorflow import keras
import fastavro

session = boto3.Session(profile_name="production")
s3_client = session.client("s3")
bucket_name = "pdte-ocr-conversion-result-prod"


fips = "12003"
tf.get_logger().setLevel("ERROR")


def read_avro_from_local(file_path):
    records = []
    with open(file_path, "rb") as avro_file:
        avro_reader = fastavro.reader(avro_file)
        for record in avro_reader:
            records.append(record)
    return records


def read_s3_file_to_string(bucket_name, object_key):
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        content = response["Body"].read().decode("utf-8")
        return content
    except Exception as e:
        print(f"Error reading file from S3: {e}")
        return None


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    cleaned_text = tf.strings.regex_replace(lowercase, r"[^a-zA-Z0-9\s]", "")
    return cleaned_text


realestate_transactio_model = keras.models.load_model(
    f"model_realestate_transaction_{fips}.keras"
)
mortage_or_deed_model = keras.models.load_model(f"model_mortgage_or_deed_{fips}.keras")

realestate_layer_from_disk = pickle.load(
    open(f"tv_layer_realestate_transaction_{fips}.pkl", "rb")
)
realestate_layer_config = realestate_layer_from_disk["config"]

mortgage_or_deed_layer_from_disk = pickle.load(
    open(f"tv_layer_mortgage_or_deed_{fips}.pkl", "rb")
)
mortgage_or_deed_config = mortgage_or_deed_layer_from_disk["config"]

realestate_vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=realestate_layer_config["max_tokens"],
    output_mode=realestate_layer_config["output_mode"],
    output_sequence_length=realestate_layer_config["output_sequence_length"],
)
realestate_vectorize_layer.set_weights(realestate_layer_from_disk["weights"])

mortgage_or_deed_vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=mortgage_or_deed_config["max_tokens"],
    output_mode=mortgage_or_deed_config["output_mode"],
    output_sequence_length=mortgage_or_deed_config["output_sequence_length"],
)
mortgage_or_deed_vectorize_layer.set_weights(
    mortgage_or_deed_layer_from_disk["weights"]
)


def is_realestate_transaction(input_text):
    cleaned_text = custom_standardization(tf.constant([input_text]))
    vectorized_text = realestate_vectorize_layer(cleaned_text)
    predictions = realestate_transactio_model.predict(vectorized_text, verbose=0)
    score = float(predictions[0])
    return True if score < 0.5 else False, score


def is_mortgage(input_text):
    cleaned_text = custom_standardization(tf.constant([input_text]))
    vectorized_text = mortgage_or_deed_vectorize_layer(cleaned_text)
    predictions = mortage_or_deed_model.predict(vectorized_text, verbose=0)
    score = float(predictions[0])
    return True if score > 0.5 else False, score


def get_document_type(input_text):
    realestate_transaction, realestate_transaction_score = is_realestate_transaction(input_text)
    if realestate_transaction:
        mortgage, mortgage_score = is_mortgage(input_text)
        if mortgage:
            return "M", realestate_transaction_score, mortgage_score
        return "D", realestate_transaction_score, mortgage_score
    else:
        return "U",  realestate_transaction_score, 0


def write_list_to_json_file(file_path, data_list):
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(data_list, json_file)


def read_file_to_string(file_path):
    with open(file_path, "r") as file:
        file_contents = file.read()
    return file_contents


if __name__ == "__main__":
    records = read_avro_from_local(f"{fips}-2023-01-01-document-types.avro")
    total_records=len(records)
    match_count = 0
    for record in records:
        try:
            s3_key = f'{record["imagefilename"].replace(".tif",".txt")}'.lower()
            dataclassstndcode, realestate_transaction_score, mortgage_score = get_document_type(
                read_s3_file_to_string(bucket_name, s3_key)
            )

            if dataclassstndcode != record["dataclassstndcode"]:
                print(
                    f"Error: {record['imagefilename']} expected {record['dataclassstndcode']} but got {dataclassstndcode} with scores {realestate_transaction_score} and {mortgage_score} "
                )
            else:
                match_count += 1
        except Exception as e:
            print(s3_key)
            print(f"Error: {e}")
    print(f"Matched {match_count} out of {total_records} {match_count/total_records}")

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.data import DatasetReader, Vocabulary
from allennlp.common.params import Params
from collections import defaultdict
import pandas as pd

if __name__ == "__main__":
    prediction_col = "edss_19"
    config = Params.from_file('master_path/master/ML4H_MSProject/task1/configs/cnn_encoder_edss19_config.jsonnet')
    serialization_dir = config["train_options"].pop("serialization_dir", None)
    params = config["params"]
    # Make predictions
    predictor = Predictor.from_path(archive_path=serialization_dir+"/model.tar.gz", cuda_device=1, predictor_name="predict.predictors.ms_note_predictor" ,dataset_reader_to_load=params["dataset_reader"])
    print("Predictor set up")
    output_file = "master_path/results/task1/predictions_"+prediction_col+"_allen_cnn.csv"
    embeddings_file = "master_path/results/task1/embeddings_"+prediction_col+"_allen_cnn.csv"
    # Read in training/test/val data and label isntances found in them
    seen_before = {}
    df_test = pd.read_csv(params["test_data_path"])
    for i, row in df_test.iterrows():
        if row["patient_id"] not in seen_before:
            seen_before[row["patient_id"]] = defaultdict(str)
        seen_before[row["patient_id"]][row["visit_date"]] = "test"
    df_test = pd.read_csv(params["train_data_path"])
    for i, row in df_test.iterrows():
        if row["patient_id"] not in seen_before:
            seen_before[row["patient_id"]] = defaultdict(str)
        seen_before[row["patient_id"]][row["visit_date"]] = "train"
    df_test = pd.read_csv(params["validation_data_path"])
    for i, row in df_test.iterrows():
        if row["patient_id"] not in seen_before:
            seen_before[row["patient_id"]] = defaultdict(str)
        seen_before[row["patient_id"]][row["visit_date"]] = "val"
    with open(output_file, 'w') as out_file:
        with open(embeddings_file, 'w') as embed_file:
            out_file.write("patient_id,date,split,"+prediction_col+",predicted_"+prediction_col+"_allen_cnn\n")
            embed_file.write("patient_id,date,split,text,embeddings_"+prediction_col+"_allen_cnn\n")
            print("Train Data")
            df = pd.read_csv(params["train_data_path"])
            for i, row in df.iterrows():
                if i % 1000 == 0:
                    print(i)
                truth = str(row[prediction_col])
                text = "\""+str(row["text"])+"\""
                if row["tokenized_text"] == "[101, 102]":
                    output = {"embeddings": [0.0]*768, "label": "-1.0"}
                    truth = "-1.0"
                    text = "\"\""
                else:
                    output = predictor.predict(row["tokenized_text"], row["patient_id"])
                embeddings = output["embeddings"]
                date = row["visit_date"]
                type_label = "train"
                text = text.replace(",", "")
                out_file.write(str(row["patient_id"])+","+str(date)+","+type_label+","+truth+","+str(output["label"])+"\n")
                embed_file.write(str(row["patient_id"])+","+str(date)+","+type_label+","+text+",\"["+", ".join([str(k) for k in embeddings])+"]\"\n")
            df = pd.read_csv(params["test_data_path"])
            print("Test Data")
            for i, row in df.iterrows():
                if i % 1000 == 0:
                    print(i)
                truth = str(row[prediction_col])
                text = "\""+str(row["text"])+"\""
                if row["tokenized_text"] == "[101, 102]":
                    output = {"embeddings": [0.0]*768, "label": "-1.0"}
                    truth = "-1.0"
                    text = "\"\""
                else:
                    output = predictor.predict(row["tokenized_text"], row["patient_id"])
                embeddings = output["embeddings"]
                date = row["visit_date"]
                type_label = "test"
                text = text.replace(",", "")
                out_file.write(str(row["patient_id"])+","+str(date)+","+type_label+","+truth+","+str(output["label"])+"\n")
                embed_file.write(str(row["patient_id"])+","+str(date)+","+type_label+","+text+",\"["+", ".join([str(k) for k in embeddings])+"]\"\n")
            df = pd.read_csv(params["validation_data_path"])
            print("Val Data")
            for i, row in df.iterrows():
                if i % 1000 == 0:
                    print(i)
                truth = str(row[prediction_col])
                text = "\""+str(row["text"])+"\""
                if row["tokenized_text"] == "[101, 102]":
                    output = {"embeddings": [0.0]*768, "label": "-1.0"}
                    truth = "-1.0"
                    text = "\"\""
                else:
                    output = predictor.predict(row["tokenized_text"], row["patient_id"])
                embeddings = output["embeddings"]
                date = row["visit_date"]
                type_label = "val"
                text = text.replace(",", "")
                out_file.write(str(row["patient_id"])+","+str(date)+","+type_label+","+truth+","+str(output["label"])+"\n")
                embed_file.write(str(row["patient_id"])+","+str(date)+","+type_label+","+text+",\"["+", ".join([str(k) for k in embeddings])+"]\"\n")
            df = pd.read_csv(params["unlabeled_data_path"])
            print("Unlabeled Data")
            for i, row in df.iterrows():
                if i % 1000 == 0:
                    print(i)
                truth = str(row[prediction_col])
                text = "\""+str(row["text"])+"\""
                if row["tokenized_text"] == "[101, 102]":
                    output = {"embeddings": [0.0]*768, "label": "-1.0"}
                    truth = "-1.0"
                    text = "\"\""
                else:
                    output = predictor.predict(row["tokenized_text"], row["patient_id"])
                embeddings = output["embeddings"]
                date = row["visit_date"]
                type_label = "unlabeled"
                if row["patient_id"] in seen_before:
                    if date in seen_before[row["patient_id"]]:
                        continue
                text = text.replace(",", "")
                out_file.write(str(row["patient_id"])+","+str(date)+","+type_label+","+truth+","+str(output["label"])+"\n")
                embed_file.write(str(row["patient_id"])+","+str(date)+","+type_label+","+text+",\"["+", ".join([str(k) for k in embeddings])+"]\"\n")

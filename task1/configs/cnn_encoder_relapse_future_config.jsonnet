local experiment_name = "cnn_future_relapse";
local model_name_or_path = "master_path/models/base_blue_bert_pt";
local hidden_size = 768;
local max_chunk_length = 512;

{	"train_options": { 
        "serialization_dir": "master_pather/results/dev/" + experiment_name,
        "file_friendly_logging": false,
        "recover": false,
        "force": true,
        "node_rank": 0,
        "batch_weight_key": "",
        "dry_run": false,
    },
	"params":{
	"dataset_reader": {
            "type": "data_scripts.datasetreaders.ms_future_relapse_reader",
            // "tokenizer": {
            //     "type": "pretrained_transformer",
            //     "model_name": model_name_or_path,
            // },
            "token_indexers": { 
                "tokens": {
                    "type": "pretrained_transformer",
                    "model_name": model_name_or_path,
                    "namespace": "tokens",
                    "max_length": max_chunk_length,
                }
            },
            // "skip_label_indexing": false,
            // "label_namespace": "edss_labels",
        },
        "train_data_path": "master_path/neurology_notes/processed_data/Final Splits/train_data.csv",
        "validation_data_path": "master_path/neurology_notes/processed_data/Final Splits/val_data.csv",
        "test_data_path": "master_path/neurology_notes/processed_data/Final Splits/test_data.csv",
        "unlabeled_data_path": "master_path/neurology_notes/processed_data/Final Splits/unlabeled_data.csv",
        "model": {
            "type": "models.ms_classifiers.ms_classifier",
            "text_field_embedder": {
                "token_embedders": {
                    "tokens": {
                        "type": "pretrained_transformer",
                        "model_name": model_name_or_path,
                        "max_length": max_chunk_length,
                    }
                }
            },
            "seq2vec_encoder": {
                "type": "cnn",
				"embedding_dim": hidden_size,
				"num_filters": 128,
				"ngram_filter_sizes": [2, 3, 4, 5, 6, 10],
            },
            "feedforward": {
                "input_dim": 768,
                "num_layers": 2,
                "hidden_dims": [500, 250],
                "activations": ["relu","relu"]
            },
            "dropout": 0.1,
            "label_namespace": "future_relapse_labels"
        },
        "data_loader": {
            "batch_sampler": {
                "type": "bucket",
                "batch_size": 5,
                "padding_noise": 0,
                // "sorting_keys": [["tokens", "tokens___token_ids"],],
            },
        },
        "validation_data_loader": {
            "batch_sampler": {
                "type": "bucket",
                "batch_size": 5,
                "padding_noise": 0,
                // "sorting_keys": [["tokens", "tokens___token_ids"],],
            },
        },
        "evaluate_on_test": true,
        "trainer": {
            "optimizer": {
                "type": "huggingface_adamw",
                "lr": 5e-4,
                "weight_decay": 0.01,
                "correct_bias": true
            },
            "learning_rate_scheduler": {
                "type": "reduce_on_plateau",
                "min_lr" : 5e-5
            },
            "patience": 5,
            "validation_metric": "+auc",
            "num_epochs": 50,
            "checkpointer": {
                 "num_serialized_models_to_keep": 1,
                 "keep_serialized_model_every_num_seconds": null,
            },
            "model_save_interval": null,
            "grad_norm": 1.0,
            "no_grad": ["embedder"],
            "grad_clipping": 1.0,
            "summary_interval": 1,
            "histogram_interval": 10,
            "should_log_parameter_statistics": true,
            "should_log_learning_rate": true,
            "log_batch_size_period": 100,
            "moving_average": null,
            "distributed": false,
            "local_rank": 0,
            "cuda_device": 2, 
            "world_size": 1,
            "num_gradient_accumulation_steps": 4,
        }
    }
}
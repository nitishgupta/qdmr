local utils = import 'utils.libsonnet';

local epochs = utils.parse_number(std.extVar("EPOCHS"));
local batch_size = utils.parse_number(std.extVar("BATCH_SIZE"));
local seed = utils.parse_number(std.extVar("SEED"));

local trainfile = std.extVar("TRAIN_FILE");
local devfile = std.extVar("DEV_FILE");

local bert_model = "bert-base-uncased";
local max_length = 128;
local bert_dim = 768;  // uniquely determined by bert_model


{
  "dataset_reader": {
    "type": "qdmr_seq2seq_reader",
    "source_tokenizer": {
      "type": "pretrained_transformer",
      "model_name": bert_model
    },
    "target_tokenizer": {
      "type": "spacy",
      "split_on_spaces": true
    },
    "source_token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": bert_model,
        "max_length": max_length,
      }
    },
    "target_token_indexers": {
      "tokens": {
        "type": "single_id",
        "namespace": "target_tokens"
      }
    }
  },

  "train_data_path": trainfile,
  "validation_data_path": devfile,

  "model": {
    "type": "qdmr_seq2seq",
    "source_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "pretrained_transformer",
          "model_name": bert_model,
          "max_length": max_length,
        }
      }
    },
    "encoder": {
      "type": "lstm",
      "input_size": bert_dim,
      "hidden_size": 100,
      "num_layers": 1,
      "bidirectional": true,
      "dropout": 0.2,
    },
    "max_decoding_steps": 40,
    "target_embedding_dim": 100,
    "target_namespace": "target_tokens",
    "attention": {
      "type": "dot_product"
    },
    "beam_size": 5
  },

  "data_loader": {
    "batch_sampler": {
        "type": "bucket",
        "batch_size": batch_size,
        "padding_noise": 0.0
    }
  },

  "trainer": {
    "checkpointer": {
      "num_serialized_models_to_keep": 1,
    },
    "num_epochs": epochs,
    "patience": 10,
    "cuda_device": 0,
    "grad_clipping": 5.0,
    "validation_metric": "+exact_match",
    // Weight decay from allennlp-models/training_config/syntax/bert_base_srl.jsonnet. E.g. rc/transformer_qa.jsonnet
    "optimizer": {
     "type": "huggingface_adamw",
      "lr": 2e-5,
      "weight_decay": 0.01,
      "parameter_groups": [
        [["bias", "LayerNorm.bias", "LayerNorm\\.weight", "layer_norm\\.weight"], {"weight_decay": 0}]
      ]
    },
  },
  "random_seed": seed,
  "numpy_seed": seed,
  "pytorch_seed": seed
}
local utils = import 'utils.libsonnet';

local epochs = utils.parse_number(std.extVar("EPOCHS"));
local batch_size = utils.parse_number(std.extVar("BATCH_SIZE"));
local seed = utils.parse_number(std.extVar("SEED"));
local emb_size = utils.parse_number(std.extVar("EMB_DIM"));

local trainfile = std.extVar("TRAIN_FILE");
local devfile = std.extVar("DEV_FILE");

{
  "dataset_reader": {
    "type": "qdmr_seq2seq_reader",
    "source_tokenizer": {
      "type": "spacy"
    },
    "target_tokenizer": {
      "type": "spacy",
      "split_on_spaces": true
    },
    "source_token_indexers": {
      "tokens": {
        "type": "single_id",
        "namespace": "source_tokens"
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
          "type": "embedding",
          "embedding_dim": emb_size,
          "trainable": true
        }
      }
    },
    "encoder": {
      "type": "lstm",
      "input_size": emb_size,
      "hidden_size": 100,
      "num_layers": 1,
      "bidirectional": true,
      "dropout": 0.2,
    },
    "max_decoding_steps": 50,
    "target_embedding_dim": 100,
    "target_namespace": "target_tokens",
    "attention": {
      "type": "cosine"
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
    // "patience": 10,
    "cuda_device": 0,
    "grad_clipping": 5.0,
    "validation_metric": "-loss",
    "optimizer": {
      "type": "adam",
      "lr": 1e-4
    }
  },
  "random_seed": seed,
  "numpy_seed": seed,
  "pytorch_seed": seed
}
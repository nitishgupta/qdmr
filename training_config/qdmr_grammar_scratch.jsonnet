local utils = import 'utils.libsonnet';

local epochs = utils.parse_number(std.extVar("EPOCHS"));
local batch_size = utils.parse_number(std.extVar("BATCH_SIZE"));
local seed = utils.parse_number(std.extVar("SEED"));
local emb_size = utils.parse_number(std.extVar("EMB_DIM"));

local trainfile = std.extVar("TRAIN_FILE");
local devfile = std.extVar("DEV_FILE");

{
  "dataset_reader": {
    "type": "qdmr_grammar_reader",
    "source_tokenizer": {
      "type": "spacy"
    },
    "source_token_indexers": {
      "tokens": {
        "type": "single_id",
        "namespace": "source_tokens"
      }
    },
  },

  "train_data_path": trainfile,
  "validation_data_path": devfile,

  "model": {
    "type": "qdmr_grammar_parser",
    "utterance_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "vocab_namespace": "source_tokens",
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
    "action_embedding_dim": 200,
    "decoder_beam_search": {
      "beam_size": 5,
    },
    "input_attention": {
      "type": "dot_product"
    },
    "max_decoding_steps": 40,
    "add_action_bias": true,
    "dropout": 0.25,
  },

  "data_loader": {
    "batch_sampler": {
        "type": "bucket",
        "batch_size": batch_size,
        "sorting_keys": ["tokens"],
    }
  },

  "trainer": {
    "checkpointer": {
      "num_serialized_models_to_keep": 1,
    },
    "num_epochs": epochs,
    //"patience": 10,
    "cuda_device": 0,
    "grad_clipping": 5.0,
    "validation_metric": "-loss",
    "optimizer": {
      "type": "adam",
      "lr": 1e-3
    }
  },
  "random_seed": seed,
  "numpy_seed": seed,
  "pytorch_seed": seed
}
local utils = import 'utils.libsonnet';

local epochs = utils.parse_number(std.extVar("EPOCHS"));
local batch_size = utils.parse_number(std.extVar("BATCH_SIZE"));
local seed = utils.parse_number(std.extVar("SEED"));
local glove_path = std.extVar("GLOVE");
local glove_size = utils.parse_number(std.extVar("GLOVE_EMB_DIM"));
local inorder = utils.boolparser(std.extVar("INORDER"));
local attn_loss = utils.boolparser(std.extVar("ATTNLOSS"));

local trainfile = std.extVar("TRAIN_FILE");
local devfile = std.extVar("DEV_FILE");

{
  "dataset_reader": {
    "type": "qdmr_seq2seq_reader",
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
    },
    "inorder": inorder,
  },

  "train_data_path": trainfile,
  "validation_data_path": devfile,

  "model": {
    "type": "qdmr_seq2seq_coverage",
    "source_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "pretrained_file": glove_path,
          "vocab_namespace": "source_tokens",
          "embedding_dim": glove_size,
          "trainable": false
        }
      }
    },
    "encoder": {
      "type": "lstm",
      "input_size": glove_size,
      "hidden_size": 100,
      "num_layers": 1,
      "bidirectional": true,
      "dropout": 0.2,
    },
    "max_decoding_steps": 50,
    "target_embedding_dim": 100,
    "target_namespace": "target_tokens",

    "use_attention_loss": attn_loss,
    "beam_size": 5
  },

//  "data_loader": {
//    "batch_sampler": {
//        "type": "bucket",
//        "batch_size": batch_size,
//        "padding_noise": 0.0
//    }
//  },

  "data_loader": {
    "batch_sampler": {
      "type": "basic",
      "sampler": {"type": "random"},
      "batch_size": batch_size,
      "drop_last": false,
    },
  },

  "trainer": {
    "checkpointer": {
      "num_serialized_models_to_keep": 1,
    },
    "num_epochs": epochs,
    "patience": 15,
    "cuda_device": 0,
    "grad_clipping": 5.0,
    "validation_metric": "+exact_match",
    "optimizer": {
      "type": "adam",
      "lr": 1e-3
    }
  },
  "random_seed": seed,
  "numpy_seed": seed,
  "pytorch_seed": seed
}
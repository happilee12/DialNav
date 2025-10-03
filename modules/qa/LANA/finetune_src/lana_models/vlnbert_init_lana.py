import torch
from lana_models.tokenization_clip import SimpleTokenizer
from transformers import T5Tokenizer

def get_tokenizer(args):
    from transformers import AutoTokenizer
    # if args.dataset == 'rxr' or args.tokenizer == 'xlm':
    #     cfg_name = 'xlm-roberta-base'
    # else:
    #     cfg_name = 'bert-base-uncased'
        
    if args.use_clip16:
        tokenizer = SimpleTokenizer()
    else:
        cfg_name = 'bert-base-uncased'
        tokenizer = AutoTokenizer.from_pretrained(cfg_name)
    return tokenizer

def get_vlnbert_models(args, config=None):
    
    from transformers import PretrainedConfig
    from lana_models.vilmodel_cmt_lana import NavCMT

    model_class = NavCMT

    model_name_or_path = args.bert_ckpt_file
    new_ckpt_weights = {}
    if model_name_or_path is not None:
        ckpt_weights = torch.load(model_name_or_path, weights_only=True)
        if model_name_or_path.split('.')[-1] != "pt":
            new_ckpt_weights = ckpt_weights
        else:
            for k, v in ckpt_weights.items():
                if k.startswith('module'):
                    new_ckpt_weights[k[7:]] = v
                else:
                    if k.startswith('lm_head'):
                        k = 'bert.' + k
                    # add next_action in weights
                    if k.startswith('next_action'):
                        k = 'bert.' + k
                    if k.startswith('pano_img_stop') or k.startswith('pano_ang_stop'):
                        k = 'bert.' + k
                    new_ckpt_weights[k] = v
    
    # if args.dataset == 'rxr' or args.tokenizer == 'xlm':
    #     cfg_name = 'xlm-roberta-base'
    # else:
    #     cfg_name = 'bert-base-uncased'
    
    # vis_config = PretrainedConfig.from_pretrained(cfg_name)     

    # offline mode ==================================== #
    # cfg_name = open('bert-base-uncased.json')
    # import json
    # cfg = json.load(cfg_name)
    cfg = {
    "architectures": [
      "BertForMaskedLM"
    ],
    "attention_probs_dropout_prob": 0.1,
    "gradient_checkpointing": False,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 768,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "layer_norm_eps": 1e-12,
    "max_position_embeddings": 512,
    "model_type": "bert",
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "pad_token_id": 0,
    "position_embedding_type": "absolute",
    "transformers_version": "4.6.0.dev0",
    "type_vocab_size": 2,
    "use_cache": True,
    "vocab_size": 30522
  }
    vis_config = PretrainedConfig.from_dict(cfg)      # offline mode
    # offline mode ==================================== #

    # if args.dataset == 'rxr' or args.tokenizer == 'xlm':
    #     vis_config.type_vocab_size = 2
    
    vis_config.max_action_steps = 100
    vis_config.image_feat_size = args.image_feat_size
    vis_config.angle_feat_size = args.angle_feat_size
    vis_config.num_l_layers = args.num_l_layers
    vis_config.num_r_layers = 0
    vis_config.num_h_layers = args.num_h_layers
    vis_config.num_x_layers = args.num_x_layers
    vis_config.hist_enc_pano = args.hist_enc_pano
    vis_config.num_h_pano_layers = args.hist_pano_num_layers

    vis_config.fix_lang_embedding = args.fix_lang_embedding
    vis_config.fix_hist_embedding = args.fix_hist_embedding
    vis_config.fix_obs_embedding = args.fix_obs_embedding

    vis_config.update_lang_bert = not args.fix_lang_embedding
    vis_config.output_attentions = True
    vis_config.pred_head_dropout_prob = 0.1     
    
    vis_config.no_lang_ca = args.no_lang_ca
    vis_config.act_pred_token = args.act_pred_token
    vis_config.max_action_steps = 50 
    vis_config.max_action_steps = 100           # NOTE important!

    # =================================================clip
    vis_config.vocab_size = 49410 if args.use_clip16 else 30522     # NOTE for clip16
    vis_config.hidden_size = 768                                    # NOTE for clip16
    vis_config.use_clip16 = args.use_clip16
    # =================================================
    
    vis_config.max_given_len = args.max_given_len
    vis_config.max_instr_len = args.max_instr_len

    visual_model = model_class.from_pretrained(
        pretrained_model_name_or_path=None, 
        config=vis_config, 
        state_dict=new_ckpt_weights)
        
    return visual_model

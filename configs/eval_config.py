model = dict(
    type="OXE_DiffusionPolicy",
    net=dict(
        type="OXE_Diffusion_KL_Sora",
        text_encoder=dict(
            type="clip",
            clip_backbone='ViT-B/32',
        ),
        image_encoder=dict(
            type="VideoAutoencoderKL",
            from_pretrained="/data1/wenyoupeng/pretrained_models/sd2/sd-vae-ft-ema/",
            delete_decoder=True,
        ),
        sora=dict(
            type='STDiT_Transformer',
            patchemb3d_in_channel=8,
            pad_emb3d=True,
            hidden_size=1152,
            from_pretrained=None,
            read_out_num_heads=16,
            patch_size=(1, 4, 4), 
            rgb_shape=[224, 224], 
            action_attn_pos='before',
            action_attn_type=dict(
                type='GatedSelfAttentionBlock',
                num_heads=12,
                ff_mult=4,
                ),
        ),
        state_dim=7,
        hidden_size=1152,
        n_layer=12,
        wo_state=False,
    ),
    num_train_steps=100,
    num_infer_steps=5,
)
data = dict(
    preprocess=dict(
        rgb_static_pad=10,
        rgb_gripper_pad=4,
        rgb_shape=[224, 224],
        rgb_mean=[0.485, 0.456, 0.406],
        rgb_std=[0.229, 0.224, 0.225],
    ),
    dataset=dict(
        dataset_root='./data/calvin_ABC_D',
        action_mode='ee_rel_pose',
        data_interval=3,
    ),
    dataloader=dict(
        workers_per_gpu=1,
        prefetch_factor=1,
    )
)
common = dict(
    sequence_length=4,
    act_dim=7,
    chunk_size=10,
    skip_frame=2,
    ep_len=360,
    num_sequences=1000,
    record_evaluation_video=False,
    test_chunk_size=1,
    exec_action_step=-1,
)
training = dict(
    compile_model=False,
    evaluate_during_training=False,
    num_epochs=20,
    log_steps=100,
    lr_max=0.0002,
    weight_decay=0.0001,
    num_warmup_epochs=1,
    gradient_accumulation_steps=4,
    load_dir='./pretrained_models/vidmanckpt',
    save_path='./pretrained_models/vidmanckpt',
    dtype='fp32',
    load_epoch=19,
    bs_per_gpu=14,
    save_epochs=1,
)

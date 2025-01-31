class Constants:
    # Input dimensions (5 numerical features + 2 gender categories)
    DRNET_INPUT_NODES = 7  # age, hpd_hyp, hpd_hyc, hpd_ast, hpd_dia, gender_F, gender_M
    DRNET_SHARED_NODES = 100
    DRNET_OUTPUT_NODES = 50
    
    # Training parameters
    DRNET_EPOCHS = 100
    DRNET_SS_EPOCHS = 1
    DRNET_LR = 1e-4
    DRNET_LAMBDA = 0.0001
    DRNET_BATCH_SIZE = 128
    ALPHA = 1
    BETA = 1

    # Adversarial VAE architecture
    Adversarial_epochs = 1000
    Adversarial_VAE_LR = 1e-3
    INFO_GAN_G_LR = 1e-4
    INFO_GAN_D_LR = 5e-4
    Adversarial_LAMBDA = 1e-5
    Adversarial_BATCH_SIZE = 128
    VAE_BETA = 1
    INFO_GAN_LAMBDA = 1.2
    INFO_GAN_ALPHA = 1

    # Encoder dimensions
    Encoder_shared_nodes = DRNET_INPUT_NODES * 2
    Encoder_x_nodes = DRNET_INPUT_NODES
    Encoder_t_nodes = 1
    Encoder_yf_nodes = 1
    Encoder_ycf_nodes = 1

    # Decoder dimensions
    Decoder_in_nodes = Encoder_x_nodes + Encoder_t_nodes + Encoder_yf_nodes + Encoder_ycf_nodes
    Decoder_shared_nodes = DRNET_INPUT_NODES * 2
    Decoder_out_nodes = DRNET_INPUT_NODES

    # Info GAN dimensions
    Info_GAN_Gen_in_nodes = 50
    Info_GAN_Gen_shared_nodes = 25
    Info_GAN_Gen_out_nodes = 1

    Info_GAN_Dis_in_nodes = DRNET_INPUT_NODES + 2
    Info_GAN_Dis_shared_nodes = 25
    Info_GAN_Dis_out_nodes = 1

    Info_GAN_Q_in_nodes = 1
    Info_GAN_Q_shared_nodes = Decoder_in_nodes
    Info_GAN_Q_out_nodes = Decoder_in_nodes
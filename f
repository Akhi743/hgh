 def preprocess_dataset_for_training(csv_path):
        data_X = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        # Define features for Lalonde dataset
        x = data_X[:, 2:9]  # age, educ, race, married, nodegree, re74, re75 
        no, dim = x.shape
        
        # Get treatment directly from data (no need to generate it)
        t = data_X[:, 1]  # treatment column
        t = Utils.convert_to_col_vector(t)
        
        # Get outcome (re78)
        y_obs = data_X[:, 9]  # re78 column
        
        # Create potential outcomes based on observed data
        # For treated units (t=1): y_f = observed outcome, y_cf = estimated
        # For control units (t=0): y_f = observed outcome, y_cf = estimated
        
        # Normalize outcomes
        y_obs = (y_obs - np.mean(y_obs)) / np.std(y_obs)

        # First assign factual outcomes (what we actually observed)
        y_f = Utils.convert_to_col_vector(y_obs)
        
        # For counterfactuals, we'll estimate using the mean difference method
        # Similar to the original's approach but using actual treatment assignment
        treated_mean = np.mean(y_obs[t.flatten() == 1])
        control_mean = np.mean(y_obs[t.flatten() == 0])
        effect = treated_mean - control_mean
        
        # Estimate counterfactuals
        y_cf = np.where(t == 1, 
                       y_obs - effect,  # For treated: subtract effect to estimate control outcome
                       y_obs + effect   # For control: add effect to estimate treated outcome
                       ).reshape(-1, 1)
        
        # Define potential outcomes matrix (keeping format consistent with original)
        potential_y = np.column_stack([y_f, y_cf])
        
        print(x.shape)
        print(potential_y.shape)
        print(y_f.shape)
        print(y_cf.shape)
        print(t.shape)
        # print(t[1])
        # print(potential_y[1])
        # print(y_f[1])
        # print(y_cf[1])
        return x, t, y_f, y_cf, no



















        class Adversarial_Manager:
    def __init__(self, encoder_input_nodes=7,  # Changed for Lalonde features
                 encoder_shared_nodes=Constants.Encoder_shared_nodes,
                 encoder_x_out_nodes=Constants.Encoder_x_nodes,
                 encoder_t_out_nodes=Constants.Encoder_t_nodes,
                 encoder_yf_out_nodes=Constants.Encoder_yf_nodes,
                 encoder_ycf_out_nodes=Constants.Encoder_ycf_nodes,
                 decoder_in_nodes=Constants.Decoder_in_nodes,
                 decoder_shared_nodes=Constants.Decoder_shared_nodes,
                 decoder_out_nodes=7,  # Changed for Lalonde features
                 gen_in_nodes=Constants.Info_GAN_Gen_in_nodes,
                 gen_shared_nodes=Constants.Info_GAN_Gen_shared_nodes,
                 gen_out_nodes=Constants.Info_GAN_Gen_out_nodes,
                 dis_in_nodes=9,  # 7 features + 2 outcomes
                 dis_shared_nodes=Constants.Info_GAN_Dis_shared_nodes,
                 dis_out_nodes=Constants.Info_GAN_Dis_out_nodes,
                 Q_in_nodes=Constants.Info_GAN_Q_in_nodes,
                 Q_shared_nodes=Constants.Info_GAN_Q_shared_nodes,
                 Q_out_nodes=Constants.Info_GAN_Q_out_nodes,
                 device=None):
        self.adversarial_vae = Adversarial_VAE(encoder_input_nodes=encoder_input_nodes,
                                             encoder_shared_nodes=encoder_shared_nodes,
                                             encoder_x_out_nodes=encoder_x_out_nodes,
                                             encoder_t_out_nodes=encoder_t_out_nodes,
                                             encoder_yf_out_nodes=encoder_yf_out_nodes,
                                             encoder_ycf_out_nodes=encoder_ycf_out_nodes,
                                             decoder_in_nodes=decoder_in_nodes,
                                             decoder_shared_nodes=decoder_shared_nodes,
                                             decoder_out_nodes=decoder_out_nodes).to(device)

        self.netG = Generator(in_nodes=gen_in_nodes,
                            shared_nodes=gen_shared_nodes,
                            out_nodes=gen_out_nodes).to(device)

        self.netD = Discriminator(in_nodes=dis_in_nodes,
                                shared_nodes=dis_shared_nodes,
                                out_nodes=dis_out_nodes).to(device)

        self.netQ0 = QHead_y0(in_nodes=Q_in_nodes,
                            shared_nodes=Q_shared_nodes,
                            out_nodes=Q_out_nodes).to(device)

        self.netQ1 = QHead_y1(in_nodes=Q_in_nodes,
                            shared_nodes=Q_shared_nodes,
                            out_nodes=Q_out_nodes).to(device)











means_X = np.empty((len(np_test_X), Constants.Encoder_x_nodes))
means_yf = np.empty((len(np_test_X), 1))
means_ycf = np.empty((len(np_test_X), 1))
means_T = np.empty((len(np_test_X), 1))

labels_yf = np.empty((len(np_test_X), 1))
labels_T = np.empty((len(np_test_X), 1))
labels_ycf = np.empty((len(np_test_X), 1))

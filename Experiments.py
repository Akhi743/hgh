from collections import OrderedDict
from datetime import date
import numpy as np
import os

from Adversarial_Manager import Adversarial_Manager
from Constants import Constants
from DR_Net_Manager import DRNet_Manager
from Metrics import Metrics
from Utils import Utils
from dataloader import DataLoader

class Experiments:
    def __init__(self, running_mode):
        self.dL = DataLoader()
        self.running_mode = running_mode
        self.np_train = None
        self.np_test = None

    def run_all_experiments(self, iterations):
        split_size = 0.8
        csv_path = "subset_500_rows.csv"
        device = Utils.get_device()
        print(f"Using device: {device}")

        # Create results directory if it doesn't exist
        os.makedirs("Results", exist_ok=True)

        results_list = []
        run_parameters = self.__get_run_parameters()
        print(f"Writing results to: {run_parameters['summary_file_name']}")

        # Lists to store metrics for each iteration
        PEHE_out_list = []
        ATE_out_list = []
        PEHE_in_list = []
        ATE_in_list = []

        with open(run_parameters["summary_file_name"], "w") as file1:
            for iter_id in range(iterations):
                print("-" * 40)
                print(f"Starting iteration: {iter_id + 1}/{iterations}")
                print("-" * 40)

                # Load and split data
                np_train_X, np_train_T, np_train_yf, np_train_ycf, \
                np_test_X, np_test_T, np_test_yf, np_test_ycf, n_treated, n_total = \
                    self.dL.load_train_test_data_random(csv_path, split_size)

                print("-----------> Training DR-NET Models <-----------")

                # Convert to tensors for training
                tensor_train = Utils.convert_to_tensor(np_train_X, np_train_T, np_train_yf, np_train_ycf)

                # Initialize Adversarial Manager
                adv_manager = Adversarial_Manager(
                    encoder_input_nodes=Constants.DRNET_INPUT_NODES,
                    encoder_shared_nodes=Constants.Encoder_shared_nodes,
                    encoder_x_out_nodes=Constants.Encoder_x_nodes,
                    encoder_t_out_nodes=Constants.Encoder_t_nodes,
                    encoder_yf_out_nodes=Constants.Encoder_yf_nodes,
                    encoder_ycf_out_nodes=Constants.Encoder_ycf_nodes,
                    decoder_in_nodes=Constants.Decoder_in_nodes,
                    decoder_shared_nodes=Constants.Decoder_shared_nodes,
                    decoder_out_nodes=Constants.Decoder_out_nodes,
                    gen_in_nodes=Constants.Info_GAN_Gen_in_nodes,
                    gen_shared_nodes=Constants.Info_GAN_Gen_shared_nodes,
                    gen_out_nodes=Constants.Info_GAN_Gen_out_nodes,
                    dis_in_nodes=Constants.Info_GAN_Dis_in_nodes,
                    dis_shared_nodes=Constants.Info_GAN_Dis_shared_nodes,
                    dis_out_nodes=Constants.Info_GAN_Dis_out_nodes,
                    Q_in_nodes=Constants.Info_GAN_Q_in_nodes,
                    Q_shared_nodes=Constants.Info_GAN_Q_shared_nodes,
                    Q_out_nodes=Constants.Info_GAN_Q_out_nodes,
                    device=device
                )

                # Set training parameters
                _train_parameters = {
                    "epochs": Constants.Adversarial_epochs,
                    "vae_lr": Constants.Adversarial_VAE_LR,
                    "gan_G_lr": Constants.INFO_GAN_G_LR,
                    "gan_D_lr": Constants.INFO_GAN_D_LR,
                    "lambda": Constants.Adversarial_LAMBDA,
                    "batch_size": Constants.Adversarial_BATCH_SIZE,
                    "INFO_GAN_LAMBDA": Constants.INFO_GAN_LAMBDA,
                    "INFO_GAN_ALPHA": Constants.INFO_GAN_ALPHA,
                    "shuffle": True,
                    "VAE_BETA": Constants.VAE_BETA,
                    "train_dataset": tensor_train
                }

                print("Training Adversarial Model...")
                adv_manager.train_adversarial_model(_train_parameters, device)
                np_y_cf = adv_manager.test_adversarial_model({"tensor_dataset": tensor_train}, device)
                print("Adversarial Model Training Complete")

                # Prepare data for DR-NET
                tensor_train_dr = Utils.convert_to_tensor(np_train_X, np_train_T, np_train_yf, np_y_cf)
                tensor_test = Utils.convert_to_tensor(np_test_X, np_test_T, np_test_yf, np_test_ycf)
                
                _dr_train_parameters = {
                    "epochs": Constants.DRNET_EPOCHS,
                    "lr": Constants.DRNET_LR,
                    "lambda": Constants.DRNET_LAMBDA,
                    "batch_size": Constants.DRNET_BATCH_SIZE,
                    "shuffle": True,
                    "ALPHA": Constants.ALPHA,
                    "BETA": Constants.BETA,
                    "train_dataset": tensor_train_dr
                }

                # Initialize and train DR-NET
                drnet_manager = DRNet_Manager(
                    input_nodes=Constants.DRNET_INPUT_NODES,
                    shared_nodes=Constants.DRNET_SHARED_NODES,
                    outcome_nodes=Constants.DRNET_OUTPUT_NODES,
                    device=device
                )

                print("Training DR-NET Model...")
                drnet_manager.train_DR_NET(_dr_train_parameters, device)
                
                # Evaluate on test and train sets
                dr_eval_out = drnet_manager.test_DR_NET({"tensor_dataset": tensor_test}, device)
                dr_eval_in = drnet_manager.test_DR_NET({"tensor_dataset": tensor_train}, device)
                
                # Calculate metrics
                PEHE_out, ATE_out = self.__process_evaluated_metric(
                    dr_eval_out["y1_hat_list"],
                    dr_eval_out["y0_hat_list"],
                    dr_eval_out["y1_true_list"],
                    dr_eval_out["y0_true_list"]
                )
                PEHE_out_list.append(PEHE_out)
                ATE_out_list.append(ATE_out)
                
                PEHE_in, ATE_in = self.__process_evaluated_metric(
                    dr_eval_in["y1_hat_list"],
                    dr_eval_in["y0_hat_list"],
                    dr_eval_in["y1_true_list"],
                    dr_eval_in["y0_true_list"]
                )
                PEHE_in_list.append(PEHE_in)
                ATE_in_list.append(ATE_in)

                # Save iteration results
                result_dict = OrderedDict({
                    "iter_id": iter_id,
                    "PEHE_out": PEHE_out,
                    "ATE_out": ATE_out,
                    "PEHE_in": PEHE_in,
                    "ATE_in": ATE_in
                })
                results_list.append(result_dict)

            # Calculate and print final statistics
            print("\n" + "#" * 21)
            print("-" * 21)
            
            # Calculate means and standard deviations
            mean_PEHE_out = np.mean(PEHE_out_list)
            std_PEHE_out = np.std(PEHE_out_list)
            mean_ATE_out = np.mean(ATE_out_list)
            std_ATE_out = np.std(ATE_out_list)
            mean_PEHE_in = np.mean(PEHE_in_list)
            std_PEHE_in = np.std(PEHE_in_list)
            mean_ATE_in = np.mean(ATE_in_list)
            std_ATE_in = np.std(ATE_in_list)

            # Write final results
            final_results = (
                f"DR_NET, PEHE_out: {mean_PEHE_out}, SD: {std_PEHE_out}\n"
                f"DR_NET, ATE Metric_out: {mean_ATE_out}, SD: {std_ATE_out}\n"
                f"DR_NET, PEHE_in: {mean_PEHE_in}, SD: {std_PEHE_in}\n"
                f"DR_NET, ATE Metric_in: {mean_ATE_in}, SD: {std_ATE_in}\n"
            )
            
            print(final_results)
            with open(run_parameters["summary_file_name"], "a") as f:
                f.write("\n#####################\n")
                f.write("---------------------\n")
                f.write(final_results)

            # Save detailed results to CSV
            Utils.write_to_csv(run_parameters["consolidated_file_path"], results_list)

    def __get_run_parameters(self):
        run_parameters = {}
        if self.running_mode == "original_data":
            run_parameters.update({
                "input_nodes": Constants.DRNET_INPUT_NODES,
                "consolidated_file_path": "Results/social_risk_results.csv",
                "summary_file_name": "Results/social_risk_stats.txt"
            })
        return run_parameters

    def __process_evaluated_metric(self, y1_hat, y0_hat, y1_true, y0_true):
        # Convert to numpy arrays
        y1_true_np = np.array(y1_true)
        y0_true_np = np.array(y0_true)
        y1_hat_np = np.array(y1_hat)
        y0_hat_np = np.array(y0_hat)
        
        # Rescale predictions and true values to original scale
        y1_true_np = self.dL.rescale_outcome(y1_true_np)
        y0_true_np = self.dL.rescale_outcome(y0_true_np)
        y1_hat_np = self.dL.rescale_outcome(y1_hat_np)
        y0_hat_np = self.dL.rescale_outcome(y0_hat_np)

        # Calculate metrics
        PEHE = Metrics.PEHE(y1_true_np, y0_true_np, y1_hat_np, y0_hat_np)
        ATE = Metrics.ATE(y1_true_np, y0_true_np, y1_hat_np, y0_hat_np)
        
        return PEHE, ATE

import torch
from torch.utils.data.dataloader import DataLoader
import torch.utils.data as data_utils

import numpy as np
import numpy.linalg as LA

import sys, time, os
import pickle

sys.path.append("..")
import modAutoEncoder as autoencoder
import modLaSDIUtils as utils


import scipy.io
from scipy.interpolate import interp2d, Rbf
from WLaSDI import WLaSDI

import matplotlib.pyplot as plt

DataDict = {
    "b1scale": ["./datasets/burgers/grfarc2visc0p001-scale.mat", 'grfarc2visc0p001'],
    "b1shift": ["./datasets/burgers/grfarc2visc0p001-shift.mat", 'grfarc2visc0p001'],
}

def num_params(model):
    return sum(p.numel() for p in model.parameters())
from omegaconf import DictConfig, ListConfig
from typing import Union
class wrapper_PyWLaSDI():
    def __init__(
            self,
            config: DictConfig
        ):
        # self.config = config
        pass

    def entrypoint(self):
        self.configure_dataset()
        self.configure_models()
        self.training_loop()
        self.configure_latent_space_dataset()
        self.configure_WSINDy()
        self.validation()
        

    def configure_dataset(self):
        data_path = "datasets/burgers/grfarc2visc0p001-shift.mat"
        data_var = "grfarc2visc0p001"
        # data_path = self.config.file.filestr
        # data_var = self.config.file.dataname

        data = scipy.io.loadmat(data_path)
        self.P = data['params']
        solution_snapshot_orig = data[data_var]
        
        solution_snapshot_orig = solution_snapshot_orig[:500, :, ::2]
        # solution_snapshot = solution_snapshot_orig.reshape(-1, solution_snapshot_orig.shape[-1]).astype('float32')
        solution_snapshot = solution_snapshot_orig.astype('float32')

        # train test set split
    
        test_ind = np.random.permutation(np.arange(solution_snapshot.shape[0]))[:int(0.2*solution_snapshot.shape[0])]
        train_ind = np.setdiff1d(np.arange(solution_snapshot.shape[0]),test_ind)

        # set trainset and testset
        self.trainset = solution_snapshot[train_ind]
        self.testset = solution_snapshot[test_ind]

        self.Ptrain = self.P[train_ind]
        self.Ptest = self.P[test_ind]
        self.nt = 51
        self.nx = 512
        self.t = np.linspace(0, 1, self.nt)
    
    def configure_models(self):

        device = autoencoder.getDevice()
        encoder_class = autoencoder.Encoder
        decoder_class = autoencoder.Decoder
        f_activation = autoencoder.SiLU

        m = 512
        f = 4
        b = 1 # used to be 36
        db = 1 # used to be 12, reduced model
        M2 = b + (m-1)*db
        M1 = 2*m
        mask = utils.create_mask_1d(m,b,db)

        # exp_name = self.config.file.name
        exp_name = "b1scale"
        AE_fname = f'model/AE_git_{exp_name}.tar'
        if os.path.isfile(AE_fname):
            try:
                encoder, decoder = autoencoder.readAEFromFile(encoder_class,decoder_class,f_activation, mask, m, f, M1, M2, device, AE_fname)
                self.skip_training = True
            except:
                print("encoder, decoder loading failed! Now trying to recreate them..." )

        self.encoder, self.decoder = autoencoder.createAE(encoder_class,
                                                decoder_class,
                                                f_activation,
                                                mask,
                                                m, f, M1, M2,
                                                device )
        
        print(f"{num_params(self.encoder)} parameters in encoder")
        print(f"{num_params(self.decoder)} parameters in decoder")

    def training_loop(self):
        if hasattr(self, "skip_training") and self.skip_training:
            return 
        batch_size = 20
        num_epochs = 2
        num_epochs_print = num_epochs//100
        early_stop_patience = num_epochs//10

        # autoencoder filename
        # exp_name = self.config.file.name
        exp_name = "b1scale"
        AE_fname = f'model/AE_git_{exp_name}.tar'
        chkpt_fname = f'checkpoint_{exp_name}.tar'

        # train
        trainset = self.trainset.reshape(-1, self.trainset.shape[-1])
        testset = self.testset.reshape(-1, self.testset.shape[-1])
        autoencoder.trainAE(self.encoder,
                            self.decoder,
                            trainset,
                            testset,
                            batch_size,
                            num_epochs,
                            num_epochs_print,
                            early_stop_patience,
                            AE_fname,
                            chkpt_fname )

    def configure_latent_space_dataset(self):
        trainset = self.trainset.reshape(-1, self.trainset.shape[-1])
        self.latent_space_SS = autoencoder.encodedSnapshots(self.encoder, trainset, self.nt, autoencoder.getDevice())

    def configure_WSINDy(self):
        #Using WSINDy
        degree = 1
        normal = 1
        self.WLaSDI_model = WLaSDI(self.encoder, self.decoder, NN=True, device=autoencoder.getDevice(), Local=True, Coef_interp=False, Coef_interp_method=Rbf, nearest_neigh=4)
        self.WLaSDI_model_coef = self.WLaSDI_model.train_dynamics(self.latent_space_SS, self.Ptrain, self.t, degree = degree, gamma = 0.1, threshold=0, overlap=0.7, L = 30, LS_vis=True)
        print(f"weak latent sindy coefficietns {self.WLaSDI_model_coef}")

    # def validation(self):
    #     for i in range(1):
    #         test_traj = self.testset[i]
    #         P = self.Ptest[i]
    #         self.validate_step(test_traj, P)
    def validation(self):
        """
        Reconstruct every trajectory in the test set,
        stash them in self.predicted_array, and save to disk.
        """
        reconstructions = []                          # <- start empty list

        for test_traj, P in zip(self.testset, self.Ptest):
            FOM_recon = self.validate_step(test_traj, P)
            reconstructions.append(FOM_recon)         # <- accumulate

        # Shape:  (n_test, nt, nx)
        self.predicted_array = np.stack(reconstructions, axis=0)

        scipy.io.savemat("predicted_array.mat",
                        {"predicted_array": self.predicted_array})

        
    def validate_step(self, snapshot_full_FOM, P):
        start = time.time()
        FOM_recon = self.WLaSDI_model.generate_ROM(snapshot_full_FOM[0], P, self.t)
        WLaSDI_time = time.time()-start
        #self.plot_final_position_error(FOM_recon, snapshot_full_FOM)
        #self.plot_relative_error(FOM_recon, snapshot_full_FOM)
        return FOM_recon  

    def plot_final_position_error(self, FOM_recon, snapshot_full_FOM):
        print('Final Position Error: {:.3}%'.format(LA.norm(FOM_recon[-1]-snapshot_full_FOM[-1])/LA.norm(snapshot_full_FOM[-1])*100))
        fig = plt.figure()
        ax = plt.axes()
        fig.suptitle('Reconstruction of FOM', y = 1.05)
        ax.set_title('Relative Error: {:.3}%'.format(LA.norm(FOM_recon[-1]-snapshot_full_FOM[-1])/LA.norm(snapshot_full_FOM[-1])*100))
        ax.plot(snapshot_full_FOM[-1], label = 'FOM')
        ax.plot(FOM_recon[-1],'--', label = 'WLaSDI')
        ax.legend()
        # ax.set_xlim(-3,6)

    def plot_relative_error(self, FOM_recon, snapshot_full_FOM):

        FOM_re = np.empty(self.nt)
        for i in range(self.nt):
            FOM_re[i] = LA.norm(FOM_recon[i]-snapshot_full_FOM[i])/LA.norm(snapshot_full_FOM[i])

        fig = plt.figure()
        fig.suptitle('Relative Error for FOM reconstruction', y = 1.05)
        ax = plt.axes()
        ax.set_title('Max Relative Error: {:.3}%'.format(np.amax(FOM_re)*100))
        ax.plot(self.t, FOM_re*100)
        ax.set_xlabel('Time')
        ax.set_ylabel('Relative Error (%)')

ep = wrapper_PyWLaSDI(None)
ep.entrypoint()
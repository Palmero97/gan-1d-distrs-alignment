"""GAN models in Pytorch for alignment of 1-d data.

The module contains the following:
    - Plotting functions for the alignments.
    - Classes for discriminator, critic and generator.
    - Classes of 3 GAN models: VGAN, WGAN and CycleGAN.
"""

import time
import torch
from torch import nn
from torch import optim

import numpy as np
import matplotlib.pyplot as plt


""" 
Functions for displaying the alignments of the different models.
"""


def plot_alignment_1d(g_data, fake_data, real_data, title=""):
    """ Plot GAN alignment (G:A->B) results.

    A grid with four figures is plotted:
    1. Histogram of a sampling of A.
    2. Histogram of a sampling of B.
    3. Scatterplot to show mapping between A and G(A).
    4. Histogram of a sampling of G(A).


    Args:
        g_data (np.array): A
        fake_data (np.array): G(A)
        real_data (np.array): B
    """

    fig = plt.figure()

    ax1 = fig.add_subplot(2, 2, 1)
    h, hx = np.histogram(g_data, bins=50)
    ax1.hist(hx[:-1], 50, weights=h, color="cornflowerblue")
    ax1.set_title("hist g_data")

    ax2 = fig.add_subplot(2, 2, 2)
    h, hx = np.histogram(real_data, bins=50)
    ax2.hist(hx[:-1], 50, weights=h, color="firebrick")
    ax2.set_title("hist real_data")

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(g_data, fake_data, color="mediumaquamarine")
    ax3.set_title("fake mapping")
    ax3.set_ylabel("fake_data")
    ax3.set_xlabel("g_data")

    ax4 = fig.add_subplot(2, 2, 4)
    h, hx = np.histogram(fake_data, bins=50)
    ax4.hist(hx[:-1], 50, weights=h, color="mediumaquamarine")
    ax4.set_title("hist fake_data")

    fig.tight_layout()

    if title != "":
        plt.title(title)
    plt.show()


def plot_dual_alignment_1d(a_data, b_fake, b_data, a_fake, title=""):
    """ Plot GAN alignments (G:A->B) and (F:B->A) results.


    A grid with 6 figures is plotted:
    1. Histogram of a sampling of A.
    2. Histogram of a sampling of G(A).
    3. Scatterplot to show mapping between A and G(A).
    4. Histogram of a sampling of B.
    5. Histogram of a sampling of F(B).
    6. Scatterplot to show mapping between B and F(B).

    Args:
        a_data (np.array): A
        b_fake (np.array): G(A)
        b_data (np.array): B
        a_fake (np.array): F(B)
    """

    fig = plt.figure()

    ax1 = fig.add_subplot(2, 3, 1)
    h, hx = np.histogram(a_data, bins=50)
    ax1.hist(hx[:-1], 50, weights=h, color="cornflowerblue")
    ax1.set_title("hist a_data")

    ax2 = fig.add_subplot(2, 3, 2)
    h, hx = np.histogram(b_fake, bins=50)
    ax2.hist(hx[:-1], 50, weights=h, color="mediumaquamarine")
    ax2.set_title("hist b_fake")

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.scatter(a_data, b_fake, color="mediumaquamarine")
    ax3.set_title("fake mapping a2b")
    ax3.set_ylabel("b_fake")
    ax3.set_xlabel("a_data")

    ax4 = fig.add_subplot(2, 3, 4)
    h, hx = np.histogram(b_data, bins=50)
    ax4.hist(hx[:-1], 50, weights=h, color="cornflowerblue")
    ax4.set_title("hist b_data")

    ax5 = fig.add_subplot(2, 3, 5)
    h, hx = np.histogram(a_fake, bins=50)
    ax5.hist(hx[:-1], 50, weights=h, color="mediumaquamarine")
    ax5.set_title("hist a_fake")

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.scatter(b_data, a_fake, color="mediumaquamarine")
    ax6.set_title("fake mapping b2a")
    ax6.set_ylabel("a_fake")
    ax6.set_xlabel("b_data")

    fig.tight_layout()

    if title != "":
        plt.title(title)
    plt.show()


"""
Adversarial training agents.
"""


class Discriminator(nn.Module):
    """Discriminator for alignment of 1-d data.

    The network is structured as a simple MLP with one input layer,
    two hidden layers and one output layer.
    """
    def __init__(self):
        super().__init__()

        n_features = 1
        n_out = 1

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.LeakyReLU(0.2),
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(32, 32),
            nn.LeakyReLU(0.2),
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(32, 32),
            nn.LeakyReLU(0.2),
        )
        self.out = nn.Sequential(
            torch.nn.Linear(32, n_out),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x


class Critic(nn.Module):
    """Critic for alignment of 1-d data.

    The network is structured as a simple MLP with one input layer,
    two hidden layers and one output layer.
    """
    def __init__(self):
        super().__init__()

        n_features = 1
        n_out = 1

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.LeakyReLU(0.2),
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(32, 32),
            nn.LeakyReLU(0.2),
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(32, 32),
            nn.LeakyReLU(0.2),

        )
        self.out = nn.Sequential(
            torch.nn.Linear(32, n_out),
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x


class Generator(nn.Module):
    """Generator for alignment of 1-d data.

    The network is structured as a simple MLP with one input layer,
    one hidden layer and one output layer.
    """

    def __init__(self):
        super().__init__()

        n_features = 1
        n_out = 1

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.LeakyReLU(0.2),
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(32, 32),
            nn.LeakyReLU(0.2),
        )

        self.out = nn.Sequential(
            nn.Linear(32, n_out),
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.out(x)
        return x


"""
GAN models.
"""


class VGAN:
    """VGAN model for 1-d data.

    The model is inspired in the implementation proposed in
    http://papers.nips.cc/paper/5423-generative-adversarial-nets.
    """
    def __init__(self, g_data_generator, real_data_generator,
                 epochs=5000, lr=2e-4, use_bce_loss=True):

        # Setting GPU training if available
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Training parameters
        self.epochs = epochs
        self.lr = lr
        self.g_data_generator = lambda x: g_data_generator(x).to(self.device)
        self.real_data_generator = lambda x: real_data_generator(x).to(
            self.device)

        # Creation of the discriminator
        self.discriminator = Discriminator()
        self.d_optim = optim.Adam(self.discriminator.parameters(), lr=lr)
        self.discriminator.to(self.device)

        # Creation of the generator
        self.generator = Generator()
        self.g_optim = optim.Adam(self.generator.parameters(), lr=lr)
        self.generator.to(self.device)

        # Loss function
        if use_bce_loss:
            self.criterion = nn.BCELoss()
        else:
            self.criterion = nn.MSELoss()

        # A list to store the loss evolution along training
        self.g_loss_during_training = []
        self.d_loss_during_training = []

    def _make_ones(self, size):
        return torch.ones(size, 1).to(self.device)

    def _make_zeros(self, size):
        return torch.zeros(size, 1).to(self.device)

    def _train_discriminator(self, real_data, fake_data, ones, zeros):
        self.d_optim.zero_grad()

        pred_real = self.discriminator(real_data)
        d_loss_real = self.criterion(pred_real, ones)
        d_loss_real.backward()

        pred_fake = self.discriminator(fake_data)
        d_loss_fake = self.criterion(pred_fake, zeros)
        d_loss_fake.backward()

        self.d_optim.step()

        return d_loss_real + d_loss_fake

    def _train_generator(self, fake_data, ones):
        self.g_optim.zero_grad()

        pred = self.discriminator(fake_data)
        g_loss = self.criterion(pred, ones)

        g_loss.backward()
        self.g_optim.step()

        return g_loss

    def trainloop(self, n_batch=128, verbose=True):
        ones = self._make_ones(n_batch)
        zeros = self._make_zeros(n_batch)

        # Every epoch sample n_batch values
        for epoch in range(self.epochs):
            # Log variables initialization
            start_time = time.time()
            d_running_loss, g_running_loss = 0., 0.

            ################################################################
            # Train discriminator
            ################################################################

            real_data = self.real_data_generator(n_batch)
            g_data = self.g_data_generator(n_batch)

            # Generate fake data and detach
            # (so gradients are not calculated for generator)
            fake_data = self.generator.forward(g_data).detach()

            d_loss = self._train_discriminator(real_data, fake_data, ones,
                                               zeros)
            d_running_loss += d_loss.item()

            ################################################################
            # Train generator
            ################################################################

            # Generate fake data
            g_data = self.g_data_generator(n_batch)
            fake_data = self.generator.forward(g_data)

            g_loss = self._train_generator(fake_data, ones)
            g_running_loss += g_loss.item()

            self.g_loss_during_training.append(g_running_loss)
            self.d_loss_during_training.append(d_running_loss)

            ################################################################

            # Print generator samples
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                # Set model to evaluation mode
                self.generator.eval()

                if verbose and (epoch % 2000 == 0):
                    real_data = self.real_data_generator(5000)
                    g_data = self.g_data_generator(5000)

                    plot_alignment_1d(
                        g_data.cpu().data,
                        self.generator.forward(g_data).cpu().data,
                        real_data.cpu().data,
                    )

                    print(
                        "Epoch %d. G loss: %f, D loss: %f, Time per epoch: %f seconds"
                        % (epoch,
                           self.g_loss_during_training[-1],
                           self.d_loss_during_training[-1],
                           (time.time() - start_time)))

                    # Set model back to train mode
                self.generator.train()

    # Evaluation function to test the already trained model
    def eval_performance(self, n=5000):
        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            self.generator.eval()

            real_data = self.real_data_generator(n)
            g_data = self.g_data_generator(n)
            fake_data = self.generator.forward(g_data)

        # Set model back to train mode
        self.generator.train()

        return g_data.cpu().data, real_data.cpu().data, fake_data.cpu().data


class WGAN:
    """WGAN model for 1-d data.

    The model is inspired in the implementation proposed in
    https://arxiv.org/abs/1701.07875.
    """
    def __init__(self, g_data_generator, real_data_generator,
                 epochs=5000, lr=2e-4, n_critic=5):

        # Setting GPU training if available
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Training parameters
        self.epochs = epochs
        self.lr = lr
        self.g_data_generator = lambda x: g_data_generator(x).to(self.device)
        self.real_data_generator = lambda x: real_data_generator(x).to(
            self.device)

        # Creation of the critic
        self.critic = Critic()
        self.c_optim = optim.RMSprop(self.critic.parameters(), lr=lr)
        self.critic.to(self.device)
        self.n_critic = n_critic

        # Creation of the generator
        self.generator = Generator()
        self.g_optim = optim.RMSprop(self.generator.parameters(), lr=lr)
        self.generator.to(self.device)

        # A list to store the loss evolution along training
        self.g_loss_during_training = []
        self.c_loss_during_training = []

    def _train_critic(self, real_data, fake_data):
        self.c_optim.zero_grad()

        c_real = self.critic(real_data)
        c_fake = self.critic(fake_data)
        c_loss = -(torch.mean(c_real) - torch.mean(c_fake))

        c_loss.backward()
        self.c_optim.step()

        # Clipping
        for p in self.critic.parameters():
            p.data.clamp_(-0.01, 0.01)

        return c_loss

    def _train_generator(self, fake_data):
        self.g_optim.zero_grad()

        c_fake = self.critic(fake_data)
        g_loss = -torch.mean(c_fake)

        g_loss.backward()
        self.g_optim.step()

        return g_loss

    def trainloop(self, n_batch=128, verbose=True):

        # Every epoch sample n_batch values
        for epoch in range(self.epochs):
            # Log variables initialization
            start_time = time.time()
            c_running_loss, g_running_loss = 0., 0.

            ################################################################
            # Train critic
            ################################################################

            # In WGAN the critic is trained several times
            for _ in range(self.n_critic):
                real_data = self.real_data_generator(n_batch)
                g_data = self.g_data_generator(n_batch)

                # Generate fake data and detach
                # (so gradients are not calculated for generator)
                fake_data = self.generator.forward(g_data).detach()

                c_loss = self._train_critic(real_data, fake_data)
                c_running_loss += c_loss.item()

            ################################################################
            # Train generator
            ################################################################

            # Generate fake data
            g_data = self.g_data_generator(n_batch)
            fake_data = self.generator.forward(g_data)

            g_loss = self._train_generator(fake_data)
            g_running_loss += g_loss.item()

            self.g_loss_during_training.append(g_running_loss)
            self.c_loss_during_training.append(c_running_loss)

            ################################################################

            # Print generator samples
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                # Set model to evaluation mode
                self.generator.eval()

                if verbose and (epoch % 2000 == 0):
                    real_data = self.real_data_generator(5000)
                    g_data = self.g_data_generator(5000)

                    plot_alignment_1d(
                        g_data.cpu().data,
                        self.generator.forward(g_data).cpu().data,
                        real_data.cpu().data,
                    )

                    print(
                        "Epoch %d. G loss: %f, C loss: %f, Time per epoch: %f seconds"
                        % (epoch,
                           self.g_loss_during_training[-1],
                           self.c_loss_during_training[-1],
                           (time.time() - start_time)))

                    # Set model back to train mode
                self.generator.train()

    # Evaluation function to test the already trained model
    def eval_performance(self, n=5000):
        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            self.generator.eval()

            real_data = self.real_data_generator(n)
            g_data = self.g_data_generator(n)
            fake_data = self.generator.forward(g_data)

        # Set model back to train mode
        self.generator.train()

        return g_data.cpu().data, real_data.cpu().data, fake_data.cpu().data


class CycleGAN:
    """CycleGAN model for 1-d data.

    The model is inspired in the implementation proposed in
    http://openaccess.thecvf.com/content_iccv_2017/html/Zhu_Unpaired_
    Image-To-Image_Translation_ICCV_2017_paper.html.
    """
    def __init__(self, a_data_generator, b_data_generator,
                 epochs=5000, lr=2e-4):

        # Setting GPU training if available
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Training parameters
        self.epochs = epochs
        self.lr = lr
        self.a_data_generator = lambda x: a_data_generator(x).to(self.device)
        self.b_data_generator = lambda x: b_data_generator(x).to(self.device)

        ################################################################
        # Discriminators
        ################################################################

        # discriminator A
        self.d_a = Discriminator()
        self.d_a_optim = optim.Adam(self.d_a.parameters(), lr=lr)
        self.d_a.to(self.device)

        # discriminator B
        self.d_b = Discriminator()
        self.d_b_optim = optim.Adam(self.d_b.parameters(), lr=lr)
        self.d_b.to(self.device)

        ################################################################
        # Generators
        ################################################################

        # generator A to B
        self.g_a2b = Generator()
        self.g_a2b_optim = optim.Adam(self.g_a2b.parameters(), lr=lr)
        self.g_a2b.to(self.device)

        # generator B to A
        self.g_b2a = Generator()
        self.g_b2a_optim = optim.Adam(self.g_b2a.parameters(), lr=lr)
        self.g_b2a.to(self.device)

        ################################################################

        # Binary cross entropy loss
        self.criterion_gan = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()

        # Lits to store the loss evolution during training
        self.d_a_loss_during_training = []
        self.d_b_loss_during_training = []
        self.g_a2b_loss_during_training = []
        self.g_b2a_loss_during_training = []

    def _make_ones(self, size):
        return torch.ones(size, 1).to(self.device)

    def _make_zeros(self, size):
        return torch.zeros(size, 1).to(self.device)

    def trainloop(self, n_batch=128, verbose=True):

        # Creating real and fake targets
        target_real = self._make_ones(n_batch)
        target_fake = self._make_zeros(n_batch)

        for epoch in range(self.epochs):
            # Log variables initialization
            start_time = time.time()

            # Sampling the real data
            real_a = self.a_data_generator(n_batch)
            real_b = self.b_data_generator(n_batch)

            ################################################################
            # Train generators A2B and B2A
            ################################################################

            self.g_a2b_optim.zero_grad()
            self.g_b2a_optim.zero_grad()

            # GAN loss
            fake_b = self.g_a2b(real_a)
            pred_fake = self.d_b(fake_b)
            loss_gan_a2b = self.criterion_gan(pred_fake, target_real)

            fake_a = self.g_b2a(real_b)
            pred_fake = self.d_a(fake_a)
            loss_gan_b2a = self.criterion_gan(pred_fake, target_real)

            # Cycle loss
            recovered_a = self.g_b2a(fake_b)
            loss_cycle_aba = self.criterion_cycle(recovered_a, real_a)  # *10.0

            recovered_b = self.g_a2b(fake_a)
            loss_cycle_bab = self.criterion_cycle(recovered_b, real_b)  # *10.0

            # Total loss
            loss_a2b = loss_gan_a2b + loss_cycle_aba
            loss_b2a = loss_gan_b2a + loss_cycle_bab

            loss_a2b.backward()
            loss_b2a.backward()

            self.g_a2b_optim.step()
            self.g_b2a_optim.step()

            ################################################################
            # Train discriminators A and B
            ################################################################

            # Discriminator A
            self.d_a_optim.zero_grad()

            # Real loss
            pred_real = self.d_a(real_a)
            loss_d_real = self.criterion_gan(pred_real, target_real)

            # Fake loss
            pred_fake = self.d_a(fake_a.detach())
            loss_d_fake = self.criterion_gan(pred_fake, target_fake)

            # Total loss
            loss_d_a = (loss_d_real + loss_d_fake) * 0.5
            loss_d_a.backward()

            self.d_a_optim.step()

            # Discriminator B
            self.d_b_optim.zero_grad()

            # Real loss
            pred_real = self.d_b(real_b)
            loss_d_real = self.criterion_gan(pred_real, target_real)

            # Fake loss
            pred_fake = self.d_b(fake_b.detach())
            loss_d_fake = self.criterion_gan(pred_fake, target_fake)

            # Total loss
            loss_d_b = (loss_d_real + loss_d_fake) * 0.5
            loss_d_b.backward()

            self.d_b_optim.step()

            ################################################################

            self.g_a2b_loss_during_training.append(
                loss_gan_a2b + loss_cycle_aba)
            self.g_b2a_loss_during_training.append(
                loss_gan_b2a + loss_cycle_bab)

            self.d_a_loss_during_training.append(loss_d_a)
            self.d_b_loss_during_training.append(loss_d_b)

            # Print generator samples
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                # Set model to evaluation mode
                self.g_a2b.eval()
                self.g_b2a.eval()

                if verbose and (epoch % 2000 == 0):
                    real_a = self.a_data_generator(5000)
                    real_b = self.b_data_generator(5000)

                    plot_dual_alignment_1d(
                        real_a.cpu().data,
                        self.g_a2b(real_a).cpu().data,
                        real_b.cpu().data,
                        self.g_b2a(real_b).cpu().data
                    )

                    print("Epoch %d. GA2B loss: %f, GB2A loss: %f, DA loss: %f,"
                          "DB loss %f, Time per epoch: %f seconds"
                          % (epoch,
                             self.g_a2b_loss_during_training[-1],
                             self.g_b2a_loss_during_training[-1],
                             self.d_a_loss_during_training[-1],
                             self.d_b_loss_during_training[-1],
                             (time.time() - start_time)))

                # Set model back to train mode
                self.g_a2b.train()
                self.g_b2a.train()

    # Evaluation function to test the already trained model
    def eval_performance(self, n=5000, a2b=True):
        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            if a2b:
                self.g_a2b.eval()

                real_data = self.b_data_generator(n)
                g_data = self.a_data_generator(n)
                fake_data = self.g_a2b(g_data)

                self.g_a2b.train()
            else:
                self.g_b2a.eval()

                real_data = self.a_data_generator(n)
                g_data = self.b_data_generator(n)
                fake_data = self.g_b2a(g_data)

                # Set model back to train mode
                self.g_b2a.train()

        return g_data.cpu().data, real_data.cpu().data, fake_data.cpu().data

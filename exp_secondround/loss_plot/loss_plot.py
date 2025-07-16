import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib as mpl
# Path to your matplotlibrc file
config_file_path = 'exp_secondround/cgan/eva_cond/matplotlibrc'

# Load your custom matplotlib rc file
mpl.rc_file(config_file_path)

fcpflow_path = 'exp_secondround/loss_plot/loss_fcpflow.csv'
vae_path = 'exp_secondround/loss_plot/loss_vae.csv'
cgan_d = 'exp_secondround/loss_plot/cgan_d.csv'
cgan_g = 'exp_secondround/loss_plot/cgan_g.csv'
ddpm_path = 'exp_secondround/loss_plot/ddpm.csv'

# Load the data
fcpflow_data = pd.read_csv(fcpflow_path)
vae_data = pd.read_csv(vae_path)
cgan_d_data = pd.read_csv(cgan_d)
cgan_g_data = pd.read_csv(cgan_g)
ddpm_data = pd.read_csv(ddpm_path)

# Keep the  "Step" and second column (loss) from both datasets and only keep "Step"<= 100000
fcpflow_loss = fcpflow_data.iloc[:, [0, 1]]
vae_loss = vae_data.iloc[:, [0, 1]]
cgan_d_loss = cgan_d_data.iloc[:, [0, 1]]
cgan_g_loss = cgan_g_data.iloc[:, [0, 1]]
ddpm_loss = ddpm_data.iloc[:, [0, 1]]

fcpflow_loss = fcpflow_loss[fcpflow_loss['Step'] <= 100000]
vae_loss = vae_loss[vae_loss['Step'] <= 100000]

c_gan_window = 40000
cgan_d_loss = cgan_d_loss[cgan_d_loss['Step'] <= c_gan_window]
cgan_g_loss = cgan_g_loss[cgan_g_loss['Step'] <= c_gan_window]

figure_size = (15, 8)
# plot the fcpflow loss start from step 0 to 100000
plt.figure(figsize=figure_size)
plt.plot(fcpflow_loss['Step'], fcpflow_loss.iloc[:, 1], label='FCPFlow Loss', color='blue', alpha=0.5)
plt.title('FCPFlow Loss Over Steps')
plt.xlabel('Step [-]')
# limit the x-axis to 100000
plt.xlim(0, 100000)
plt.ylabel('Loss [-]')
plt.grid()
plt.legend()
plt.savefig('exp_secondround/loss_plot/fcpflow_loss_plot.png')
plt.close()

# Plot the VAE loss start from step 0 to 100000
plt.figure(figsize=figure_size)
plt.plot(vae_loss['Step'], vae_loss.iloc[:, 1], label='VAE Loss', color='orange', alpha=0.5)
plt.title('VAE Loss Over Steps')
plt.xlabel('Step [-]')
# limit the x-axis to 100000
plt.xlim(0, 100000)
plt.ylabel('Loss [-]')
plt.grid()
plt.legend()
plt.savefig('exp_secondround/loss_plot/vae_loss_plot.png')
plt.close()

# Plot the CGAN discriminator loss start from step 0 to 100000
plt.figure(figsize=figure_size)
plt.plot(cgan_d_loss['Step'], cgan_d_loss.iloc[:, 1], label='WGAN-GP Discriminator Loss', color='green', alpha=0.5)
plt.plot(cgan_g_loss['Step'], cgan_g_loss.iloc[:, 1], label='WGAN-GP Generator Loss', color='red', alpha=0.5)
plt.xlabel('Step [-]')
plt.ylabel('Loss [-]')
plt.title('WGAN-GP Loss Over Steps')
plt.xlim(0, c_gan_window)
plt.grid()
plt.legend()
plt.savefig('exp_secondround/loss_plot/cgan_loss_plot.png')
plt.close()

# Plot the DDPM loss start from step 0 to 100000
plt.figure(figsize=figure_size)
plt.plot(ddpm_data['Step'], ddpm_data.iloc[:, 1], label='DDPM Loss', color='purple', alpha=0.5)
plt.title('DDPM Loss Over Steps')
plt.xlabel('Step [-]')
# limit the x-axis to 100000
plt.xlim(0, 100000)
plt.ylabel('Loss [-]')
plt.grid()
plt.legend()
plt.savefig('exp_secondround/loss_plot/ddpm_loss_plot.png')
plt.close()
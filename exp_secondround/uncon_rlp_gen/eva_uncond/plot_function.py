import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
# Path to your matplotlibrc file
config_file_path = 'exp_secondround/uncon_rlp_gen/eva/matplotlibrc'
import seaborn as sns

# Load your custom matplotlib rc file
mpl.rc_file(config_file_path)


def generate_time_labels(data_shape, gap=1):
    if data_shape == 24:  # hourly data
        return [f'{i:02d}:00' for i in range(0, 24, gap)]
    elif data_shape == 48:  # 30-minute interval data
        return [f'{i//2:02d}:{(i%2)*30:02d}' for i in range(0, 48, gap)]
    elif data_shape == 96:  # 15-minute interval data
        quarters = ['00', '15', '30', '45']
        return [f'{i//4:02d}:{quarters[i%4]}' for i in range(0, 96, gap)]
    else:
        raise ValueError("Unsupported data shape for time labels.")


def plot_consumption(original_data, plot_data, title, interval, path, show_color_bar=True):
    # Calculate the total consumption for each day
    original_data_sum = np.sum(original_data, axis=1)

    # Get the min and max consumption to normalize the colors
    min_consumption = np.min(original_data_sum)
    max_consumption = np.max(original_data_sum)

    # Calculate the total consumption for each day
    plot_consumption_sum = np.sum(plot_data, axis=1)
    
    # Create a new color map
    cmap = plt.get_cmap('coolwarm')

    # Create a figure and a set of subplots
    fig, ax = plt.subplots()
    ax.grid(True)  # Add grid lines to the plot
    
    # Iterate through each day
    for i in range(plot_data.shape[0]):
        # Normalize the total consumption for this day to get a color
        norm_consumption = (plot_consumption_sum[i] - min_consumption) / (max_consumption - min_consumption)
        color = cmap(norm_consumption)
        # Plot the consumption for this day with the calculated color
        ax.plot(plot_data[i], marker='o', color=color, alpha=0.1, markersize=3)
    ax.set_xlim(0, len(plot_data[0])-1)  # Adjust the x-axis limits
    
    if show_color_bar:
        # Create a color bar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_consumption, vmax=max_consumption))
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', label='Daily Consumption [kWh]')
        cbar.set_label('Daily Consumption [kWh]', size=25)
        cbar.ax.tick_params(labelsize=25)    
    
     # Calculate the new y-axis upper limit
    y_max = np.max(original_data)
    y_min = np.min(original_data)
    new_y_max = y_max * 1.05
    new_y_min = y_min

    # Set the y-axis limits
    ax.set_ylim(new_y_min, new_y_max)

    # Time formatting
    gap = original_data.shape[1]*4/24
    gap = int(gap)
    hours = generate_time_labels(original_data.shape[1], gap)
    # hours = [f'{i:02d}:00' for i in range(0, original_data.shape[1],  gap)]
    ax.set_xticks(np.arange(0, original_data.shape[1], gap))
    ax.set_xticklabels(hours, rotation=45, ha='right')
    
    
    # Set the x and y labels
    ax.set_xlabel(f'Hour of the Day [{interval}]', fontsize = 25, labelpad=20)
    ax.set_ylabel('Electricity Consumption [kWh]', fontsize = 25, labelpad=20)
    # ax.xticks(fontsize=25)  # adjust font size as needed
    # Show the plot
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(title, fontsize = 25, pad=20)
    plt.tight_layout()
    plt.savefig(path)
    


def plot_figure(re_data, scaler, con_dim, path='Generated Data.png'):
    orig_data = scaler.inverse_transform(re_data.detach().numpy())
    # print(orig_data[:, -con_dim].mean())
    cmap = plt.get_cmap('RdBu_r')
    fig, ax = plt.subplots()
    for i, condition in zip(orig_data[:,:-con_dim], orig_data[:, -con_dim]):
        # Convert the condition into a color
        color = cmap((condition - orig_data[:, -con_dim].min()) /
                        (orig_data[:, -con_dim].max() - orig_data[:, -con_dim].min()))
        ax.plot(i, color=color)
        
    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(
        vmin=orig_data[:, -con_dim].min(), vmax=orig_data[:, -con_dim].max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Condition Value scaled',
                    rotation=270, labelpad=20)
    # plt.show()
    plt.savefig(path)
    

def plot_consumption_annual(original_data, plot_data, title, interval, show_color_bar=True):
    # Calculate the total consumption for each day
    # original_data_sum = np.sum(original_data, axis=1)

    # Get the min and max consumption to normalize the colors
    min_consumption = np.min(original_data[:,-1])
    max_consumption = np.max(original_data[:,-1])
    
    # Create a new color map
    cmap = plt.get_cmap('coolwarm')

    # Create a figure and a set of subplots
    fig, ax = plt.subplots()
    ax.grid(True)  # Add grid lines to the plot
    
    # Iterate through each day
    for i,condition in zip(plot_data[:,:-2], plot_data[:, -1]):
        # Normalize the total consumption for this day to get a color
        color = cmap((condition - min_consumption) /
                        (max_consumption - min_consumption))

        # Plot the consumption for this day with the calculated color
        ax.plot(i, marker='o', color=color, alpha=0.1, markersize=3)
    ax.set_xlim(0, len(plot_data[0])-3)  # Adjust the x-axis limits
    
    if show_color_bar:
        # Create a color bar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_consumption, vmax=max_consumption))
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', label='Daily Consumption [kWh]')
        cbar.set_label('Annual Consumption [GWh]', size=18)
        cbar.ax.tick_params(labelsize=18)
        
     # Calculate the new y-axis upper limit
    y_max = np.max(original_data[:,:-2])
    new_y_max = y_max * 1.05

    # Set the y-axis limits
    ax.set_ylim(0, new_y_max)

    # Time formatting
    gap = (original_data.shape[1]-2)*4/24
    gap = int(gap)
    hours = generate_time_labels(original_data.shape[1]-2, gap)
    # hours = [f'{i:02d}:00' for i in range(0, original_data.shape[1],  gap)]
    ax.set_xticks(np.arange(0, original_data.shape[1]-2, gap))
    ax.set_xticklabels(hours, rotation=45, ha='right')
    
    
    # Set the x and y labels
    ax.set_xlabel(f'Hour of the Day [{interval}]', fontsize = 18, labelpad=20)
    ax.set_ylabel('Electricity Consumption [kWh]', fontsize = 20, labelpad=20)
    # ax.xticks(fontsize=25)  # adjust font size as needed
    # Show the plot
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(title, fontsize = 20, pad=20)
    plt.tight_layout()
    plt.show()
    
    
def dong_box_plot(data1, data2, data3, labels, name):
    # Combine the data into a list
    data1 = data1.flatten()
    data2 = data2.flatten()
    data3 = data3.flatten()
    # data4 = data4.flatten()
    # data5 = data5.flatten()
    
    data = [data1, data2, data3]

    
    # Create a box plot
    plt.figure(figsize=(7,4))
    plt.boxplot(data)

    plt.xticks([1, 2, 3], labels, rotation=30, ha='right')
    # Add labels and title (optional)
    # plt.xlabel('Models')
    plt.ylabel('Consumption [kWh]')
    # plt.title('Box Plot of Comparison of Consumption' + name, pad = 20)

    # Display the plot
    plt.show()
    
    
def weather_plot(weather_inf, daily_con, x_label):
    plt.figure(figsize=(6, 7))
    sns.scatterplot(x=weather_inf, y=daily_con)
    plt.ylim(daily_con.min(), daily_con.max())
    # plt.xlim(weather_inf.min(), weather_inf.max())
    plt.xlabel(x_label, fontsize = 35)
    plt.ylabel('Daily Consumption [kWh]',  fontsize = 35)
    plt.show()
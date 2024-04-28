import glob
import os
import pickle
import pprint
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeAlias

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def load_data(checkpoint_dir: str = '.') -> tuple[str, dict] | None:
    checkpoints = get_checkpoints(checkpoint_dir)
    if len(checkpoints) > 0:
        return load_checkpoint(checkpoints)


def get_checkpoints(dir: str = '.') -> list[dict[str, Any]]:

    pattern = '*_gen_*.pkl'
    file_paths = glob.glob(os.path.join(dir, pattern))

    checkpoints = []

    pattern = r"^(.*?)_gen_(\d+)\.pkl$"
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        match = re.match(pattern, filename)
        if match:
            file_id = match.group(1)
            gen_number = int(match.group(2))

            checkpoint_dict = {
                'file': file_path,
                'id': file_id,
                'gen': gen_number
            }

            checkpoints.append(checkpoint_dict)

        else:
            print(f"No match found for {filename}.")

    return checkpoints


def load_checkpoint(checkpoints: list[dict[str, Any]], id: str | None = None, gen: int | None = None) -> tuple[str, dict]:

    if id is not None:
        checkpoints = [cp for cp in checkpoints if cp['id'] == id]

    if gen is not None:
        checkpoint = [cp for cp in checkpoints if cp['gen'] == gen][0]
    else:
        # get the latest checkpoint
        checkpoint = sorted(
            checkpoints, key=lambda cp: cp['gen'], reverse=True)[0]

    with open(checkpoint['file'], 'rb') as f:
        checkpoint_dict = pickle.load(f)

    return checkpoint['file'], checkpoint_dict


def plot_dendrogram(population):

    # Compute the linkage matrix using Ward's method (minimize the variance of clusters being merged)
    Z = linkage(population, method='ward')

    # Visualize the dendrogram
    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(Z)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Individual Index')
    plt.ylabel('Distance')
    st.pyplot(fig)


def plot_heatmap(population, gene_labels):

    Z = linkage(population, method='ward')
    order = leaves_list(Z)
    clustered_population = population[order]

    # Create a figure and an axes instance
    fig, ax = plt.subplots(figsize=(20, 10))

    # Use Seaborn to plot the heatmap on the created axes
    sns.heatmap(clustered_population, ax=ax, cmap='vlag', center=0,
                linewidths=.5, cbar_kws={"shrink": .5},
                xticklabels=gene_labels if gene_labels is not None else np.arange(
                    clustered_population.shape[1]),
                )

    # Set titles and labels
    ax.set_title('Heatmap of Clustered Genetic Algorithm Population')
    ax.set_xlabel('Gene Index')
    ax.set_ylabel('Clustered Individual Index')

    # Display the figure using Streamlit
    st.pyplot(fig)


def plot_pca(population, fitnesses, n_components: int | float = 2, n_clusters: int = 3):

    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(population)

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(pca_result)

    cluster_labels = kmeans.predict(pca_result)

    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    # PCA with fitness
    scatter1 = axs[0].scatter(
        pca_result[:, 0], pca_result[:, 1], alpha=0.7, c=fitnesses, cmap='viridis')
    fig.colorbar(scatter1, ax=axs[0], label='Fitness Score')
    axs[0].set_title('PCA Scatter Plot of Chromosomes wtih Fitnesses')
    axs[0].set_xlabel('Principal Component 1')
    axs[0].set_ylabel('Principal Component 2')
    axs[0].grid(True)

    # PCA with clusters
    scatter2 = axs[1].scatter(
        pca_result[:, 0], pca_result[:, 1], alpha=0.7, c=cluster_labels, cmap='viridis')

    cbar = fig.colorbar(scatter2, ticks=np.arange(0, n_clusters))
    cbar.set_label('PCA Cluster')
    # Set tick labels to cluster numbers
    cbar.set_ticklabels(np.arange(n_clusters).astype(str).tolist())

    axs[1].set_title('PCA Scatter Plot of Chromosomes with Clusters')
    axs[1].set_xlabel('Principal Component 1')
    axs[1].set_ylabel('Principal Component 2')
    axs[1].grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    st.pyplot(fig)


def plot_fitness(log):

    if len(log) == 0:
        raise ValueError("Fitness has not yet been calculated.")

    # Convert log data into a pandas DataFrame for easy plotting
    df = pd.DataFrame([{'generation': gen, 'fitness': fitness}
                      for gen, log_item in enumerate(log) for fitness in log_item['fitnesses']])

    # Create a figure and axes for Matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x='generation', y='fitness',
        markers=True, dashes=False,
    )

    ax.set_title('Average Fitness over Generations')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Average Fitness')

    # Display the plot using Streamlit
    st.pyplot(fig)


def create_perturbation_plot(genes, expr_vals):

    df = pd.DataFrame({
        'genes': genes,
        'expr_vals': expr_vals
    })

    # Normalize and map colors
    norm = plt.Normalize(df['expr_vals'].min(),  # type: ignore
                         df['expr_vals'].max())  # type: ignore
    colors = plt.cm.viridis(norm(df['expr_vals']))  # type: ignore

    # Create the plot
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    sns.barplot(x='genes', y='expr_vals', palette=colors, data=df)
    plt.title('Expression Values by Gene with Heatmap Coloring')
    plt.xlabel('Genes')
    plt.ylabel('Expression Values')
    plt.xticks(rotation=90)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)

    return plt


def create_fitness_plot(fitness_values, selected_idx=None):

    # Sorting the fitness values and their indices
    sorted_fitness_idx = np.argsort(fitness_values)
    sorted_fitnesses = fitness_values[sorted_fitness_idx]

    # Apply the Viridis colormap to the sorted values
    norm = plt.Normalize(sorted_fitnesses.min(),  # type: ignore
                         sorted_fitnesses.max())  # type: ignore
    viridis = plt.cm.viridis(norm(sorted_fitnesses))  # type: ignore

    # Highlighting logic
    if selected_idx is not None:
        colors = [viridis[i] if i != sorted_fitness_idx[selected_idx]
                  else 'red' for i in range(len(fitness_values))]
    else:
        colors = viridis

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=[str(i) for i in sorted_fitness_idx],
                y=sorted_fitnesses, palette=colors)
    ax.set_title('Chromosome fitness')
    ax.set_xlabel('Chromosome index')
    ax.set_ylabel('Fitness')
    ax.set_xticklabels([str(i) for i in sorted_fitness_idx])

    return fig


def fitness_distribution(fitnesses):

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=fitnesses)
    ax.set_title('Fitness Distribution')
    ax.set_xlabel('Fitness')
    ax.set_ylabel('Frequency')

    st.pyplot(fig)


def dir_selector(path: str = '.'):

    dirs = [name for name in os.listdir(path)
            if os.path.isdir(os.path.join(path, name))]

    rel_dir = st.selectbox('Select a checkpoint directory', dirs)
    if rel_dir:
        return os.path.join(path, rel_dir)


def main():

    data = None
    data_file = None

    with st.sidebar:

        st.title('🧬 GA Checkpoint Dashbord')

        data_dir = dir_selector(path=ROOT_DIR)
        if data_dir:
            ret = load_data(data_dir)
            if ret is not None:
                data_file, data = ret

        if data is not None and data_file is not None:

            st.info(f'{os.path.basename(data_file)} loaded.')

            if 'run_parameters' in data:
                params = data['run_parameters']

                st.header('Run parameters')

                if 'gene_labels' in params:
                    st.subheader('Genes')
                    st.code((params['gene_labels']))

                if 'start_time' in params:
                    st.subheader('Run start')
                    st.text(
                        (params['start_time'].strftime("%Y-%m-%d %H:%M:%S")))

                if 'stop_time' in params:
                    st.subheader('Run end')
                    st.text(
                        (params['stop_time'].strftime("%Y-%m-%d %H:%M:%S")))

                if all(key in params for key in ['start_time', 'stop_time']):
                    st.subheader('Run duration')
                    st.text((params['stop_time'] - params['start_time']))

                with st.expander('View all parameters'):
                    st.code(pprint.pformat(data['run_parameters']))

            else:
                st.warning('No run parameters found.')

    # color config
    sns.set_theme()

    if data is None:
        st.error('No data found.', icon="🤖")

    else:
        # Plot Fitnesses
        st.header('Population Analysis')
        plot_fitness(data['log'])

        st.divider()

        st.header('Generation Analysis')
        # Select a generation
        latest_gen = len(data['log']) - 1
        if latest_gen > 0:
            selected_gen = st.slider(
                'Select Generation', 0, latest_gen, latest_gen)
        else:
            selected_gen = 0

        # Extract the log for the selected generation
        log = data['log'][selected_gen]

        # Display heatmap for the selected generation
        st.write(f"Heatmap for Generation {selected_gen}")

        # Fetch the selected population data
        gene_labels = data['gene_labels'] if 'gene_labels' in data.keys(
        ) else None
        plot_heatmap(log['population'], gene_labels=gene_labels)

        tab_fitness_dist, tab_dendrogram, tab_pca = st.tabs(
            ["Fitness Histogram", "Dendrogram", "Chromosome PCA"])

        with tab_fitness_dist:
            # Display the distribution of fitnesses in the current generation
            fitness_values = log['fitnesses']
            st.subheader(f"Fitness Distribution for Generation {selected_gen}")
            fitness_distribution(fitness_values)

        with tab_dendrogram:
            # Display the dendrogram clustering of chromosomes
            st.subheader(
                f"Clustered Chromosome Dendrogram for Generation {selected_gen}")
            plot_dendrogram(log['population'])

        with tab_pca:
            # Display the PCA of the chromosomes
            st.subheader(f"Chromosome PCA for Generation {selected_gen}")
            with st.popover("PCA Settings"):
                n_components = st.number_input(
                    'Number of PCA components', value=2, step=1)
                n_clusters = st.number_input(
                    'Number of K-means clusters', value=3, step=1)
            plot_pca(log['population'], log['fitnesses'],
                     int(n_components), int(n_clusters))

        st.divider()

        st.header('Chromosome Analysis')

        if not 'dynamo' in log.keys():
            st.warning('No Dynamo chromosome data found.', icon="🤖")

        else:
            population = log['dynamo']
            chromosome_idx = st.selectbox(
                'Select chromosome', options=np.arange(len(population)))
            chromosome = population[chromosome_idx]

            st.subheader('Genes')
            st.code(chromosome['genes'])

            st.subheader('Perturbation values')
            perturbation_plt = create_perturbation_plot(
                chromosome['genes'], chromosome['expr_vals'])
            st.pyplot(perturbation_plt)  # type: ignore

            with st.popover('View table'):
                st.dataframe(pd.DataFrame({
                    'genes': chromosome['genes'],
                    'expr_vals': chromosome['expr_vals'],
                }))

            st.subheader('Fitness level')
            fitness_plt = create_fitness_plot(
                log['fitnesses'], selected_idx=chromosome_idx)
            st.pyplot(fitness_plt)
            st.caption(
                f'The selected chromosome, **Chromosome {chromosome_idx}**, is :red[highlighted red] in the above graph.')

            with st.popover('View table'):
                st.dataframe(pd.DataFrame({
                    'chromosome': np.arange(len(log['fitnesses'])),
                    'fitness': log['fitnesses'],
                }).set_index('chromosome'))

            st.subheader('Transition Stategraph')
            st.dataframe(pd.DataFrame(chromosome['mtx']))

            st.subheader('Target Score')
            st.code(chromosome['score'])

        # (['genes', 'expr_vals', 'mtx', 'score'])


# ROOT_DIR: str = '/Users/matthewmazurek/Library/CloudStorage/GoogleDrive-matthewmazurek@gmail.com/My Drive/perturbation_screen'
ROOT_DIR: str
if len(sys.argv) > 1:
    ROOT_DIR = sys.argv[1]
else:
    home_directory = os.path.expanduser('~')
    ROOT_DIR = home_directory

if __name__ == '__main__':
    main()

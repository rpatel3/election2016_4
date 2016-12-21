import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import csv
from bs4 import BeautifulSoup
from IPython.display import SVG

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

def map_county_data(clusters, outname, county_svg, colors):
    """Create an SVG map with colors populated
    by a dictionary of (FIPS, cluster) value pairs.
    
    clusters: a dictionary containing the fips id and the cluster label
    outname: The name of the file the map is output to
    county_svg: Base map of empty counties
    colors: The colors for each cluster label
    """
    county_svg = open(county_svg, 'r').read()
    
    soup = BeautifulSoup(county_svg, "html.parser")
    paths = soup.findAll('path')    

    # County style
    path_style = 'font-size:12px;fill-rule:nonzero;stroke:#FFFFFF;stroke-opacity:1; stroke-width:0.1;stroke-miterlimit:4;stroke-dasharray:none;stroke-linecap:butt; marker-start:none;stroke-linejoin:bevel;fill:'    
    
    for p in paths:
        if p['id'] not in ['State_Lines', 'separator']:
            try:
                cluster_class = clusters[int(p['id'])]
            except: 
                continue
            
            color = colors[cluster_class]
            p['style'] = path_style + color
    

    f = open(outname, 'w')
    f.write(soup.prettify())
    f.close()
    
    
def assign_colors(ratios):
    """Assigns colors to each cluster based on the ratio of dem voters to republican votes.
    """
    blue_red = ['#EC0517','#DA092E','#C70E46','#B5135D','#A21774','#901C8B','#7D20A2','#6B25B9','#582AD1','#462EE8','#3333FF']
    ratios = (ratios - ratios.min()) / (ratios.max() - ratios.min()) * (len(blue_red) - 1)
    ratios = dict(ratios.apply(round))
    
    return {k: blue_red[v] for k,v in ratios.items()}    


def compute_ratios(df):
    """Computes the ratio of dem votes to republican voters for each cluster"""
    
    assert 'cluster' in df.columns
    groups = df.groupby('cluster')
    ratios = groups.apply(lambda x: (x.votes_dem / x.votes_rep).mean())
    
    return ratios


def run_pca(data, predictors, num_components=3):

    X = data[predictors]
    X = (X - X.mean()) / X.std()

    pca = PCA()
    pca.fit(X)
    ratios = pca.explained_variance_ratio_

    plt.bar(list(range(len(ratios))), ratios)
    plt.title('Explained Variance Ratios of Each Principal Component')
    plt.xlabel('PC #')
    plt.ylabel('Explained Variance')
    plt.show()
    
    loadings = pd.DataFrame(pca.components_, columns=predictors)
    loadings.index = ['PC_'+str(i) for i in loadings.index]
    loadings = loadings[:num_components]
    loadings_sqr = loadings**2
    sns.set(font_scale=.7)
    sns.heatmap(loadings_sqr.transpose(), linewidths=0.5, cmap="BuGn", 
                 annot=False)
    plt.show()
    
    print('Most important features for each PC:')
    for i in range(len(loadings_sqr)):
        pc_row = loadings_sqr.iloc[i]
        print('PC_'+str(i))
        print(pc_row.nlargest(3))
        print()
    
    comp_df = pd.DataFrame(pca.transform(X), index=X.index)
    comp_df = comp_df.iloc[:,:num_components]
    comp_df.columns = ['PC_' + str(x) for x in comp_df.columns]
    
    return comp_df


def display_cluster_info(data):
    """Displays the voting pattern information for each cluster. 
       Data has a cluster labels column"""
    
    groups = data.groupby('cluster')
    
    print('Cluster Voting Patterns')
    for c, g_df in groups:
        print('Cluster:', c)
        r_per = (g_df.votes_rep / g_df.total_votes).mean()
        print('Mean R %:', r_per)
        
        d_per = (g_df.votes_dem / g_df.total_votes).mean()
        print('Mean d %:', d_per)
        

def get_kmeans_clusters(og_data, pc_df, num_clusters=3):
    """Returns a dictionary of fips->cluster # and a dictionary of 
       cluster # -> color"""
    
    kmeans = KMeans(n_clusters=num_clusters).fit(pc_df)

    data = pd.concat([og_data, pd.Series(kmeans.labels_)], axis=1).rename(columns={0: 'cluster'})
    
    display_cluster_info(data)
    
    ratios = compute_ratios(data)
    colors = assign_colors(ratios)

    # convert into a dict of fips to cluster number
    keys = list(data['fips'])
    values = list(data['cluster'])
    clusters = dict(zip(keys, values))
    
    return clusters, colors
    

def run_clustering_analysis(data, predictors, clust_algo, vis_file, 
                 num_pc=3, num_clusters=3):

    pc_df = run_pca(data, predictors, num_components=num_pc)

    clusters, colors = clust_algo(data, pc_df, num_clusters=num_clusters)

    map_county_data(clusters, vis_file, 'county_map.svg', colors)

#     SVG(vis_file)

import pandas as pd
import plotly.plotly as py

import csv
from bs4 import BeautifulSoup

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix
import numpy as np


def map_county_data(percentage, outname, county_svg, dem = False):
    """Create an SVG map with colors populated
    by a dictionary of (FIPS, rate) value pairs.
    """

    county_svg = open(county_svg, 'r').read()
    county_data = csv.reader(open('processed_data.csv'), delimiter = ",")
    
    soup = BeautifulSoup(county_svg, "html.parser")
    paths = soup.findAll('path')
    red_colors = ['#FF0033', "#FF4066", "#FF8099", '#FFBFCC', '#FFFFFF']
    blue_colors  = ['#3333FF', "#6666FF", "#9999FF", "#CCCCFF", "#FFFFFF"]
    
    path_style = 'font-size:12px;fill-rule:nonzero;stroke:#FFFFFF;stroke-opacity:1; stroke-width:0.1;stroke-miterlimit:4;stroke-dasharray:none;stroke-linecap:butt; marker-start:none;stroke-linejoin:bevel;fill:'    
    
    if dem:
        colors = blue_colors
    else:
        colors = red_colors
    
    for p in paths:
        if p['id'] not in ['State_Lines', 'separator']:
            #print(str(p['id']))
            try:
                rate = percentage[int(p['id'])]
            except: 
                continue
                
            if rate < 0.1:
                color_class = 4
            elif rate < 0.3: 
                color_class = 3
            elif rate < 0.5: 
                color_class = 2
            elif rate < 0.7: 
                color_class = 1
            elif rate < 0.9: 
                color_class = 0
            
            color = colors[color_class]
            p['style'] = path_style + color
    

    f = open(outname, 'w')
    f.write(soup.prettify())
    f.close()
    
def map_county_data_compare(dem_16, rep_16, outname, county_svg):
    """Create an SVG map with colors populated
    by a dictionary of (FIPS, rate) value pairs.
    """

    county_svg = open(county_svg, 'r').read()
    county_data = csv.reader(open('processed_data.csv'), delimiter = ",")
    
    soup = BeautifulSoup(county_svg, "html.parser")
    paths = soup.findAll('path')
    blue_red = ['#FF0000','#EC0517','#DA092E','#C70E46','#B5135D','#A21774','#901C8B','#7D20A2','#6B25B9','#582AD1','#462EE8','#3333FF', '#FFFFFF']
    # County style
    path_style = 'font-size:12px;fill-rule:nonzero;stroke:#FFFFFF;stroke-opacity:1; stroke-width:0.1;stroke-miterlimit:4;stroke-dasharray:none;stroke-linecap:butt; marker-start:none;stroke-linejoin:bevel;fill:'

    for p in paths:
        if p['id'] not in ['State_Lines', 'separator']:
            #print(str(p['id']))
            try:
                rate = rep_16[int(p['id'])]
                rate2 = dem_16[int(p['id'])]
            except: 
                continue
            if rate < 0.1 or rate2 > 0.9:
                color_class = 9
            elif rate < 0.2 or rate2 > 0.8: 
                color_class = 8
            elif rate < 0.3 or rate2 > 0.7: 
                color_class = 7
            elif rate < 0.4 or rate2 > 0.6: 
                color_class = 6
            elif rate < 0.5 or rate2 > 0.5: 
                color_class = 5
            elif rate < 0.6 or rate2 > 0.4: 
                color_class = 4
            elif rate < 0.7 or rate2 > 0.3: 
                color_class = 3
            elif rate < 0.8 or rate2 > 0.2: 
                color_class = 2
            elif rate < 0.9 or rate2 > 0.1: 
                color_class = 0
            elif rate < 1.0: 
                color_class = 1
            else: color_class = 10

            color = blue_red[color_class]
            p['style'] = path_style + color
    

    f = open(outname, 'w')
    f.write(soup.prettify())
    f.close()
    
    
    
def map_county_data_difference(party_12, party_16, outname, county_svg):
    """Create an SVG map with colors populated
    by a dictionary of (FIPS, rate) value pairs.
    """
    county_svg = open(county_svg, 'r').read()
    county_data = csv.reader(open('processed_data.csv'), delimiter = ",")
    
    soup = BeautifulSoup(county_svg, "html.parser")
    paths = soup.findAll('path')
    blue_red = ['#999999','#EC0517','#DA092E','#C70E46','#B5135D','#A21774','#901C8B','#7D20A2','#6B25B9','#582AD1','#462EE8','#3333FF', '#FFFFFF']
    # County style
    path_style = 'font-size:12px;fill-rule:nonzero;stroke:#FFFFFF;stroke-opacity:1; stroke-width:0.1;stroke-miterlimit:4;stroke-dasharray:none;stroke-linecap:butt; marker-start:none;stroke-linejoin:bevel;fill:'

    for p in paths:
        if p['id'] not in ['State_Lines', 'separator']:
            #print(str(p['id']))
            try:
                rate = party_12[int(p['id'])]
                rate2 = party_16[int(p['id'])]
            except: 
                continue
            if rate == 'red' and rate2 == 'blue':
                color_class = 9
            elif rate == 'blue' and rate2 == 'red':
                color_class = 1
            else:
                color_class = 0
                
            color = blue_red[color_class]
            p['style'] = path_style + color
    
    f = open(outname, 'w')
    f.write(soup.prettify())
    f.close()
    
    
    
def plot_pca_kmeans(X_in, n_clusters):
   
    pca = PCA(n_components=3).fit(X_in)
    
    transformed_pca = pca.transform(X_in)
    
    X = transformed_pca
    
    kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)

    centers = kmeans.fit(X).cluster_centers_
    estimators = {'k_means_': kmeans}

    for name, est in estimators.items():
        fig = plt.figure(figsize=(4, 3))
        plt.clf()
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

        plt.cla()
        est.fit(X)
        labels = est.labels_

        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels.astype(np.float))
        
        ax.set_title(name)
        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        ax.set_xlabel('component1')
        ax.set_ylabel('component2')
        ax.set_zlabel('component3')
        
    print(pca.explained_variance_ratio_)
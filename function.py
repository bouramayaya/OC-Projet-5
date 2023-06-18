#  --------------------------------------------------------------------------
#     Librairies necessaires
#  --------------------------------------------------------------------------
from typing import List

import seaborn as sns
import missingno as msno
# from IPython.core.display_functions import display
from IPython.display import display, Markdown
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from functools import reduce
import unicodedata
from wordcloud import WordCloud, STOPWORDS  # , ImageColorGenerator
import dataframe_image as dfi
from scipy.stats import anderson

# Normalisation des données
from sklearn.preprocessing import scale

import matplotlib as mpl
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import homogeneity_score, adjusted_rand_score, adjusted_mutual_info_score
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import silhouette_samples


# Clustering Librairies import
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer, InterclusterDistance
from kmodes.kprototypes import KPrototypes

from plotly.subplots import make_subplots

# import seaborn as sns
# white,dark,whitegrid,darkgrid,ticks
# Scale and Context: poster, paper,notebook,talk
# sns.set_context('notebook', font_scale = 1)
# sns.set_style('darkgrid')
sns.set_style("whitegrid")
sns.set_context('notebook')
# sns.set_style("whitegrid")
# sns.color_palette("crest", as_cmap=True)
# sns.color_palette("viridis", as_cmap=True)
sns.color_palette("coolwarm", as_cmap=True)

import warnings
warnings.filterwarnings('ignore')

# https://github.com/SkalskiP/courses/tree/master
# import dataframe_image as dfi
# !pip install dataframe_image --no-warn-script-location
# Ci dessous quelques valeurs de palette seaborn.
# muted, pastel, coolwarm,'Accent', 'cubehelix',
# 'gist_rainbow', 'terrain', 'viridis', vlag

#  --------------------------------------------------------------------------
#     Quelques regalages et paramètres fixés
#  --------------------------------------------------------------------------

imgPath = 'D:/OpenClassrooms/Projet 5/Soutenance/'

mycolors = ['#FFBB00', '#3C096C', '#9D4EDD', '#FFE270','#EAD39C', '#7D5E3C', '#A10115', "#4CAF50", '#D72C16', '#F0EFEA', '#C0B2B5', '#221f1f', "black",
            "hotpink", "b", "#4CAF50",'#EAD39C', '#7D5E3C']
AllColors = ['#99ff99', '#66b3ff', '#D72C16', '#4F6272', '#B7C3F3', '#ff9999', '#ffcc99', '#ff6666', '#DD7596', '#8EB897',
             '#c2c2f0', '#DDA0DD', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
             '#7f7f7f', '#bcbd22', '#17becf', '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33',
             '#a65628', '#f781bf', "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
             "#000000", "b", "#4CAF50"]

# font_title = {'family': 'serif', 'color': '#114b98',
# 'weight': 'bold', 'size': 16, }

font_title = {'family': 'serif', 'color': '#114b98', 'weight': 'bold', 'size': 16, }
font_title2 = {'family': 'serif', 'color': '#114b98', 'weight': 'bold', 'size': 12, }
font_title3 = {'family': 'serif', 'color': '#4F6272', 'weight': 'bold', 'size': 10, }

font1 = {'family': 'serif', 'color': 'blue', 'weight': 'normal', 'size': 15, }
font2 = {'family': 'serif', 'color': 'blue', 'weight': 'normal', 'size': 13, }
font3 = {'family': 'serif', 'color': 'blue', 'weight': 'normal', 'size': 10, }

random_state = 42


#  --------------------------------------------------------------------------
#  Flatten les axes d'un graph (multi -subplots) et supprimer les subplots vides.
#  --------------------------------------------------------------------------

def trim_axs(axs, N):
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]


#  --------------------------------------------------------------------------
#     Fusion d'une liste de dataframe pandas
#  --------------------------------------------------------------------------

def pandasMerge(tableList: list, on: list, how: str = None):
    df = reduce(lambda first, second: first.merge(second, on=on, how=how), tableList)
    return df


#  --------------------------------------------------------------------------
#     Calcul de la valeur (modalité) la plus fréquente.
#  --------------------------------------------------------------------------

def mostFreqValues(data, keylist: list, varlist: list = None):
    """
    Cette fonction retourne une dataframe pandas.
    :param varlist:
    :param keylist: Variable ID (groupby variable)
    :param data: Dataframe en entree
    :return: dataframe pandas.
    """
    if not varlist:
        otherlist = list(set(data.columns) - set(keylist))
        df = data.groupby(keylist)[otherlist].agg(lambda x: stats.mode(x, keepdims=True)[0][0]).reset_index()
    else:
        df = data.groupby(keylist)[varlist].agg(lambda x: stats.mode(x, keepdims=True)[0][0]).reset_index()
    return df





def getKeyDict(d, val):
    return [k for k, v in d.items() if v == val]


# indCodeNameDict = dict(zip(dfa['Indicator Code'], dfa['Indicator Name']))

#  --------------------------------------------------------------------------
#  Graphique (en subplots) des valeurs manquantes d'un dataframe.
#  --------------------------------------------------------------------------


def missingGraph(df, indicatorList, indCodeNameDict):
    nplot = len(indicatorList)
    figsize = (20, 30)
    cols = 2
    rows = nplot // cols + 1

    axs = plt.figure(figsize=figsize, constrained_layout=True).subplots(rows, cols)
    axs = trim_axs(axs, nplot)
    for ax, ind in zip(axs, indicatorList):
        subset = df.loc[df['Indicator Name'] == ind, "2000":"2015"]
        msno.matrix(subset, width_ratios=(15, 1), fontsize=12, labels=True,
                    sparkline=False, freq=None, ax=ax)
        ax.set_title(str(getKeyDict(indCodeNameDict, ind)) + " : " + str(ind), fontsize=16)
        # cols = [i.get_text() for i in m.axes.get_xticklabels()]
    plt.show()


# Police d'un titre


MultiCountFont = {'family': 'serif', 'color': '#4F6272',  # "#B7C3F3",
                  'weight': 'bold', 'size': 10, }


#  --------------------------------------------------------------------------
#  Graphique Count (multi-graphe) sur plusieurs variables.
#  --------------------------------------------------------------------------


def numberBycategory(data, x, axis, title, labelsize, rotation, ylabsize, saturation, palette):
    g = sns.countplot(x=x, data=data, saturation=saturation, ax=axis, palette=palette)
    for i, label1 in enumerate(axis.containers):
        axis.bar_label(label1, label_type='edge', fontsize=labelsize)
    for tick in axis.get_xticklabels():
        tick.set_rotation(rotation)
    # plt.xlabel(xtitle, color='black')

    # plt.ylabel(ytitle, color='black')
    # ax.tick_params(axis='x', colors='black')
    # g.set_xticks([])
    g.set_xlabel('')
    g.set_ylabel('Nombre', size=ylabsize)
    # ax.title.set_text(title, fontdict = {'font.size':10})
    g.set_title(title, fontdict=font_title3)
    return g


def g_multi_count(data, listvar, title, ncols=2, figsize=(12, 5), labelsize=8, rotation=0,
                  ylabsize=8, saturation=1, palette='Set3', graphName=''):
    nplot = len(listvar)
    nrows = nplot // ncols + 1
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols,  # sharey=True,  sharex=True,
                            constrained_layout=True, squeeze=False, figsize=figsize)
    axs = trim_axs(axs, nplot)
    for ax, var in zip(axs, listvar):
        numberBycategory(data, var, ax, var, labelsize, rotation, ylabsize, saturation, palette)
    fig.text(0.5, 0.90, title, ha="center", fontdict=font_title)
    if graphName != '':
        fig.savefig(imgPath + graphName, bbox_inches='tight')
    # fig.subplots_adjust(top=0.88)
    # fig.tight_layout()
    plt.show()


#  --------------------------------------------------------------------------
#  Graphique avec une aggreg function  (multi-graphe) sur plusieurs variables.
#  --------------------------------------------------------------------------


def aggBycategory(data, x, y: float, agg_func, axis, title, labelsize, rotation,
                  xlabsize, ylabsize, palette, saturation):
    dftemp = (data.groupby(x).agg({y: agg_func}).reset_index().round(0))
    g = sns.barplot(x=x, y=y, data=dftemp, saturation=saturation, ax=axis, palette=palette)
    for label1 in axis.containers:
        axis.bar_label(label1, label_type='edge', fontsize=labelsize)  # color= AllColors[i],
    for tick in axis.get_xticklabels():
        tick.set_rotation(rotation)
    # plt.xlabel(xtitle, color='black')
    # plt.ylabel(ytitle, color='black')
    axis.tick_params(axis='x', colors='black')
    axis.tick_params(axis='y', colors='black')
    for tick in axis.xaxis.get_major_ticks():
        tick.label.set_fontsize(xlabsize)

    for tick in axis.yaxis.get_major_ticks():
        tick.label.set_fontsize(ylabsize)

    # axes.legend(prop=dict(size=10))
    #    axis.yaxis.set_tick_params(labelsize=ylabsize)
    # g.set_xticks([])
    g.set_xlabel('', size=xlabsize)
    g.set_ylabel(agg_func + ' of ' + y, size=ylabsize)
    # ax.title.set_text(title, fontdict = {'font.size':10})
    g.set_title(title + "\n", fontdict=font_title3)
    return g


#  --------------------------------------------------------------------------
# muted, pastel, coolwarm,'Accent', 'cubehelix',
# 'gist_rainbow', 'terrain', 'viridis', vlag
#  --------------------------------------------------------------------------

def g_multi_agg(data, listvar, aggvar, agg_func, title, ncols=2, figsize=(12, 5), labelsize=8,
                rotation=0, xlabsize=8, ylabsize=8, palette='coolwarm', saturation=0.85, graphName='',
                shareyy: bool = True, sharexx: bool = False):
    nplot = len(listvar)
    nrows = nplot // ncols + 1
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=shareyy, sharex=sharexx,
                            constrained_layout=True, squeeze=False, figsize=figsize)
    axs = trim_axs(axs, nplot)
    for ax, var in zip(axs, listvar):
        aggBycategory(data, var, aggvar, agg_func, ax, var, labelsize,
                      rotation, xlabsize, ylabsize, palette, saturation)
    fig.text(0.5, 0.90, title, ha="center", fontdict=font_title2)
    if graphName != '':
        fig.savefig(imgPath + graphName, bbox_inches='tight')
    # fig.subplots_adjust(top=0.88)
    # fig.tight_layout()
    plt.show()


#  --------------------------------------------------------------------------
#     Function countplot de seaborn
#  --------------------------------------------------------------------------

def countplot2(data, x, title, xtitle, ytitle, figsize=(10, 3),
               labelsize=8, rotation=20, graphName: str = None, style='fast'):
    plt.style.use(style)  # 'fivethirtyeight', 'ggplot', 'bmh','seaborn-v0_8-whitegrid'
    fig = plt.figure(figsize=figsize)
    ax = sns.countplot(x=x, data=data, saturation=1)
    for i, label1 in enumerate(ax.containers):
        ax.bar_label(label1, label_type='edge', fontsize=labelsize)  # color= AllColors[i],
    for tick in ax.get_xticklabels():
        tick.set_rotation(rotation)
    plt.xlabel(xtitle, color='black')
    plt.ylabel(ytitle, color='black')
    ax.tick_params(axis='x', colors='black')
    fig.text(0.5, 0.90, title, ha="center", fontdict=font_title2)
    # plt.tight_layout()
    if graphName:
        plt.savefig(imgPath + graphName, bbox_inches='tight')
    plt.show()


#  --------------------------------------------------------------------------
#     Fonction barplot de seaborn
#  --------------------------------------------------------------------------


def barplot2(data, x, y, title, xtitle, ytitle, agg_func, figsize=(10, 3),
             labelsize=8, rotation=20, graphName: str = None, style='ggplot', palette='muted'):
    dftemp = (data.groupby(x).agg({y: agg_func}).reset_index())
    dftemp = dftemp.sort_values(by=[y], ascending=False)
    plt.style.use(style)  # 'fivethirtyeight', 'ggplot', 'bmh','seaborn-v0_8-whitegrid'
    fig = plt.figure(figsize=figsize)
    ax = sns.barplot(x=x, y=y, data=dftemp, palette=palette)  # muted, pastel,coolwarm,'Accent','cubehelix',
    # 'gist_rainbow', 'terrain', 'viridis',vlag
    for i, label1 in enumerate(ax.containers):
        ax.bar_label(label1, label_type='edge', fontsize=labelsize)  # color= AllColors[i],
    for tick in ax.get_xticklabels():
        tick.set_rotation(rotation)

    plt.xlabel(xtitle, color='black')
    plt.ylabel(ytitle, color='black')
    # plt.title(title )
    ax.tick_params(axis='x', colors='black')
    fig.text(0.5, 0.90, title, ha="center", fontdict=font_title2)
    # plt.tight_layout()
    if graphName:
        plt.savefig(imgPath + graphName, bbox_inches='tight')
    plt.show()


def barplot3(pd_df, varX, varY, agg_func, title, xrotation=0, labrotation=45,
             barlabsize=8, labcolor='m', figsize=(10, 3), graphName: str = None):
    fig = plt.figure(figsize=figsize)
    pd_df = (pd_df.groupby(varX).agg({varY: agg_func}).reset_index().round(0).sort_values(by=[varX]))
    g = sns.barplot(x=varX, y=varY, data=pd_df)
    g.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    g.set(xlabel=varX, ylabel=varY)
    # x labels
    g.set_xticklabels(pd_df[varX])
    for item in g.get_xticklabels():
        item.set_rotation(xrotation)
    # bar label orientation
    for i, v in enumerate(pd_df[varY].items()):
        g.text(i, v[1], "{:}".format(v[1]), color=labcolor, va='bottom',
               rotation=labrotation, size=barlabsize)

    fig.text(0.5, 0.90, title, ha="center", fontdict=font_title2)
    plt.tight_layout()
    if graphName:
        fig.savefig(imgPath + graphName, bbox_inches='tight')
    plt.show()


# Filtrer une base de données selon les valeurs d'une seconde base.

def filterKeys(data1, key1, data2, key2):
    return data1.loc[data1[key1].isin(list(set(data2[key2].unique().tolist()))), :]


# Graphique
# features_100g = ['proteins_100g', 'fat_100g', 'carbohydrates_100g', 'sugars_100g', 'salt_100g', 'sodium_100g',
# 'saturated-fat_100g', 'fiber_100g', 'nutrition-score-fr_100g', 'nutrition-score-uk_100g']


def graph(dataframe, features_100g):
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(21, 40))
    sub = 0
    for i in range(len(features_100g)):
        fig.add_subplot(6, 2, i + 1)
        colonne = features_100g[i]
        ax = sns.boxplot(x="pnns_groups_1", y=colonne, data=dataframe[dataframe["pnns_groups_1"] != "unknown"])
        for tick in ax.get_xticklabels():
            tick.set_rotation(20)
        sub += 1

    fig.text(0.5, 0.90, r"Distribution des variables nutritionnelles" "\n"
                        "par catégories pnns_groups_1", ha="center",
             fontdict=font_title)
    plt.show()


def differenceKeys(data1, key1, data2, key2):
    print('Les keys presents dans les deux dataframes *{0}* & *{1}*:'.format(namestr(data1, globals()),
                                                                             namestr(data2, globals())),
          len(list(set(data1[key1].unique().tolist()) & set(data2[key2].unique().tolist()))))
    print("Les keys presents uniquement dans *{}* :".format(namestr(data1, globals())),
          len(list(set(data1[key1].unique().tolist()) - set(data2[key2].unique().tolist()))))
    print("Les keys presents uniquement dans *{}* :".format(namestr(data2, globals())),
          len(list(set(data2[key2].unique().tolist()) - set(data1[key1].unique().tolist()))))

    print('--------------------------------')
    df0 = data1.loc[data1[key1].isin(list(set(data1[key1].unique().tolist()) & set(data2[key2].unique().tolist()))), :]
    df1 = data1.loc[data1[key1].isin(list(set(data1[key1].unique().tolist()) - set(data2[key2].unique().tolist()))), :]
    df2 = data2.loc[data2[key2].isin(list(set(data2[key2].unique().tolist()) - set(data1[key1].unique().tolist()))), :]
    return print('--------------------------------'), df0, df1, df2


#  --------------------------------------------------------------------------
#  Supprimer les accents
#  --------------------------------------------------------------------------


def remove_accents(string):
    return ''.join(char for char in unicodedata.normalize('NFD', string)
                   if unicodedata.category(char) != 'Mn')


#  --------------------------------------------------------------------------
#  Detection de doublons dans la base
#  --------------------------------------------------------------------------


def doublons(data, varlist: list = None):
    """
    Cette fonction retourne les lignes dupliquées de la base.
    :param varlist:
    :param data: dataframe pandas en entrée.
    :return: un dataframe avec les lignes en double
    """
    if not varlist:
        return data.loc[data.duplicated(keep=False), :]
    else:
        return data.loc[data.duplicated(subset=varlist, keep=False), :]


# Creer un graphique
# fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
# axs = axs.flatten()

# for i, var in enumerate(listVar):
# sns.barplot(x=var, y='y', data=df, ax=axs[i], estimator=np.mean)
# axs[i].set_xticklabels(axs[i].get_xtickelabels(), rotation=90)
# Ajuster l'espace entre les graphs
# fig.tight_layout()
# Afficher le graphique
# plt.show()


def vgraphCountPlot(data, x, hue='', figsize=(10, 8), labelsize=8, rotation=90,
                    graphName: str = None):
    """
    Cette fonction produit le graphique avec la méthode countplot de seaborn (sns).
    :param data: dataframe en format pandas
    :param x: la variable qualitative
    :param hue: Optionnel
    :return: graphique
    """
    plt.figure(figsize=figsize)
    if hue != '':
        ax = sns.countplot(data=data, x=x, hue=hue)
        for label1 in ax.containers:
            ax.bar_label(label1, size=labelsize)
    else:
        ax = sns.countplot(data=data, x=x)
        for label1 in ax.containers:
            ax.bar_label(label1, size=labelsize)

    locs, labels = plt.xticks()
    plt.setp(labels, rotation=rotation)
    plt.tight_layout()
    if graphName:
        plt.savefig(imgPath + graphName, bbox_inches='tight')
    plt.show()


def hgraphCountPlot(data, x, hue='', figsize=(10, 8), labelsize=8,
                    rotation: int = 90, graphname: str = None):
    """
    Cette fonction produit le graphique avec la méthode countplot de seaborn (sns).
    :param data: dataframe en format pandas
    :param x: la variable qualitative
    :param hue: Optionnel
    :return: graphique
    """

    plt.figure(figsize=figsize)
    if hue != '':
        ax = sns.countplot(data=data, y=x, hue=hue)
        for label1 in ax.containers:
            ax.bar_label(label1, size=labelsize)
    else:
        ax = sns.countplot(data=data, y=x)
        for label1 in ax.containers:
            ax.bar_label(label1, size=labelsize)

    locs, labels = plt.xticks()
    plt.setp(labels, rotation=rotation)
    plt.tight_layout()
    if graphname:
        plt.savefig(imgPath + graphname, bbox_inches='tight')
    plt.show()


def graphCountPlot(data, x, hue='', orient='v', figsize=(10, 8), labelsize=8):
    """
    Cette fonction produit le graphique avec la méthode countplot de seaborn (sns).
    :param orient:
    :param figsize:
    :param labelsize:
    :param data: dataframe en format pandas
    :param x: la variable qualitative
    :param hue: Optionnel
    :return: graphique
    """
    fig, ax = plt.figure(figsize=figsize).subplots(1, 1)
    if (hue != '') & (orient == 'h'):
        ax = sns.countplot(data=data, y=x, hue=hue)
        for label1 in ax.containers:
            ax.bar_label(label1, size=labelsize)
    elif (hue != '') & (orient == 'v'):
        ax = sns.countplot(data=data, x=x, hue=hue)
        for label1 in ax.containers:
            ax.bar_label(label1, size=labelsize)
    elif (hue == '') & (orient == 'h'):
        ax = sns.countplot(data=data, y=x)
        for label1 in ax.containers:
            ax.bar_label(label1, size=labelsize)
    elif (hue == '') & (orient == 'v'):
        ax = sns.countplot(data=data, x=x)
        for label1 in ax.containers:
            ax.bar_label(label1, size=labelsize)
    locs, labels = plt.xticks()
    fig.setp(labels, rotation=90)  # Rotation des labels de l'axe X
    fig.tight_layout()
    plt.show()


def convert_to_date(data, dates_features, format1='%Y-%m-%d %H:%M:%S.%f'):
    data[dates_features] = data[dates_features].apply(pd.to_datetime, format=format1)


def valeursManquantes(df, axis=0):
    """
    Cette fonction retourne les valeurs manquantes d'un dataframe selon l'axe spécifié
    :param df: dataframe en entrée
    :param axis: 0 ou 1
    :return: dataframe avec les stats sur les missings
    """
    # valeurs manquantes de la table
    misvalCount = df.isnull().sum(axis=axis)
    # Pourcentage
    misvalPercent = 100 * df.isnull().sum(axis=axis) / df.shape[axis]
    if axis == 0:
        # Variables de la base
        dfVar = df.dtypes
        # Concatenation
        misvalTable = pd.concat([misvalCount, misvalPercent, dfVar], axis=1)
    else:
        # Concatenation
        misvalTable = pd.concat([misvalCount, misvalPercent], axis=1)
        # Renommer les colonnes
    misvalTable = misvalTable.rename(columns={0: 'Missings', 1: 'Pourcentage', 2: 'type variable'})

    # Sort the table by percentage of missing descending
    misvalTable = misvalTable.sort_values(by='Pourcentage', ascending=False).round(3)

    return misvalTable


def obsWithMissings(data):
    """
    retourne les lignes presentant des données manquantes sur au moins une colonne
    :param data: dataframe
    :return: dataframe avec données manquantes sur une colonne
    """
    mask = data.isnull().any(axis=1)
    df_masked = data[mask]
    return df_masked


def Explodetuple(m):
    liste1 = []
    for t in range(m):
        if t in [0, 1]:
            liste1.append(0.1)
        else:
            liste1.append(0)
    return tuple(liste1)


def percentFreq(values):
    def my_format(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        return '{:.1f}%\n({v:d})'.format(pct, v=val)

    return my_format


def repartitionTypeVar(data, figsize=(6, 3), title="Repartition par types de variables \n",
                       graphName: str = None):
    df = data.dtypes.value_counts()
    L = len(df)
    labels = list(df.index)
    sizes = list(df)
    # print(labels,"\n",sizes)
    explode = Explodetuple(L)
    colors = AllColors[:L]
    fig1, ax1 = plt.subplots(figsize=figsize)
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct=percentFreq(df), shadow=True, startangle=0)
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax1.axis('equal')
    plt.tight_layout()
    plt.title(label=title, fontdict=font_title2)
    plt.legend()
    if graphName:
        plt.savefig(imgPath + graphName, bbox_inches='tight')
    plt.show()
    plt.close()
    df = df.reset_index()
    df.columns = ['Types de variables', 'Nombre']
    display(df.reset_index(drop=True))


def camembert(data, col: str, figsize=(7, 5), title='', graphName: str = None):
    df = data[col].value_counts().reset_index()
    L = len(df[col])
    labels = list(df['index'])
    sizes = list(df[col])
    # print(labels,"\n",sizes)
    explode = Explodetuple(L)
    colors = AllColors[:L]
    fig1, ax1 = plt.subplots(figsize=figsize)
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=0)
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax1.axis('equal')
    plt.title(label=title, fontdict=font_title2)
    plt.tight_layout()
    if graphName:
        fig1.savefig(imgPath + graphName, bbox_inches='tight')
    plt.show()


def graphbarplot(data, indicator: list, labelValue: bool, n: int = 30, pos: float = 2.1,
                 style: str = "whitegrid", figsize: tuple[int, int] = (12, 10)):
    data.sort_values(by=[indicator], ascending=False, inplace=True)
    sns.set(style=style)
    plt.figure(figsize=figsize)
    plt.title(indicator, size=13)
    sns.barplot(y=data.iloc[:n]['Country Name'], x=data.iloc[:n][indicator])
    if labelValue:
        for y, x in enumerate(data.iloc[:n][indicator]):
            plt.annotate(str(np.around(x, decimals=2)), xy=(x + pos, y), va='center', ha='right', color='white',
                         fontsize=10)
    else:
        pass
    plt.show()


def freqSimple2(data, col_names):
    # Importations
    import pandas as pd
    from IPython.display import display, Markdown
    # import numpy as np
    for col_name in col_names:
        effectifs = data[col_name].value_counts()
        modalites = effectifs.index  # l'index de effectifs contient les modalités
        tab = pd.DataFrame(modalites, columns=[col_name])  # création du tableau à partir des modalités
        tab["Nombre"] = effectifs.values
        tab["Frequence"] = tab["Nombre"] / len(data)  # len(data) renvoie la taille de l'échantillon
        # tab = tab.sort_values(col_name)   # tri des valeurs de la variable X (croissant)
        tab["Freq. cumul"] = tab["Frequence"].cumsum()  # cumsum calcule la somme cumulée
        display(Markdown('------------------------------------'))
        display(Markdown('#### Fréquence sur la variable ***' + col_name + '***'))
        display(Markdown('------------------------------------'))
        display(tab)


# -----------------------------------------------------------------------
def selectionVar(data, typevar: list):
    """
    Cette fonction récupère dans la base en entrée toutes les variables
    qui ayant le type typevar spécifié comme argument
    :param data: dataframe (pandas)
    :param typevar: type de variables à selectionner
    :return: la liste des variables du type spécifié.
    """
    listVar = list(data.select_dtypes(include=typevar).columns)
    return listVar


# ------------------------------------------------------------------------------------#
def check_duplicated(data):
    """
    Cette fonction booleenne vérifie la présence de doublons dans le dataframe.
    :param data: Il s'agit du dataframe en entrée.
    :return: Il retourne True of False
    """
    mask = data.duplicated(keep='first')
    nb_doublon = len(data[mask])
    if nb_doublon == 0:
        return False
    else:
        return True


# ------------------------------------------------------------------------------------#

def dedoublonner(data, varList=None):
    """
    Cette fonction permet de supprimer les doublons dans une base de données (dataframe)
    :param data: le dataframe (pandas)
    :param varList: Liste de variables à considerer comme identifiant.
    Si cette liste est vide, l'ens des variables de la base est considérée
    :return: le dataframe dedoublonnée
    """
    if not varList:
        data.drop_duplicates(inplace=True)
    else:
        data.drop_duplicates(subset=varList, inplace=True)
    return data


# ------------------------------------------------------------------------------------#
def completude(data):
    """
    Cette fonction évalue le taux de completion des colonnes d'un dataframe
    :param data: dataframe (pandas)
    :return: l'ensemble des colonnes avec le taux de completion dans un dataframe
    """
    # Importations
    # import pandas as pd
    # import numpy as np

    var_dict = {}
    for col in data.columns:
        var_dict[col] = []
        var_dict[col].append(round((data[col].notna().sum() / data.shape[0]) * 100, 2))
        var_dict[col].append(data[col].isna().sum())
        var_dict[col].append(round(data[col].isna().mean() * 100, 2))

    return pd.DataFrame.from_dict(data=var_dict, orient="index",
                                  columns=["Taux completion", "Nb missings", "%missings"]) \
        .sort_values(by="Taux completion", ascending=False)


# test
# ------------------------------------------------------------------------------------#

def fillingRate(data, figsize=(3, 3), grahName=''):
    """
    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    figsize : TYPE, optional
        DESCRIPTION. The default is (3, 3).
    grahName : TYPE, optional
        DESCRIPTION. The default is ''.

    Returns
    -------
    None.
    """

    filled = data.notna().sum().sum() / (data.shape[0] * data.shape[1])
    missing = data.isna().sum().sum() / (data.shape[0] * data.shape[1])

    taux = [filled, missing]
    labels = ["%filled", "%missing"]

    fig, ax = plt.subplots(figsize=figsize)
    plt.title("Taux de completion \n", fontdict=font_title)
    ax.axis("equal")
    explode = (0.1, 0)
    ax.pie(taux, explode=explode, labels=labels, autopct='%1.2f%%', shadow=True, )
    plt.legend(labels)
    if grahName != '':
        plt.savefig(imgPath + grahName, bbox_inches='tight')
    plt.show()
    plt.close()


# -----------------------------------------------------------------------
def CompareLog(data, target, targetLog, title=None, figsize=(12, 5), sharexx: bool = False, shareyy: bool = False):
    """
    Cette fonction permet de comparer les distributions d'une variable et et de son logarithme
    Ces deux deux graphes seront affichés côte à côte.
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=sharexx, sharey=shareyy, figsize=figsize)
    sns.histplot(data=data, x=target, color='darkblue', ax=axes[0])  # stat="density",
    axes[0].set_title("Données initiales", color='#2cb7b0', size=12)
    sns.histplot(data=data, x=targetLog, ax=axes[1], color='darkblue')  # stat="density",
    axes[1].set_title("Application du logarithme", color='#2cb7b0', size=12)
    if title:
        plt.suptitle(title, fontdict=font_title)
    plt.show()


def namestr(obj, namespace):
    """
    fonction retourne le nom en string
    :param obj:
    :param namespace:
    :return:
    """
    return [name for name in namespace if namespace[name] is obj]


def get_dataframe_name(df):
    """
    Cette fonction retourne le nom du dataframe en string
    :param df: dataframe
    :return: nom du dataframe en format string
    """
    name = [x for x in globals() if globals()[x] is df][0]
    return name


def get_dataframe_name2(df):
    """
    Cette fonction retourne le nom du dataframe en string
    :param df: dataframe
    :return: nom du dataframe en format string
    """
    for x in globals():
        if globals()[x] is df:
            return x


def apercu(datasets, titles):
    """
    Cette fonction fournie les stats sur une liste de base (dataframe)
    :param titles:
    :param datasets: liste de dataframe en pandas
    :return: dataframe style
    """
    # Importations
    # import pandas as pd

    # datasets = [customers_df, geolocation_df, items_df, payments_df, reviews_df, orders_df, products_df, sellers_df,
    # category_translation_df]
    # titles=[namestr(data,globals()) for data in datasets]
    data_summary = pd.DataFrame({}, )
    data_summary['datasets'] = titles
    data_summary['columns'] = [', '.join([col for col in data.columns]) for data in datasets]
    data_summary['nb_lignes'] = [data.shape[0] for data in datasets]
    data_summary['nb_colonnes'] = [data.shape[1] for data in datasets]
    data_summary['doublons'] = [data.duplicated().sum() for data in datasets]
    data_summary['nb_NaN'] = [data.isnull().sum().sum() for data in datasets]
    data_summary['NaN_Columns'] = [', '.join([col for col, null in data.isnull().sum().items() if null > 0]) for data in
                                   datasets]
    return data_summary.style.background_gradient(cmap='YlGnBu')


def apercu2(datasets, titles):
    data_summary = pd.DataFrame({}, )
    data_summary['datasets'] = titles
    data_summary['nb_lignes'] = [data.shape[0] for data in datasets]
    data_summary['nb_colonnes'] = [data.shape[1] for data in datasets]
    data_summary['doublons'] = [data.duplicated().sum() for data in datasets]
    data_summary['nb_NaN'] = [data.isnull().sum().sum() for data in datasets]
    return data_summary.style.background_gradient(cmap='YlGnBu')


def infoDataFrame(data):
    """
    Cette fonction affiche les stats relatives a la dataframe en entrée
    :param data: dataframe pandas
    :return: dataframe avec nombre de lignes et de colonnes et des infos sur le dataframe
    """
    display(Markdown('------------------------------------'))
    display(Markdown('#### Info générales sur la base : {0}'.format(namestr(data, globals()))))
    display(Markdown('------------------------------------'))
    # print('--------------------------------------------------------------------------')
    # print('Info générales sur la base : {0}'.format(namestr(data, globals())))
    data.info()
    print(" ")
    print(" ")
    nb_ligne = data.shape[0]
    nb_colonne = data.shape[1]
    print('Le jeu de données {} a {} lignes et {} colonnes.'.format(namestr(data, globals()),
                                                                    nb_ligne, nb_colonne))
    df = pd.DataFrame({'Variable': ['lignes', 'colonnes'], 'nombre': [nb_ligne, nb_colonne]})
    print(" ")
    # display(df)
    return df


# Fonction PCA


def fillingRate(data, grahName: str = None):
    filled = data.notna().sum().sum() / (data.shape[0] * data.shape[1])
    missing = data.isna().sum().sum() / (data.shape[0] * data.shape[1])

    taux = [filled, missing]
    labels = ["%filled", "%missing"]

    fig, ax = plt.subplots(figsize=(5, 5))
    plt.title("Taux de completion \n", fontdict=font_title)
    ax.axis("equal")
    explode = (0.1, 0)
    ax.pie(taux, explode=explode, labels=labels, autopct='%1.2f%%', shadow=True, )
    plt.legend(labels)
    if grahName:
        plt.savefig(imgPath + grahName, bbox_inches='tight')
    plt.show()
    plt.close()


# fillingRate(dfBuild, 'tauxCompletion.png')

# Les colonnes avec des WN, des corrections liées au climat (weather normalised)
def searchfeature(data, suffix):
    varList = [col for col in data.columns if suffix in col]
    return varList


# varList=searchfeature(dfBuild, 'WN')

# supprimer les variables avec un grand nombre de valeurs manquantes
def DeleteMissingValue(data, threshold, cols=''):
    if cols == '':
        listeCols = data.columns.tolist()
    else:
        listeCols = cols
    pct_null = data[listeCols].isnull().sum() / len(data)
    missing_features = pct_null[pct_null > threshold].index
    data.drop(missing_features, axis=1, inplace=True)


def OutliersDetection(data, var, level=0.99, graphName=''):
    UB = data[var].quantile(level)
    LB = data[var].quantile(1 - level)  # |(data[var] < LB)
    outliers = (data.loc[((data[var] > UB)), :][['PrimaryPropertyType', 'PropertyName'] + [var]]
                .sort_values(by=[var], ascending=False))

    print('Lower Bound :', LB, 'Upper Bound :', UB)
    print("Nous avons :", outliers.shape[0], ' outliers')
    # plt.boxplot(data[var])
    # plt.boxplot(data[var].values,patch_artist=True,)

    format_mapping = {var: "{:,.0f}"}  # ,"max": "{:.0f}%"
    display(outliers.style.format(format_mapping))
    fig = plt.figure(figsize=(8, 4))

    plt.subplot(121)  # subplot 1
    red_circle = dict(markerfacecolor='red', marker='o')
    mean_shape = dict(markerfacecolor='green', marker='D', markeredgecolor='green')
    sns.boxplot(data=data[var], flierprops=red_circle, showmeans=True, meanprops=mean_shape)

    plt.subplot(122)  # subplot 2
    plt.title('Repartition des Outliers')
    plt.hist(data=outliers, x='PrimaryPropertyType', color='b')
    plt.xticks(rotation=20, fontsize=9)
    plt.tight_layout()
    if graphName != '':
        graphName1 = 'df' + graphName
        dfi.export(outliers, imgPath + graphName1)
        plt.savefig(imgPath + graphName, bbox_inches='tight')
    plt.show()


# OutliersDetection(dfBuild, 'SiteEnergyUse(kBtu)', level=0.995,graphName='OutlierEnergy.png')


def nuagedemots(data, col, figsize=(16, 12), color="white", graphName: str = None):
    """
    Cette fonction retourne un graphique de nuage de mots.
    :param data: le dataframe en entree.
    :param col: La colonne contenant les données nominales
    :return: Graphique nuage de mots
    """
    display(Markdown('------------------------------------'))
    display(Markdown('#### Nuage de mots sur : {0}'.format(col)))
    display(Markdown('------------------------------------'))
    fig = plt.figure(1, figsize=figsize)
    ax1 = fig.add_subplot(1, 1, 1)
    # Creation de la variable text
    df = data.loc[data[col].notnull(), :]
    text = ' '.join(cat for cat in df[col])
    # Carte avec les mots: background_color="salmon"
    word_cloud = WordCloud(width=2000, height=1000, normalize_plurals=False, random_state=1,  # colormap="Pastel1",
                           collocations=False, stopwords=STOPWORDS, background_color=color, ).generate(text)
    ax1.imshow(word_cloud, interpolation="bilinear")
    # Afficher le nuage
    plt.imshow(word_cloud)
    plt.axis("off")
    if graphName:
        plt.savefig(imgPath + graphName, bbox_inches='tight')
    plt.show()
    plt.close()


# Imputation des valeurs manquantes.

def imputevalues(data, subsetlist, keylist, varlist):
    listables = []
    varlist2 = [vari + '_most' for vari in varlist]
    for var, var2 in zip(varlist, varlist2):
        dft = mostFreqValues(data.drop_duplicates(subset=subsetlist)[keylist + [var]].dropna(),
                             keylist=keylist, varlist=varlist)
        dft.columns = [keylist + [var2]]
        listables.append(dft)
    df = pandasMerge(listables, on=keylist, how='outer')
    dataTemp = pandasMerge([data, df], on=keylist, how='left')
    dataTemp[varlist] = dataTemp.fillna(dataTemp[varlist2])
    dataTemp.drop(columns=varlist2, axis=1, inplace=True)
    return df


# ---------------------------------------------------------------------------
# Test de normalité
# ---------------------------------------------------------------------------

def test_AndersonDarling(data, var, seuil=1):
    mask = data[var].notnull()
    datamask = data[mask]
    dataTest = datamask[var]
    # Application du Test d'Anderson-Darling
    liste = list(anderson(dataTest, dist='norm')[2])
    statistic = anderson(dataTest, dist='norm')[0]
    critical_value = anderson(dataTest, dist='norm')[1][liste.index(seuil)]
    return print('Distribution normale ' + var + ' :', statistic <= critical_value)


# ---------------------------------------------------------------------------
#  Matrice de corrélation linéaire: , cmap='Spectral'
# ---------------------------------------------------------------------------

def matriceCorrelation(data, graphName='correlation.png', figsize=(15, 15)):
    upp_mat = np.triu(data.corr(method='spearman', numeric_only=True))
    plt.figure(figsize=figsize)
    ax = sns.heatmap(data.corr(method='spearman', numeric_only=True), annot=True, fmt=".2f",
                     cmap='coolwarm', annot_kws={'size': 6}, linewidth=1,
                     mask=upp_mat)  # cmap = 'coolwarm',cmap = 'Wistia'
    plt.xticks(rotation=35, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.title("Coefficients de corrélation de Spearman", fontdict=font1)
    plt.savefig(imgPath + graphName, bbox_inches='tight', dpi=400)
    plt.show()


def strongCorr(data, threshold=0.7):
    corr = data.corr(method='spearman', numeric_only=True)
    corrPairs = corr.unstack().sort_values(kind="quicksort")
    high_corr = (pd.DataFrame(corrPairs[(abs(corrPairs) > threshold)])
                 .reset_index().rename(columns={0: 'corr_coeff'}))
    high_corr = high_corr[(high_corr.index % 2 == 0) & (high_corr['level_0'] != high_corr['level_1'])]
    high_corr.sort_values('corr_coeff', ascending=False, inplace=True)
    return high_corr


def boxplotgraph(data, x, y, rotation=90, figsize=(15, 8), graphName: str = None):
    fig = plt.figure(figsize=figsize)
    sns.boxplot(x=x, y=y, data=data)
    plt.xticks(rotation=rotation)
    plt.title("Boxplot " + x, fontdict=font1)
    if graphName:
        fig.savefig(imgPath + graphName, bbox_inches='tight', dpi=400)
    plt.show()


# ---------------------------------------------------------------------------
# Normalisation des données
# ---------------------------------------------------------------------------

def normalise_data(data, varlist: list = None):
    if varlist:
        X = data[varlist]
    else:
        varlist = data.columns
        X = data

    X_scaled = scale(X)
    idx = ["mean", "std"]
    display(pd.DataFrame(X_scaled, columns=varlist).describe().round(2).loc[idx, :])
    X_scaled = pd.concat([data[list(set(data.columns) - set(varlist))],
                          pd.DataFrame(X_scaled, columns=varlist)], axis=1)
    return X_scaled


def grisearch_kmeans(X, kmax, figsize=(8, 3)):
    K_list = []
    elbow = []
    silhouette = []
    davies_bouldin = []
    calinski_harabasz = []

    for i in range(2, kmax + 1):
        # On instancie un k-means pour k clusters
        model = Pipeline(steps=[("preprocessor", preprocessor),
                          ("kmeans", KMeans(n_clusters=i, random_state=random_state))])
        # On entraine
        model.fit(X)

        # Kmeans labels
        # labels = model.predict(X)
        kmeans_labels = model.named_steps['kmeans'].labels_
        X["kmeans_label"] = kmeans_labels
        # On recupère le K
        K_list.append(i)
        # On enregistre l'inertie obtenue
        elbow.append(model.named_steps['kmeans'].inertia_)
        # Coefficient de silhouette
        silhouette.append(silhouette_score(X, kmeans_labels))
        # Indice de Davies-Bouldin
        davies_bouldin.append(davies_bouldin_score(X, kmeans_labels))
        # Calinski-Harabasz
        calinski_harabasz.append(calinski_harabasz_score(X, kmeans_labels))

    plt.figure(figsize=figsize)
    plt.plot(range(1, kmax), elbow)
    plt.title("La méthode Elbow")
    plt.xlabel("nombre de cluster")
    plt.ylabel("Inertie intra-classe")
    plt.show()

    dataframe = pd.DataFrame({'K': K_list, 'elbow': elbow, 'silhouette': silhouette,
                              'davies_bouldin': davies_bouldin,
                              'calinski_harabasz': calinski_harabasz})
    dataframe.sort_values(by=['silhouette', 'davies_bouldin', 'calinski_harabasz'],
                          ascending=[True, False, True])
    return dataframe


# grisearch_kmeans(X, 10,figsize=(8,3))


# ---------------------------------------------------------------------------
#  Plot radar chart des clusters d'un K-means clustering
# ---------------------------------------------------------------------------


def plot_radar_chart(cluster_data):
    labels = cluster_data.columns[:-1]
    num_vars = len(labels)

    # Calculer les angles pour le radar chart
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    # Préparer les données pour le radar chart
    cluster_values = cluster_data.iloc[:, :-1].values.tolist()
    cluster_values += cluster_values[:1]

    # Tracer le radar chart
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, cluster_values, linewidth=1, linestyle='solid')
    ax.fill(angles, cluster_values, alpha=0.25)

    # Ajouter les étiquettes des variables
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    # Ajouter une légende avec les clusters
    cluster_name = cluster_data['Cluster'].iloc[0]
    ax.legend([f'Cluster {cluster_name}'])

    # Afficher le plot
    plt.show()


def silhouettescore(X, k_list=(2, 11), gaphname='nbClusterElbow.png'):
    kmeans_visualizer = Pipeline([
        ("preprocessor", preprocessor),
        ("kelbowvisualizer", KElbowVisualizer(KMeans(random_state=random_state),
                                              K=k_list, n_jobs=-1))])
    kmeans_visualizer.fit(X)
    kmeans_visualizer.named_steps['kelbowvisualizer'].show(outpath=imgPath + gaphname,
                                                           bbox_inches='tight', dpi=400)


metrics = ['distortion', 'silhouette', 'calinski_harabasz']


def elbowmetrics(X, preprocessor, metrics: list = None, k_list=(2, 11),
                 cols=2, figsize=(12, 5), graphname='elbow_metrics.png'):
    if metrics:
        metrics_list = metrics
    else:
        metrics_list = ['distortion']

    nplot = 2 * len(metrics_list)
    rows = len(metrics_list)  # nplot // cols + 1

    fig, axs = plt.subplots(rows, cols, figsize=figsize, sharex=False, sharey=False)
    axs = trim_axs(axs, nplot)

    for i, m in enumerate(metrics_list):
        kmeans_visualizer = Pipeline([
            ("preprocessor", preprocessor),
            ("kelbowvisualizer", KElbowVisualizer(KMeans(random_state=random_state),
                                                  K=k_list, metric=m, n_jobs=-1,
                                                  ax=axs[2 * i]))])
        kmeans_visualizer.fit(X)
        kmeans_visualizer.named_steps['kelbowvisualizer'].finalize()

        # Meilleur valeur du K
        K = kmeans_visualizer.named_steps['kelbowvisualizer'].elbow_value_
        # Silhouette Visualizer
        silhouette_visualizer = Pipeline([
            ("preprocessor", preprocessor),
            ("silhouettevisualizer", SilhouetteVisualizer(KMeans(n_clusters=K, random_state=random_state),
                                                          ax=axs[2 * i + 1]))])
        silhouette_visualizer.fit(X)
        silhouette_visualizer.named_steps['silhouettevisualizer'].show(bbox_inches='tight', dpi=400)
    fig.text(0.198, .99, 'Clustering: Silhouette scores',
             fontsize=15, fontweight='bold', fontfamily='serif')
    fig.savefig(imgPath + graphname, bbox_inches='tight', dpi=400)
    plt.show()
    return K


def distanceInterCluster(X, preprocessor, K, graphname: str = None):
    distance_visualizer = Pipeline([
        ("preprocessor", preprocessor),
        ("distancevisualizer", InterclusterDistance(KMeans(n_clusters=K,
                                                           random_state=random_state),
                                                    n_jobs=-1, ))])
    distance_visualizer.fit(X)
    if graphname:
        distance_visualizer.named_steps['distancevisualizer'].show(outpath=imgPath + graphname,
                                                                   bbox_inches='tight', dpi=400)
    else:
        distance_visualizer.named_steps['distancevisualizer'].show(bbox_inches='tight', dpi=400)


def rows_cols_update(rows, row, cols, col):
    if col + 1 > cols:
        row, col = row + 1, 1
    else:
        row, col = row, col + 1
    return row, col


def plot_radars(data, group, cols=2, figsize=(15, 8)):
    nplot = len(data[group].drop_duplicates().values)
    row, col = 1, 1
    rows = int(np.ceil(nplot / cols))  # .astype(int) # nplot // cols + 1

    #     scaler = MinMaxScaler()
    #     data = pd.DataFrame(scaler.fit_transform(data),
    #                         index=data.index,
    #                         columns=data.columns).reset_index()

    # fig = go.Figure()
    # Créer une instance de la classe make_subplots avec les dimensions souhaitées
    fig = make_subplots(rows=rows, cols=cols, specs=[[{'type': 'polar'}] * cols] * rows,
                        subplot_titles=[f'Cluster {cat} \n' for cat in data[group]])
    # Spécification des tailles des figures
    # fig.update_layout(
    #     width=1000,  # Largeur totale de la figure
    #     height=900,  # Hauteur totale de la figure
    #     # # Les valeurs ci-dessous spécifient les proportions de chaque sous-tracé par rapport à la taille
    #     # # totale de la figure
    #     # # Vous pouvez ajuster ces valeurs en fonction de vos besoins
    #     # row_heights=[0.4, 0.6],  # Hauteurs des rangées (proportions)
    #     # column_widths=[0.6, 0.4]  # Largeurs des colonnes (proportions)
    # )
    for k in data[group]:
        fig.add_trace(go.Scatterpolar(
            r=data[data[group] == k].iloc[:, 1:].values.reshape(-1),
            theta=data.columns[1:],
            fill='toself',
            name='Cluster ' + str(k)
        ), row=row, col=col)
        # Ajout des sous titres
        fig.update_layout(title_text=f"Cluster {k}")
        # Update des lignes et colonnes
        row, col = rows_cols_update(rows, row, cols, col)

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,  # Supprimer la légende
        title={
            'text': "Comparaison des moyennes par variable des clusters ",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        title_font_color="blue",
        title_font_size=20)

    fig.show()


def pca_eboulis(X, preprocessor, limit=95, figsize=(12, 5)):
    # PCA Pipeline
    pca = Pipeline([("preprocessor", preprocessor),
                    ("pca", PCA(svd_solver='full'))])
    pca.fit(X)
    X_projected = pca.transform(X)

    # Variance expliquée
    varexpl = pca.named_steps['pca'].explained_variance_ratio_ * 100

    display(Markdown('------------------------------------'))
    display(Markdown('##### Variance expliquée par axe :'))
    scree = (pca.named_steps['pca'].explained_variance_ratio_ * 100).round(2)
    print(scree)

    display(Markdown('------------------------------------'))
    display(Markdown('##### Variance expliquée cumulée :'))
    scree_cum = scree.cumsum().round(2)
    print(scree_cum)

    # Variance expliquée cumulée
    plt.figure(figsize=figsize)
    plt.bar(np.arange(len(varexpl)) + 1, varexpl)

    cumSumVar = varexpl.cumsum()
    plt.plot(np.arange(len(varexpl)) + 1, cumSumVar, c="red", marker='o')
    plt.axhline(y=95, linestyle="--", color="green", linewidth=1)

    valid_idx = np.where(cumSumVar >= limit)[0]
    min_plans = valid_idx[cumSumVar[valid_idx].argmin()] + 1
    plt.axvline(x=min_plans, linestyle="--", color="green", linewidth=1)

    plt.xlabel("rang de l'axe d'inertie")
    plt.xticks(np.arange(len(varexpl)) + 1)
    plt.ylabel("pourcentage d'inertie")
    plt.title("{}% de la variance totale est expliquée" \
              " par les {} premiers axes".format(limit, min_plans))
    plt.show(block=False)

    return X_projected, limit, min_plans


def correlation_graph(X, min_plans, preprocessor, x_y, figsize=(10, 4)):
    """Affiche le graphe des correlations

    Positional arguments :
    -----------------------------------
    pca : sklearn.decomposition.PCA : notre objet PCA qui a été fit
    x_y : list ou tuple : le couple x,y des plans à afficher, exemple [0,1] pour F1, F2
    features : list ou tuple : la liste des features (ie des dimensions) à représenter
    """
    features = X.columns
    # Extrait x et y
    x, y = x_y

    # Instanciation pca
    pca = Pipeline([("preprocessor", preprocessor),
                    ("pca", PCA(n_components=min_plans, svd_solver='full'))])
    pca.fit(X)
    # Taille de l'image (en inches)
    fig, ax = plt.subplots(figsize=figsize)

    # Pour chaque composante :
    for i in range(0, pca.named_steps['pca'].components_.shape[1]):
        # Les flèches
        ax.arrow(0, 0,
                 pca.named_steps['pca'].components_[x, i],
                 pca.named_steps['pca'].components_[y, i],
                 head_width=0.07,
                 head_length=0.07,
                 width=0.02, )

        # Les labels
        plt.text(pca.named_steps['pca'].components_[x, i] + 0.05,
                 pca.named_steps['pca'].components_[y, i] + 0.05,
                 features[i])

    # Affichage des lignes horizontales et verticales
    plt.plot([-1, 1], [0, 0], color='grey', ls='--')
    plt.plot([0, 0], [-1, 1], color='grey', ls='--')

    # Nom des axes, avec le pourcentage d'inertie expliqué
    plt.xlabel('F{} ({}%)'.format(x + 1, round(100 * pca.named_steps['pca'].explained_variance_ratio_[x], 1)))
    plt.ylabel('F{} ({}%)'.format(y + 1, round(100 * pca.named_steps['pca'].explained_variance_ratio_[y], 1)))

    plt.title("Cercle des corrélations (F{} et F{}) \n".format(x + 1, y + 1), fontdict=font_title2)

    # Le cercle
    an = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(an), np.sin(an))  # Add a unit circle for scale

    # Axes et display
    plt.axis('equal')
    plt.grid(False)
    plt.show(block=False)



# --- Define K-Means Functions ---
def nbclusters(X, metrics:list, k_list:list=(2, 10), figsize=(14, 5)):
    color_palette=['#FFCC00', '#54318C']
    set_palette(color_palette)
    title=dict(fontsize=12, fontweight='bold', style='italic', fontfamily='serif')
    text_style=dict(fontweight='bold', fontfamily='serif')
    layout=(1, len((metrics)))

    fig = plt.figure(figsize=figsize)
    for i, m in enumerate(metrics):
        ax = fig.add_subplot(*layout,i+1)
        kmeans_visualizer = Pipeline([
            ("preprocessor", preprocessor),
            ("kelbowvisualizer", KElbowVisualizer(KMeans(random_state=random_state),
                                                  K=k_list, metric=m, n_jobs=-1,
                                                  ax=ax))])
        kmeans_visualizer.fit(X)
        kmeans_visualizer.named_steps['kelbowvisualizer'].finalize()

        kmeans_visualizer.named_steps['kelbowvisualizer'].ax.set_title(m+' Score Elbow\n', **title)
        kmeans_visualizer.named_steps['kelbowvisualizer'].ax.tick_params(labelsize=7)
        # for text in kmeans_visualizer.named_steps['kelbowvisualizer'].ax.legend_.texts: text.set_fontsize(9)
        for spine in kmeans_visualizer.named_steps['kelbowvisualizer'].ax.spines.values():
            spine.set_color('None')
        kmeans_visualizer.named_steps['kelbowvisualizer'].ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), borderpad=2, frameon=False, fontsize=8)
        kmeans_visualizer.named_steps['kelbowvisualizer'].ax.grid(axis='y', alpha=0.5, color='#9B9A9C', linestyle='dotted')
        kmeans_visualizer.named_steps['kelbowvisualizer'].ax.grid(axis='x', alpha=0)
        kmeans_visualizer.named_steps['kelbowvisualizer'].ax.set_xlabel('\nK Values', fontsize=9, **text_style)
        kmeans_visualizer.named_steps['kelbowvisualizer'].ax.set_ylabel(m+' Scores\n', fontsize=9, **text_style)

    plt.suptitle('Clustering: nombre optimal de clusters (K)', fontsize=14, **text_style)
    plt.gcf().text(0.9, 0.05, 'github.com/bouramayaya', style='italic', fontsize=7)
    plt.tight_layout()
    plt.show()



    def calculate_figsize(num_subplots, subplot_size):
        rows = int(num_subplots ** 0.5)
    cols = (num_subplots + rows - 1) // rows
    figsize = (cols * subplot_size[0], rows * subplot_size[1])
    return rows, cols, figsize

def create_subplots(num_subplots, subplot_size, cols=None):
    if cols:
        rows = int(np.ceil(num_subplots / cols))
        figsize=(cols * subplot_size[0], rows * subplot_size[1])
    else:
        rows = int(num_subplots ** 0.5)
        cols = (num_subplots + rows - 1) // rows
        figsize=(cols * subplot_size[0], rows * subplot_size[1])
    fig, axes = plt.subplots(nrows=rows+2, ncols=cols,
                             constrained_layout=True, squeeze=False, figsize=figsize, dpi=96)
    # fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)
    fig.subplots_adjust(hspace=0.2)  # Ajustement de l'espacement vertical entre les sous-plots

    return rows, cols, figsize, fig, axes


def make_spider2(df, cols, rows, row, cluster_var, title, color):

    # number of variable
    categories=list(df)[1:]
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(cols, rows, row+1, polar=True)

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='black',fontfamily='serif',fontweight='light', size=8)
    #ax.set_xticks([]) # turn labels off if you want - can look quite nice

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0,0.2,0.30,0.40,0.50,0.75,0.100],
               ["0","0.2","0.3","0.4","0.5","0.75","1"],
               color="grey", size=4)
    plt.ylim(0,1)


    # Ind1
    values=df.loc[row].drop(cluster_var).values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
    ax.fill(angles, values, color=color, alpha=0.4)

    # Add a title
    plt.title(title, size=10, fontfamily='serif',fontweight='bold', y=1.2)
    plt.tight_layout()

def viz_radar2(X_data, model,  subplot_size, scaler = MinMaxScaler(), cols=None):
    my_dpi=96

    model.fit(X_data)
    labels = model.labels_
    X_clusters=X_data.assign(clusters=labels)
    X_clusters = X_clusters.groupby('clusters', as_index=False).mean()
    my_palette = plt.cm.get_cmap("crest", len(X_clusters.index))

    unique, counts = np.unique(labels, return_counts=True)
    num_subplots=len(unique)
    if cols:
        rows, cols, figsize, fig, axes=create_subplots(num_subplots, subplot_size, cols)
    else:
        rows, cols, figsize, fig, axes=create_subplots(num_subplots, subplot_size)
        # rows, cols, figsize=calculate_figsize(num_subplots, subplot_size)
    # fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=figsize, dpi=my_dpi)
    print('Nombre rows :', rows)
    print('Nombre cols :', cols)
    print('Nombre figsize :', figsize)

    axes = trim_axs(axes, num_subplots)
    data1=X_clusters[['clusters']]
    clusters_list=X_clusters['clusters'].tolist()
    data2=X_clusters.drop('clusters', axis=1)

    group2=list(data2.columns)
    data2 = pd.DataFrame(scaler.fit_transform(data2),
                         columns=data2.columns)
    data=pd.concat([data1, data2], axis=1)
    display(data)
    # Loop to plot
    for i, row in enumerate(clusters_list):
        make_spider2(data, rows, cols, row, 'clusters',
                     title='Cluster: '+str(X_clusters['clusters'][row]),
                     color=mycolors[row], ax=axes[i]) # '#244747'
    fig.show()

# Libraries
import matplotlib.pyplot as plt
import pandas as pd
from math import pi
mycolors = ['#A10115',"#4CAF50", '#C0B2B5', '#221f1f',  "hotpink", "#4CAF50",
            '#EAD39C', '#7D5E3C','#c2c2f0', '#DDA0DD', '#1f77b4', '#ff7f0e', '#2ca02c',
            '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', ]

def make_spider(df, row, cluster_var, subtitle, color, layout):

    # number of variable
    categories=list(df)[1:]
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(*layout,row+1, polar=True)

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='black',fontfamily='serif',fontweight='light', size=8)
    #ax.set_xticks([]) # turn labels off if you want - can look quite nice

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0,0.2,0.30,0.40,0.50,0.75,0.100],
               ["0","0.2","0.3","0.4","0.5","0.75","1"],
               color="grey", size=4)
    plt.ylim(0,1)


    # Ind1
    values=df.loc[row].drop(cluster_var).values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
    ax.fill(angles, values, color=color, alpha=0.4)

    # Add a subtitle
    plt.title(subtitle, size=10, fontfamily='serif',fontweight='bold', y=1.2)
    plt.tight_layout()

def viz_radar(X_data, model,  cols=4, scaler = MinMaxScaler(feature_range=(0.1,1)), dpi=96,
              title=None, graphName=None):
    my_dpi=96
    model.fit(X_data)
    labels = model.labels_
    X_clusters=X_data.assign(clusters=labels)
    X_clusters = X_clusters.groupby('clusters', as_index=False).mean()
    my_palette = plt.cm.get_cmap("crest", len(X_clusters.index))

    unique, counts = np.unique(labels, return_counts=True)
    num_subplots=len(unique)
    rows = int(np.ceil(num_subplots / cols))
    fig=plt.figure(figsize=((1100/(4*my_dpi))*cols, (1000/(4*my_dpi))*rows), dpi=dpi)
    layout=(rows, cols)
    data1=X_clusters[['clusters']]
    data2=X_clusters.drop('clusters', axis=1)

    group2=list(data2.columns)
    data2 = pd.DataFrame(scaler.fit_transform(data2),
                         columns=data2.columns)
    data=pd.concat([data1, data2], axis=1)

    # Loop to plot
    for row in range(0, len(data.index)):
        make_spider(data, row, 'clusters',
                    subtitle='Cluster: '+str(X_clusters['clusters'][row]),
                    color=mycolors[row], layout=layout) # '#244747'
    if title:
        fig.text(0.5, 1.1, title, ha="center", fontdict=font_title2)
    if graphName:
        fig.savefig(imgPath + graphName, bbox_inches='tight')
    # fig.subplots_adjust(top=0.88)
    # fig.tight_layout()
    fig.show()

# viz_radar(X_data=X_scaled.iloc[:,:-1],
    # model=KMeans(4, random_state=random_state), dpi=96,
    # title='Diagramme radar des clusters',
# graphName='RFM_radar_plot.png')


def make_spider_all(df, row, cluster_var, title, color): # , ax=ax

    # number of variable
    categories=list(df)[1:]
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(1, 1, 1, polar=True)

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='black',fontfamily='serif',fontweight='light', size=8)
    #ax.set_xticks([]) # turn labels off if you want - can look quite nice

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0,0.2,0.30,0.40,0.50,0.75,0.100],
               ["0","0.2","0.3","0.4","0.5","0.75","1"],
               color="grey", size=4)
    plt.ylim(0,1)


    # Ind1
    values=df.loc[row].drop(cluster_var).values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
    ax.fill(angles, values, color=color, alpha=0.4)

    # Add a title
    plt.title(title, size=10, fontfamily='serif',fontweight='bold', y=1.2)
    plt.tight_layout()


def viz_radar_all(X_data, model, title, scaler = MinMaxScaler(), zoom=4):
    my_dpi=96
    model.fit(X_data)
    labels = model.labels_
    X_clusters=X_data.assign(clusters=labels)
    X_clusters = X_clusters.groupby('clusters', as_index=False).mean()
    my_palette = plt.cm.get_cmap("crest", len(X_clusters.index))

    unique, counts = np.unique(labels, return_counts=True)
    num_subplots=len(unique)
    # rows = int(np.ceil(num_subplots / cols))
    axes=plt.figure(figsize=((1000/(zoom*my_dpi)), (800/(zoom*my_dpi))), dpi=96)
    layout=(1, 1)
    data1=X_clusters[['clusters']]
    data2=X_clusters.drop('clusters', axis=1)

    group2=list(data2.columns)
    data2 = pd.DataFrame(scaler.fit_transform(data2),
                         columns=data2.columns)
    data=pd.concat([data1, data2], axis=1)
    # Create a color palette:
    my_palette = plt.cm.get_cmap("crest", len(data.index))
    # Loop to plot
    for row in range(0, len(data.index)):
        make_spider_all(data, row, 'clusters',
                        title=title,
                        color=mycolors[row]) # , ax=axes)mycolors[row]





#  --------------------------------------------------------------------------
#     Plot histogramme
#  --------------------------------------------------------------------------
import scipy.stats as st
def plot_histograms(df, features, bins=30, figsize=(12, 7), color='grey',
                    skip_outliers=False, thresh=3, layout=(3, 3), graphName=None):
    fig = plt.figure(figsize=figsize, dpi=96)

    for i, c in enumerate(features, 1):
        ax = fig.add_subplot(*layout, i)
        if skip_outliers:
            ser = df[c][np.abs(st.zscore(df[c])) < thresh]
        else:
            ser = df[c]
        ax.hist(ser, bins=bins, color=color)
        ax.set_title(c)
        ax.vlines(df[c].mean(), *ax.get_ylim(), color='red', ls='-', lw=1.5)
        ax.vlines(df[c].median(), *ax.get_ylim(), color='green', ls='-.', lw=1.5)
        ax.vlines(df[c].mode()[0], *ax.get_ylim(), color='goldenrod', ls='--', lw=1.5)
        ax.legend(['mean', 'median', 'mode'])
        # ax.title.set_fontweight('bold')
        # xmin, xmax = ax.get_xlim()
        # ax.set(xlim=(0, xmax/5))

    plt.tight_layout(w_pad=0.5, h_pad=0.65)

    if graphName:
        plt.savefig(imgPath + graphName, bbox_inches='tight', dpi=400);
    plt.show()


def elbowmetrics(X, preprocessor, metrics: list = None, k_list=(2, 11),
                 cols=2, figsize=(12, 5), graphname='elbow_metrics.png'):
    if metrics:
        metrics_list = metrics
    else:
        metrics_list = ['distortion']

    nplot = 2 * len(metrics_list)
    rows = len(metrics_list)  # nplot // cols + 1

    fig, axs = plt.subplots(rows, cols, figsize=figsize, sharex=False, sharey=False)
    axs = trim_axs(axs, nplot)

    for i, m in enumerate(metrics_list):
        kmeans_visualizer = Pipeline([
            ("preprocessor", preprocessor),
            ("kelbowvisualizer", KElbowVisualizer(KMeans(random_state=random_state),
                                                  K=k_list, metric=m, n_jobs=-1,
                                                  ax=axs[2 * i]))])
        kmeans_visualizer.fit(X)
        kmeans_visualizer.named_steps['kelbowvisualizer'].finalize()

        # Meilleur valeur du K
        K = kmeans_visualizer.named_steps['kelbowvisualizer'].elbow_value_
        # Silhouette Visualizer
        silhouette_visualizer = Pipeline([
            ("preprocessor", preprocessor),
            ("silhouettevisualizer", SilhouetteVisualizer(KMeans(n_clusters=K, random_state=random_state),
                                                          ax=axs[2 * i + 1]))])
        silhouette_visualizer.fit(X)
        silhouette_visualizer.named_steps['silhouettevisualizer'].show(bbox_inches='tight', dpi=400)
    fig.text(0.198, .99, 'Clustering: Silhouette scores',
             fontsize=15, fontweight='bold', fontfamily='serif')
    fig.savefig(imgPath + graphname, bbox_inches='tight', dpi=96)
    plt.show()
    return K


# ------------------------------------------------------------------------
# Nombre de clusters optimal de Bouldin Davies
# -----------------------------------------------------------------------

from sklearn.metrics import davies_bouldin_score


def nbclusters_Bouldin(data, centers):
    scores = []
    for K in centers:
        kmeans = KMeans(n_clusters=K, random_state=random_state)
        labels = kmeans.fit_predict(data)
        scores.append(davies_bouldin_score(data, labels))
    plt.plot(centers, scores, linestyle='--', marker='o', color='b');
    plt.xlabel('K');
    plt.ylabel('Davies Bouldin score');
    plt.title('Davies Bouldin score vs. K')
    nb_bd = centers[np.argmin(scores)]
    print(clr.color + '*' * 53 + clr.end)
    print('le nombre optimal de clusters est de :', nb_bd)
    print(clr.color + '*' * 53 + clr.end)
    return nb_bd;


# Intercluster distance Map avec le k optimal
def distanceInterCluster(X, preprocessor, K, graphname: str = None):
    distance_visualizer = Pipeline([
        ("preprocessor", preprocessor),
        ("distancevisualizer", InterclusterDistance(KMeans(n_clusters=K, random_state=random_state)))])
    distance_visualizer.fit(X)
    if graphname:
        distance_visualizer.named_steps['distancevisualizer'].show(outpath=imgPath + graphname,
                                                                   bbox_inches='tight', dpi=96)
    else:
        distance_visualizer.named_steps['distancevisualizer'].show(bbox_inches='tight', dpi=96)


def visualizer(X, K, var1, var2, echelle=1000, figsize=(14, 8),
               titre='Visualisation clustering'):
    kmeans = Pipeline(steps=[('preprocessor', preprocessor),
                             ('kmeans', KMeans(K, random_state=random_state))])
    kmeans.fit(X)
    y_kmeans = kmeans.fit_predict(X)
    # --- Figures Settings ---
    listecolors = ['#FFBB00', '#3C096C', '#A10115', '#C0B2B5', '#221f1f', "black",
                   "hotpink", "b", "#4CAF50", '#EAD39C', '#7D5E3C', '#9D4EDD', '#FFE270']
    cluster_colors = listecolors[:K]
    labels = ['Cluster ' + (str(i + 1)) for i in range(K)] + ['Centroids']
    title = dict(fontsize=15, fontweight='bold', style='italic', fontfamily='serif')
    text_style = dict(fontweight='bold', fontfamily='serif')
    scatter_style = dict(linewidth=0.65, edgecolor='#100C07', alpha=0.85)
    legend_style = dict(borderpad=2, frameon=False, fontsize=8)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize, dpi=96)

    # --- Silhouette Plots ---
    # s_viz = SilhouetteVisualizer(kmeans, ax=ax1, colors=cluster_colors)
    s_viz = Pipeline([
        ("preprocessor", preprocessor),
        ("silhouettevisualizer", SilhouetteVisualizer(KMeans(n_clusters=K, random_state=random_state),
                                                      ax=ax1, colors=cluster_colors))])

    s_viz.fit(X)
    s_viz.named_steps['silhouettevisualizer'].finalize()
    s_viz.named_steps['silhouettevisualizer'].ax.set_title('Graphique Silhouette des clusters\n', **title)
    s_viz.named_steps['silhouettevisualizer'].ax.tick_params(labelsize=7)
    for text in s_viz.named_steps['silhouettevisualizer'].ax.legend_.texts:
        text.set_fontsize(9)
    for spine in s_viz.named_steps['silhouettevisualizer'].ax.spines.values():
        spine.set_color('None')
    s_viz.named_steps['silhouettevisualizer'].ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), **legend_style)
    s_viz.named_steps['silhouettevisualizer'].ax.grid(axis='x', alpha=0.5, color='#9B9A9C', linestyle='dotted')
    s_viz.named_steps['silhouettevisualizer'].ax.grid(axis='y', alpha=0)
    s_viz.named_steps['silhouettevisualizer'].ax.set_xlabel('\n Coefficient Values', fontsize=9, **text_style)
    s_viz.named_steps['silhouettevisualizer'].ax.set_ylabel('Cluster Labels\n', fontsize=9, **text_style)

    # --- Clusters Distribution ---
    y_kmeans_labels = list(set(y_kmeans.tolist()))
    for i in y_kmeans_labels:
        ax2.scatter(X.loc[y_kmeans == i, var1], X.loc[y_kmeans == i, var2], s=50, c=cluster_colors[i], **scatter_style)
    ax2.scatter(kmeans.named_steps['kmeans'].cluster_centers_[:, 0],
                kmeans.named_steps['kmeans'].cluster_centers_[:, 1],
                s=65, c='#0353A4', label='Centroids', **scatter_style)
    for spine in ax2.spines.values():
        spine.set_color('None')
    ax2.set_title('Scatter Plot Clusters Distributions\n', **title)
    ax2.legend(labels, bbox_to_anchor=(0.95, -0.05), ncol=5, **legend_style)
    ax2.grid(axis='both', alpha=0.5, color='#9B9A9C', linestyle='dotted')
    ax2.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    ax2.spines['bottom'].set_visible(True)
    ax2.spines['bottom'].set_color('#CAC9CD')

    # --- Waffle Chart ---
    unique, counts = np.unique(y_kmeans, return_counts=True)
    df_waffle = dict(zip(unique, counts))
    total = sum(df_waffle.values())
    wfl_square = {key: value / echelle for key, value in df_waffle.items()}
    wfl_label = {key: round(value / total * echelle, 2) for key, value in df_waffle.items()}

    ax3 = plt.subplot(2, 2, (3, 4))
    ax3.set_title('Pourcentage de chaque cluster\n', **title)
    ax3.set_aspect(aspect='auto')
    Waffle.make_waffle(ax=ax3, rows=6, values=wfl_square, colors=cluster_colors,
                       labels=[f"Cluster {i + 1} - ({k}%)" for i, k in wfl_label.items()], icons='child', icon_size=30,
                       legend={'loc': 'upper center', 'bbox_to_anchor': (0.5, -0.05), 'ncol': 4, 'borderpad': 2,
                               'frameon': False, 'fontsize': 10})
    ax3.text(0.01, -0.09, f'** 1 square ≈ {echelle} Clients', weight='bold', style='italic', fontsize=8)

    # --- Suptitle & WM ---
    plt.suptitle(titre, fontsize=14, **text_style)
    plt.gcf().text(0.9, 0.03, 'github.com/bouramayaya', style='italic', fontsize=8)
    plt.tight_layout()
    plt.show();


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from umap import UMAP


# pip uninstall umap
# pip install umap-learn
# import umap.umap_ as umap
# !pip uninstall umap
# !pip install umap-learn -i https://mirrors.ustc.edu.cn/pypi/web/simple
# from umap import UMAP

# git clone https://github.com/lmcinnes/umap
# cd umap
# pip install --user -r requirements.txt
# python setup.py install --user

def reduction(X, n_components):
    # PCA
    pipe1 = Pipeline([
        ('preprocessor', preprocessor),
        ('pca', PCA(n_components=n_components, random_state=random_state))
    ])

    X_pca = pd.DataFrame(pipe1.fit_transform(X))

    # TSNE
    pipe2 = Pipeline([
        ('preprocessor', preprocessor),
        ('tsne', TSNE(n_components=n_components, verbose=0, perplexity=45,
                      random_state=random_state, n_jobs=-1)),
    ])
    X_tsne = pd.DataFrame(pipe2.fit_transform(X))

    # UMAPgoo
    pipe3 = Pipeline([
        ('preprocessor', preprocessor),
        ('umap', UMAP(n_components=n_components, random_state=random_state, n_jobs=-1)),
    ])
    X_umap = pd.DataFrame(pipe3.fit_transform(X))
    return X_pca, X_tsne, X_umap, pipe1.named_steps['pca'].explained_variance_ratio_


import matplotlib.cm as cm


def projection_cluster(X, layout=(1, 3), nb_clusters=None, graphname=None):
    text_style = dict(fontweight='bold', fontfamily='serif')
    ann = dict(textcoords='offset points', va='center', ha='center', fontfamily='serif', style='italic')
    title = dict(color='#114b98', fontsize=15, fontweight='bold', style='italic', fontfamily='serif')
    bbox = dict(boxstyle='round', pad=0.3, color='#FFDA47', alpha=0.6)
    ncols = layout[1]
    nrows = layout[0]
    fig, ax = plt.subplots(*layout, figsize=(5 * ncols, 4 * nrows))

    if nb_clusters:
        colors = AllColors[:nb_clusters]
        color_palette = AllColors[:nb_clusters]  # ['#472165', '#FFBB00', '#3C096C', '#9D4EDD', '#FFE270']
        set_palette(color_palette)

        model_pca = KMeans(n_clusters=nb_clusters, random_state=random_state)
        model_pca.fit(X_pca)

        model_tsne = KMeans(n_clusters=nb_clusters, random_state=random_state)
        model_tsne.fit(X_tsne)

        model_umap = KMeans(n_clusters=nb_clusters, random_state=random_state)
        model_umap.fit(X_umap)

        # colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        sns.scatterplot(x=X_pca.iloc[:, 0], y=X_pca.iloc[:, 1], data=X, legend="full",
                        s=75, lw=0, alpha=1, c=model_pca.labels_, edgecolor='k', ax=ax[0])
        ax[0].set_title(r"Projection des clusters-PCA",
                        **title)

        sns.scatterplot(x=X_tsne.iloc[:, 0], y=X_tsne.iloc[:, 1], data=X, legend="full",
                        s=75, lw=0, alpha=1, c=model_tsne.labels_, edgecolor='k', ax=ax[1])
        ax[1].set_title(r"Projection des clusters-TSNE", **title)

        sns.scatterplot(x=X_umap.iloc[:, 0], y=X_umap.iloc[:, 1], data=X, legend="full",
                        s=75, lw=0, alpha=1, c=model_umap.labels_, edgecolor='k', ax=ax[2])
        ax[2].set_title(r"Projection des clusters-UMAP", **title)

        # sns.scatterplot(x=X_isomap.iloc[:,0], y=X_isomap.iloc[:,1], data=X, legend="full",
        #                 s=75, lw=0, alpha=1, c=colors, edgecolor='k', ax=ax[2])
        # ax[2].set_title("Projection-ISOMAP", **title)

    else:
        colors = ['#D72C16']
        sns.scatterplot(x=X_pca.iloc[:, 0], y=X_pca.iloc[:, 1], data=X, legend="full", c=colors, ax=ax[0])
        ax[0].set_title("Projection-PCA", **title)

        sns.scatterplot(x=X_tsne.iloc[:, 0], y=X_tsne.iloc[:, 1], data=X, legend="full", c=colors, ax=ax[1])
        ax[1].set_title("Projection-TSNE", **title)

        sns.scatterplot(x=X_umap.iloc[:, 0], y=X_umap.iloc[:, 1], data=X, legend="full", c=colors, ax=ax[2])
        ax[2].set_title("Projection-UMAP", **title)

        # sns.scatterplot(x=X_isomap.iloc[:,0], y=X_isomap.iloc[:,1], data=X, legend="full",  ax=ax[2])
        # ax[2].set_title("Projection-ISOMAP", **title)
    if graphname:
        fig.savefig(imgPath + graphname, bbox_inches='tight')


from mpl_toolkits.mplot3d import Axes3D


def projection_3D(X, K, n_components=3):
    model_pca = KMeans(n_clusters=K, random_state=random_state)
    model_pca.fit(X_pca3)
    model_tsne = KMeans(n_clusters=K, random_state=random_state)
    model_tsne.fit(X_tsne3)
    model_umap = KMeans(n_clusters=K, random_state=random_state)
    model_umap.fit(X_umap3)

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ax.scatter(X_pca3['PC1'], X_pca3['PC2'], X_pca3['PC3'], marker='o', s=30, edgecolor='k', c=model_pca.labels_)
    ax.set_xlabel('PC1 - ' + '{:.1f}%'.format(variance_explique3[0] * 100))
    ax.set_ylabel('PC2 - ' + '{:.1f}%'.format(variance_explique3[1] * 100))
    ax.set_zlabel('PC3 - ' + '{:.1f}%'.format(variance_explique3[2] * 100))
    # ax.view_init(elev=15, azim=45)

    ax1 = fig.add_subplot(1, 3, 2, projection='3d')
    ax1.scatter(X_tsne3['TSNE1'], X_tsne3['TSNE2'], X_tsne3['TSNE3'], marker='o',
                s=30, edgecolor='k', c=model_tsne.labels_)
    ax1.set_xlabel('TSNE1')
    ax1.set_ylabel('TSNE2')
    ax1.set_zlabel('TSNE3')
    # ax1.view_init(elev=15, azim=45)

    ax2 = fig.add_subplot(1, 3, 3, projection='3d')
    ax2.scatter(X_umap3['UMAP1'], X_umap3['UMAP2'], X_umap3['UMAP3'], marker='o',
                s=30, edgecolor='k', c=model_umap.labels_)
    ax2.set_xlabel('UMAP1')
    ax2.set_ylabel('UMAP2')
    ax2.set_zlabel('UMAP3')
    # ax2.view_init(elev=15, azim=45)
    plt.show()


import matplotlib.lines as lines


def plot_centroid_3D(X, name, n_clusters, model, varlist, commentaires=None):
    kmeans_model = Pipeline([("preprocessor", preprocessor),
                             (name, model)])
    kmeans_model.fit(X)
    kmeans_labels = kmeans_model.named_steps[name].labels_
    X_clusters = pd.DataFrame(data=preprocessor.fit_transform(X), index=X.index, columns=X.columns)
    X_clusters["Cluster"] = kmeans_labels
    labels = pd.DataFrame(kmeans_labels)
    # X_clusters_data = X_clusters.groupby("clusters", as_index=False).mean()
    # X_clusters_data
    # kmeans_sel = KMeans(n_clusters=2, random_state=random_state).fit(cluster_scaled)
    # labels = pd.DataFrame(kmeans_sel.labels_)
    # X_clusters = X_clusters.assign(Cluster=labels)
    grouped_km = X_clusters.groupby(['Cluster']).mean().round(1)
    grouped_km2 = X_clusters.groupby(['Cluster']).mean().round(1).reset_index()
    grouped_km2['Cluster'] = grouped_km2['Cluster'].map(str)
    grouped_km2
    mycolors = ['#2a333f', '#7D5E3C', '#939da6', '#0f4c81', '#be3e35', '#70090a', '#244747', '#A10115', '#D72C16',
                '#F0EFEA', '#C0B2B5', '#221f1f', "black", "hotpink", "b", "#4CAF50", '#EAD39C']
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(grouped_km2[varlist[0]], grouped_km2[varlist[1]], grouped_km2[varlist[2]],
               color=mycolors[:n_clusters], alpha=0.5, s=500)

    # add annotations one by one with a loop
    for line in range(0, grouped_km.shape[0]):
        ax.text(grouped_km2[varlist[0]][line],
                grouped_km2[varlist[1]][line],
                grouped_km2[varlist[2]][line], s=('Cluster \n' + grouped_km2['Cluster'][line]),
                horizontalalignment='center', fontsize=12, fontweight='light', fontfamily='serif')

    ax.set_xlabel(varlist[0])
    ax.set_ylabel(varlist[1])
    ax.set_zlabel(varlist[2])

    fig.text(0.15, 1.0, 'Visualisation clusters en 3D', fontsize=15, fontweight='bold', fontfamily='serif')
    fig.text(0.15, .98, 'Nous observons l\'espace occupé par chaque cluster (moyenne/cluster)',
             fontsize=12, fontweight='light', fontfamily='serif')

    fig.text(1.172, 1.0, 'Insight', fontsize=20, fontweight='bold', fontfamily='serif')
    fig.text(1.172, 0.75, commentaires, fontsize=12, fontweight='light', fontfamily='serif')

    l1 = lines.Line2D([1, 1], [0, 1], transform=fig.transFigure, figure=fig, color='black', lw=0.2)
    fig.lines.extend([l1])


def description_cluster(X, KBest, model, save=None):
    # --- Add K-Means Prediction to Data Frame ----
    model1 = Pipeline([("preprocessor", preprocessor),
                       ("clusterer", model)])
    model1.fit(X)
    labels = model1.named_steps['clusterer'].labels_
    df_cluster = X[X.columns]
    df_cluster['clusters'] = labels + 1
    df_cluster['clusters'] = 'Cluster ' + df_cluster['clusters'].astype(str)

    # --- Calculationg Overall Mean from Current Data Frame ---
    df_profile_overall = pd.DataFrame()
    df_profile_overall['Overall'] = df_cluster.describe().loc[['mean']].T

    # --- Summarize Mean of Each Clusters ---
    df_cluster_summary = (df_cluster.groupby('clusters').describe().T.reset_index()
                          .rename(columns={'level_0': 'Variables', 'level_1': 'Metrics'}))
    df_cluster_summary = df_cluster_summary[df_cluster_summary['Metrics'] == 'mean'].set_index('Variables')

    # --- Combining Both Data Frame ---
    print(clr.start + '.: Statistiques par cluster :.' + clr.end)
    print(clr.color + '*' * 33)
    # f = {'Cluster 1':'{:.2f}'} #column col Cluster 1 to 2 decimals
    df_profile = np.round(df_cluster_summary.join(df_profile_overall).reset_index(), decimals=3)
    df_styled = df_profile.style.format(precision=2).background_gradient(cmap='YlOrBr').hide_index()
    display(df_styled)
    if save:
        dfi.export(df_styled, imgPath + save)

    return df_profile


def desc_cluster_bar(df_cluster, varlist, cluster_var='clusters',
                     ncols=3, figsize=(14, 5)):
    nplot = len(varlist)
    # nrows = int(np.ceil(nplot / ncols))
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=figsize)
    axs = trim_axs(axs, nplot)
    for i, var in enumerate(varlist):
        sns.barplot(x=cluster_var, y=var, data=df_cluster, ax=axs[i])
        axs[i].set_title(var, fontweight="bold", fontfamily='serif', fontsize=12, color='#244247')
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.8,
                        wspace=0.4,
                        hspace=0.4)

    fig.tight_layout()
    plt.show()



def desc_cluster_boxplot(df_cluster, varlist, cluster_var='clusters',
                         ncols=3, figsize=(14, 5), outliers=True, rotation=0):
    nplot = len(varlist)
    nrows = int(np.ceil(nplot/ncols))
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=figsize)
    axs = trim_axs(axs, nplot)
    for i, var in enumerate(varlist):
        sns.boxplot(x=cluster_var, y=var, data=df_cluster, ax=axs[i], showfliers=outliers,  # , fliersize=3
                    palette=['#244247', '#91b8bd', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33',
                             '#a65628', '#f781bf', "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00",
                             "#CC79A7",
                             "#000000", "b", "#4CAF50"], boxprops=dict(alpha=.9))
        axs[i].set_title(var, fontweight="bold", fontfamily='serif', fontsize=13, color='#244247')
        axs[i].set_xlabel('')

        # sns.despine(right=True)
        # sns.despine(offset=10, trim=True)

        # axs[i].legend().set_visible(False)
        for tick in axs[i].get_xticklabels():
            tick.set_rotation(rotation)

    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.8,
                        wspace=0.4,
                        hspace=0.4)

    fig.tight_layout()
    plt.show()

    def rows_cols_update(rows, row, cols, col):
        if col + 1 > cols:
            row, col = row + 1, 1
        else:
            row, col = row, col + 1

    return row, col;


def plot_radars(X_data, model, cols=2, scaler=MinMaxScaler()):
    model.fit(X_data)
    labels = model.labels_
    X_clusters = X_data.assign(clusters=labels)
    X_clusters = X_clusters.groupby('clusters', as_index=False).mean()

    unique, counts = np.unique(labels, return_counts=True)
    nplot = len(set(unique))
    row, col = 1, 1
    rows = int(np.ceil(nplot / cols))  # .astype(int) # nplot // cols + 1

    data1 = X_clusters[['clusters']]
    data2 = X_clusters.drop('clusters', axis=1)

    group2 = list(data2.columns)
    data2 = pd.DataFrame(scaler.fit_transform(data2),
                         columns=data2.columns)
    data = pd.concat([data1, data2], axis=1)
    # data=data[[group]+group2]
    # fig = go.Figure()
    # Créer une instance de la classe make_subplots avec les dimensions souhaitées
    fig = make_subplots(rows=rows, cols=cols, specs=[[{'type': 'polar'}] * cols] * rows,
                        subplot_titles=[f'Cluster {cat} \n' for cat in data['clusters']])
    # Spécification des tailles des figures
    fig.update_layout(
        autosize=True,
        width=1100,  # Largeur totale de la figure
        height=500 * rows,  # Hauteur totale de la figure
    )
    for k in data['clusters']:
        fig.add_trace(go.Scatterpolar(
            r=data[data['clusters'] == k].iloc[:, 1:].values.reshape(-1),
            theta=data.columns[1:],
            fill='toself',
            name='Cluster ' + str(k)
        ), row=row, col=col)
        # Ajout des sous titres
        fig.update_layout(title_text=f"Cluster {k}")
        # Update des lignes et colonnes
        row, col = rows_cols_update(rows, row, cols, col)

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,  # Supprimer la légende
        title={
            'text': "Comparaison des moyennes par variable des clusters ",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        title_font_color='#114b98',
        title_font_size=20)

    fig.show()


from math import pi


def make_spider(df, row, cluster_var, subtitle, color, layout):
    # number of variable
    categories = list(df)[1:]
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(*layout, row + 1, polar=True)

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='black', fontfamily='serif', fontweight='light', size=8)
    # ax.set_xticks([]) # turn labels off if you want - can look quite nice

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0, 0.2, 0.30, 0.40, 0.50, 0.75, 0.100],
               ["0", "0.2", "0.3", "0.4", "0.5", "0.75", "1"],
               color="grey", size=4)
    plt.ylim(0, 1)

    # Ind1
    values = df.loc[row].drop(cluster_var).values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
    ax.fill(angles, values, color=color, alpha=0.4)

    # Add a subtitle
    plt.title(subtitle, size=10, fontfamily='serif', fontweight='bold', y=1.2)
    plt.tight_layout()


def viz_radar(X_data, model, cols=4, scaler=MinMaxScaler(feature_range=(0.1, 1)), dpi=96,
              zoom=4, title=None, graphName=None):
    my_dpi = 96
    model.fit(X_data)
    labels = model.labels_
    X_clusters = X_data.assign(clusters=labels)
    X_clusters = X_clusters.groupby('clusters', as_index=False).mean()
    # my_palette = plt.cm.get_cmap("crest", len(X_clusters.index))

    unique, counts = np.unique(labels, return_counts=True)
    my_palette = sns.color_palette("muted", len(unique))

    num_subplots = len(unique)
    rows = int(np.ceil(num_subplots / cols))
    fig = plt.figure(figsize=((1100 / (zoom * my_dpi)) * cols, (1000 / (zoom * my_dpi)) * rows), dpi=dpi)
    layout = (rows, cols)
    data1 = X_clusters[['clusters']]
    data2 = X_clusters.drop('clusters', axis=1)

    group2 = list(data2.columns)
    data2 = pd.DataFrame(scaler.fit_transform(data2),
                         columns=data2.columns)
    data = pd.concat([data1, data2], axis=1)

    # Loop to plot
    for row in range(0, len(data.index)):
        make_spider(data, row, 'clusters',
                    subtitle='Cluster: ' + str(X_clusters['clusters'][row]),
                    color=my_palette[row], layout=layout)  # '#244747'
    if title:
        fig.text(0.5, 1.1, title, ha="center", fontdict=font_title2)
    if graphName:
        fig.savefig(imgPath + graphName, bbox_inches='tight')
    # fig.subplots_adjust(top=0.88)
    # fig.tight_layout()
    fig.show()


def make_spider_all(df, row, cluster_var, title, color, ax):
    # number of variable
    categories = list(df)[1:]
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(1, 1, 1, polar=True)

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='black', fontfamily='serif', fontweight='light', size=8)
    # ax.set_xticks([]) # turn labels off if you want - can look quite nice

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0, 0.2, 0.30, 0.40, 0.50, 0.75, 0.100],
               ["0", "0.2", "0.3", "0.4", "0.5", "0.75", "1"],
               color="grey", size=4)
    plt.ylim(0, 1)

    # Ind1
    values = df.loc[row].drop(cluster_var).values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
    ax.fill(angles, values, color=color, alpha=0.4)

    # Add a title
    plt.title(title, size=10, fontfamily='serif', fontweight='bold', y=1.2)
    plt.tight_layout()


def viz_radar_all(X_data, model, title, scaler=MinMaxScaler(), zoom=4):
    my_dpi = 96
    model.fit(X_data)
    labels = model.labels_
    X_clusters = X_data.assign(clusters=labels)
    X_clusters = X_clusters.groupby('clusters', as_index=False).mean()
    my_palette = plt.cm.get_cmap("crest", len(X_clusters.index))

    unique, counts = np.unique(labels, return_counts=True)
    num_subplots = len(unique)
    # rows = int(np.ceil(num_subplots / cols))
    axes = plt.figure(figsize=((1000 / (zoom * my_dpi)), (800 / (zoom * my_dpi))), dpi=96)
    layout = (1, 1)
    data1 = X_clusters[['clusters']]
    data2 = X_clusters.drop('clusters', axis=1)

    group2 = list(data2.columns)
    data2 = pd.DataFrame(scaler.fit_transform(data2),
                         columns=data2.columns)
    data = pd.concat([data1, data2], axis=1)
    # Create a color palette:
    my_palette = sns.color_palette("muted", len(data.index))
    # Loop to plot
    for row in range(0, len(data.index)):
        make_spider_all(data, row, 'clusters',
                        title=title,
                        color=my_palette[row], ax=axes)



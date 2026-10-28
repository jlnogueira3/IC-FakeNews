import numpy as np
import pandas as pd
import sklearn as skl
import scipy
import datetime

import pickle 

from sklearn import feature_extraction
from sklearn import linear_model
from sklearn import svm
from sklearn import neighbors
from sklearn import tree
from sklearn import ensemble
from sklearn import naive_bayes
from sklearn import model_selection
from sklearn import metrics

#from sklearn.externals import joblib
#import sklearn.external.joblib as extjoblib
#import joblib

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pylab

import collections

import os
import sys

def plota_offline(methodsList, baseline = 0.90, annotateDist = 0.01, titulo = None):
        
    markers = ['o',"s",'^','v',"1","<",'*','X']
    colors = ['orange','brown','blue','green','purple','olive','brown']
    styles = ['-','-',':',':','-.','-.']
    
    for i in range( len(methodsList) ):
        
        metodo, res, baseline = methodsList[i]
   
        
        if baseline is not None: 
            
            # verifica se e um numero
            if isinstance(baseline, (int, float)):
                plt.plot([0, 11],[baseline,baseline],color='red',linestyle='--', linewidth=1.0,zorder=1, label=metodo + ' (expectation)')#label='Nov and Dec 2019') 
            else:
               plt.plot(np.arange(14), baseline, color=colors[i], linestyle=':', marker=markers[i], linewidth=2.0, label=None , markersize=6)#label='Nov and Dec 2019')             
    
               #value2anotate = baseline[0]
               #plt.annotate('%1.2f' %np.round(value2anotate,2),xy=(0-annotateDist,value2anotate+annotateDist),fontsize='large',rotation=0)
    
               #value2anotate = baseline[1]
               #plt.annotate('%1.2f' %np.round(value2anotate,2),xy=(1-annotateDist,value2anotate+annotateDist),fontsize='large',rotation=0)



        #df_count[['F-measure']].plot(marker='o')]
        plt.plot(np.arange(14), res[ res.columns[1] ], marker=markers[i], label=res.columns[1], linestyle='-', color = colors[i], linewidth=1.0, markersize=6)
    
            # escreve o valor da medida no grafico
            #for j in range( len(res) ):
            #    
            #    if i == 0 or i == 1:
            #        plt.annotate('%1.2f' %np.round(res[ res.columns[i+1] ].iloc[j],2),xy=(j-annotateDist,res[ res.columns[i+1] ].iloc[j]+annotateDist),fontsize='large',rotation=0)

            
        plt.plot([1, 2],[baseline[1],res[metodo].iloc[2]],color='red',linestyle=':', linewidth=1.0,zorder=1, label=None)#label='Nov and Dec 2019') 


    ymin, ymax = plt.ylim()   # return the current xlim
    xmin, xmax = plt.xlim()   # return the current xlim
    
    ymax = 0.9
    ymin = 0.10
    plt.ylim([ymin,ymax]) # set the xlim 
    
    plt.annotate('Expectation at the end of 2019',xy=(0.6, ymax-0.08), xytext=(1.5, ymax-0.05),arrowprops=dict(facecolor='black', shrink=0.05),fontsize='large',rotation=0,bbox=dict(boxstyle="round", fc="w"))
    plt.annotate('}',xy=(-0.2, ymax-0.1), xytext=(-0.2, ymax-0.1),fontsize=50,rotation=90)

    plt.yticks(np.arange(np.round(ymin,2), np.round(ymax,2), 0.02))
    
    #plt.axvline(x=1, color='gray',linestyle='--',linewidth=1.0)
    # adiciona uma area colorida
    plt.axvspan(0, 1, alpha=0.2, color='gray')
    #plt.axhspan(0, 1, 1, 2, alpha=0.2, color='gray') 
    #plt.fill_between(plt.ylim(), plt.ylim(), 1)
    #plt.margins(x=0)

    
    plt.ylabel(r'F-measure',fontsize='large') 
    #plt.xlabel(r'Month',fontsize='large') 
    
    months = ['Nov 2019','Dec 2019','Jan 2020', 'Feb 2020', 'Mar 2020', 'Apr 2020', 'May 2020', 'Jun 2020', 'Jul 2020', 'Aug 2020', 'Sep 2020', 'Oct 2020', 'Nov 2020', 'Dec 2020']
    #plt.xticks( np.arange(0,12), np.arange(1,13) )
    plt.xticks( np.arange(0,14), months)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=45, ha='right',fontsize='large') #rotate x labels #ha is the horizontal alignment of the labels 
    
    #plt.setp( fontsize='large' ) #rotate x labels #ha is the horizontal alignment of the labels 
    plt.yticks(fontsize='large')

    #plt.legend(fontsize='large')
    #plt.legend(bbox_to_anchor=(0., 1.005, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0., fontsize='large')
    plt.legend(bbox_to_anchor=(0., 1.005, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0., fontsize='large')
    #plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
    #plt.legend(loc='best', fontsize='large', ncol=4)

    # cria um título para o gráfico
    # if titulo is not None:
    #     #plt.title(titulo, fontsize='x-large', pad=70)  
    #     plt.title(titulo, fontsize='x-large')  
        
    #     # altura do titulo
    #     #plt.rcParams['axes.titley'] = 1.20    # y is in axes-relative coordinates.
    #     plt.rcParams['axes.titlepad'] = -60  # pad is in points...

    plt.grid(axis='y')
    
    #plt.rcParams['xtick.major.pad']='1'
    
    # salva em uma figura png
    #titulo = titulo.replace(' ','')
    #titulo = titulo.replace(')','')
    #titulo = titulo.replace('(','')
    #plt.savefig(path, format='png', bbox_inches='tight', dpi=500) 

    #plt.show()

    
    
def plota_online(methodsList, baseline = 0.90, annotateDist = 0.01, titulo = None):
       
    #df_count[['countClass0','countClass2']].plot(marker='.', linestyle='-',)
    
    #cria um gráfico de barras com a frequência de cada classe
    #sns.barplot(x="date", hue=['countClass0'], data=df_count)

    markers = ['o',"s",'^','v',"1","<",'*','X']
    colors = ['orange','brown','blue','green','purple','olive','brown']
    styles = ['-','-',':',':','-.','-.']
    
    for i in range( len(methodsList) ):
        
        metodo, res, baseline = methodsList[i]
   
        
        if baseline is not None: 
            
            # verifica se e um numero
            if isinstance(baseline, (int, float)):
                plt.plot([0, 11],[baseline,baseline],color='red',linestyle='--', linewidth=1.0,zorder=1, label=metodo + ' (expectation)')#label='Nov and Dec 2019') 
            else:
               plt.plot(np.arange(14), baseline, color=colors[i], linestyle=':', marker=markers[i], linewidth=2.0, label=metodo + ' (expectation)' , markersize=6)#label='Nov and Dec 2019')             
    
               #value2anotate = baseline[0]
               #plt.annotate('%1.2f' %np.round(value2anotate,2),xy=(0-annotateDist,value2anotate+annotateDist),fontsize='large',rotation=0)
    
               #value2anotate = baseline[1]
               #plt.annotate('%1.2f' %np.round(value2anotate,2),xy=(1-annotateDist,value2anotate+annotateDist),fontsize='large',rotation=0)



        #df_count[['F-measure']].plot(marker='o')]
        plt.plot(np.arange(14), res[ res.columns[2] ], marker=markers[i], label=res.columns[2], linestyle='-', color = colors[i], linewidth=1.0, markersize=6)
    
            # escreve o valor da medida no grafico
            #for j in range( len(res) ):
            #    
            #    if i == 0 or i == 1:
            #        plt.annotate('%1.2f' %np.round(res[ res.columns[i+1] ].iloc[j],2),xy=(j-annotateDist,res[ res.columns[i+1] ].iloc[j]+annotateDist),fontsize='large',rotation=0)
           

        #plt.plot([1, 2],[baseline[1],res[metodo + ' (offline)'].iloc[2]],color='red',linestyle='--', linewidth=1.0,zorder=1, label=None)#label='Nov and Dec 2019') 
        plt.plot([1, 2],[baseline[1],res[metodo + ' (online)'].iloc[2]],color='red',linestyle=':', linewidth=1.0,zorder=1, label=None)#label='Nov and Dec 2019') 



    ymin, ymax = plt.ylim()   # return the current xlim
    xmin, xmax = plt.xlim()   # return the current xlim
    
    ymax = 0.9
    ymin = 0.10
    plt.ylim([ymin,ymax]) # set the xlim 
    
    plt.annotate('Expectation at the end of 2019',xy=(0.6, ymax-0.08), xytext=(1.5, ymax-0.05),arrowprops=dict(facecolor='black', shrink=0.05),fontsize='large',rotation=0,bbox=dict(boxstyle="round", fc="w"))
    plt.annotate('}',xy=(-0.2, ymax-0.1), xytext=(-0.2, ymax-0.1),fontsize=50,rotation=90)

    plt.yticks(np.arange(np.round(ymin,2), np.round(ymax,2), 0.02))
    
    #plt.axvline(x=1, color='gray',linestyle='--',linewidth=1.0)
    # adiciona uma area colorida
    plt.axvspan(0, 1, alpha=0.2, color='gray')
    #plt.axhspan(0, 1, 1, 2, alpha=0.2, color='gray') 
    #plt.fill_between(plt.ylim(), plt.ylim(), 1)
    #plt.margins(x=0)

    
    plt.ylabel(r'F-measure',fontsize='large') 
    #plt.xlabel(r'Month',fontsize='large') 
    
    months = ['Nov 2019','Dec 2019','Jan 2020', 'Feb 2020', 'Mar 2020', 'Apr 2020', 'May 2020', 'Jun 2020', 'Jul 2020', 'Aug 2020', 'Sep 2020', 'Oct 2020', 'Nov 2020', 'Dec 2020']
    #plt.xticks( np.arange(0,12), np.arange(1,13) )
    plt.xticks( np.arange(0,14), months)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=45, ha='right',fontsize='large') #rotate x labels #ha is the horizontal alignment of the labels 
    
    #plt.setp( fontsize='large' ) #rotate x labels #ha is the horizontal alignment of the labels 
    plt.yticks(fontsize='large')

    #plt.legend(fontsize='large')
    #plt.legend(bbox_to_anchor=(0., 1.005, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0., fontsize='large')
    plt.legend(bbox_to_anchor=(0., 1.005, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0., fontsize='large')
    #plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
    #plt.legend(loc='best', fontsize='large', ncol=4)

    # cria um título para o gráfico
    # if titulo is not None:
    #     #plt.title(titulo, fontsize='x-large', pad=70)  
    #     plt.title(titulo, fontsize='x-large')  
        
    #     # altura do titulo
    #     #plt.rcParams['axes.titley'] = 1.20    # y is in axes-relative coordinates.
    #     plt.rcParams['axes.titlepad'] = -60  # pad is in points...

    plt.grid(axis='y')
    
    #plt.rcParams['xtick.major.pad']='1'
    
    # salva em uma figura png
    #titulo = titulo.replace(' ','')
    #titulo = titulo.replace(')','')
    #titulo = titulo.replace('(','')
    #plt.savefig(path, format='png', bbox_inches='tight', dpi=500) 

    #plt.show()
    
    
def plota_online2(methodsList, baseline = 0.90, annotateDist = 0.01, titulo = None):
       
    #df_count[['countClass0','countClass2']].plot(marker='.', linestyle='-',)
    
    #cria um gráfico de barras com a frequência de cada classe
    #sns.barplot(x="date", hue=['countClass0'], data=df_count)

    markers = ['o',"s",'^','v',"1","<",'*','X']
    colors = ['orange','brown','blue','green','purple','olive','brown']
    styles = ['-','-',':',':','-.','-.']
    
    for i in range( len(methodsList) ):
        
        metodo, res, baseline = methodsList[i]
   
        
        if baseline is not None: 
            
            # verifica se e um numero
            if isinstance(baseline, (int, float)):
                plt.plot([0, 11],[baseline,baseline],color='red',linestyle='--', marker=markers[i], linewidth=1.0, markersize=6, zorder=1, label=metodo + ' (expectation)')#label='Nov and Dec 2019') 
            else:
               plt.plot(np.arange(14), baseline, color=colors[i], linestyle=':', marker=markers[i], linewidth=2.0, label=None , markersize=6)#label='Nov and Dec 2019')             
    
               #value2anotate = baseline[0]
               #plt.annotate('%1.2f' %np.round(value2anotate,2),xy=(0-annotateDist,value2anotate+annotateDist),fontsize='large',rotation=0)
    
               #value2anotate = baseline[1]
               #plt.annotate('%1.2f' %np.round(value2anotate,2),xy=(1-annotateDist,value2anotate+annotateDist),fontsize='large',rotation=0)



        #df_count[['F-measure']].plot(marker='o')]
        plt.plot(np.arange(14), res[ res.columns[2] ], marker=markers[i], label=res.columns[2], linestyle='-', color = colors[i], linewidth=1.0, markersize=6)
        plt.plot(np.arange(14), res[ res.columns[1] ], marker=markers[i], label=res.columns[1], dashes=[3, 2,  2, 2, 2, 2], color = colors[i], linewidth=2.0, markersize=6)
    
            # escreve o valor da medida no grafico
            #for j in range( len(res) ):
            #    
            #    if i == 0 or i == 1:
            #        plt.annotate('%1.2f' %np.round(res[ res.columns[i+1] ].iloc[j],2),xy=(j-annotateDist,res[ res.columns[i+1] ].iloc[j]+annotateDist),fontsize='large',rotation=0)
           

        plt.plot([1, 2],[baseline[1],res[metodo + ' (offline)'].iloc[2]],color=colors[i],linestyle=':', linewidth=1.0,zorder=1, label=None)#label='Nov and Dec 2019') 
        plt.plot([1, 2],[baseline[1],res[metodo + ' (online)'].iloc[2]],color=colors[i],linestyle=':', linewidth=1.0,zorder=1, label=None)#label='Nov and Dec 2019') 

    #plt.annotate('{',xy=(1, 0.80), xytext=(1, 0.82),arrowprops=dict(facecolor='black', shrink=0.05),fontsize='large',rotation=90)

    ymin, ymax = plt.ylim()   # return the current xlim
    xmin, xmax = plt.xlim()   # return the current xlim
    
    ymax = 0.9
    ymin = 0.10
    plt.ylim([ymin,ymax]) # set the xlim 
    
    plt.annotate('Expectation at the end of 2019',xy=(0.6, ymax-0.08), xytext=(1.5, ymax-0.05),arrowprops=dict(facecolor='black', shrink=0.05),fontsize='large',rotation=0,bbox=dict(boxstyle="round", fc="w"))
    plt.annotate('}',xy=(-0.2, ymax-0.1), xytext=(-0.2, ymax-0.1),fontsize=50,rotation=90)

    plt.yticks(np.arange(np.round(ymin,2), np.round(ymax,2), 0.02))
    #
    
    #plt.axvline(x=1, color='gray',linestyle='--',linewidth=1.0)
    # adiciona uma area colorida
    plt.axvspan(0, 1, alpha=0.2, color='gray')
    #plt.axhspan(0, 1, 1, 2, alpha=0.2, color='gray') 
    #plt.fill_between(plt.ylim(), plt.ylim(), 1)
    #plt.margins(x=0)

    
    plt.ylabel(r'F-measure',fontsize='large') 
    #plt.xlabel(r'Month',fontsize='large') 
    
    months = ['Nov 2019','Dec 2019','Jan 2020', 'Feb 2020', 'Mar 2020', 'Apr 2020', 'May 2020', 'Jun 2020', 'Jul 2020', 'Aug 2020', 'Sep 2020', 'Oct 2020', 'Nov 2020', 'Dec 2020']
    #plt.xticks( np.arange(0,12), np.arange(1,13) )
    plt.xticks( np.arange(0,14), months)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=45, ha='right',fontsize='large') #rotate x labels #ha is the horizontal alignment of the labels 
    
    #plt.setp( fontsize='large' ) #rotate x labels #ha is the horizontal alignment of the labels 
    plt.yticks(fontsize='large')

    #plt.legend(fontsize='large')
    #plt.legend(bbox_to_anchor=(0., 1.005, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0., fontsize='large')
    plt.legend(bbox_to_anchor=(0., 1.005, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0., fontsize='large')
    #plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
    #plt.legend(loc='best', fontsize='large', ncol=4)

    # cria um título para o gráfico
    # if titulo is not None:
    #     #plt.title(titulo, fontsize='x-large', pad=70)  
    #     plt.title(titulo, fontsize='x-large')  
        
    #     # altura do titulo
    #     #plt.rcParams['axes.titley'] = 1.20    # y is in axes-relative coordinates.
    #     plt.rcParams['axes.titlepad'] = -60  # pad is in points...

    plt.grid(axis='y')
    
    #plt.rcParams['xtick.major.pad']='1'
    
    # salva em uma figura png
    #titulo = titulo.replace(' ','')
    #titulo = titulo.replace(')','')
    #titulo = titulo.replace('(','')
    #plt.savefig(path, format='png', bbox_inches='tight', dpi=500) 

    #plt.show()

def return_dfMethod_toPlot( df_scores, metodo, isOnline = None, medidaPrincipal = None ):

    df_method = df_scores[ df_scores['metodo'] == metodo ] 

    # transforma a tabela para que cada metodo fique em uma linha e cada base em uma coluna
    df_method = df_method.pivot(index='month', columns='base_dados', values=medidaPrincipal)
    df_method = df_method.reset_index()
    
    df_method = df_method.sort_values(['month'], ascending=True)   
    
    # arruma os valores de 2019
    baseline = np.zeros(14)
    baseline[2:] = np.nan
    baseline[0] = df_method['2019 (offline)-TrainOct'].iloc[10]
    baseline[1] = df_method['2019 (offline)-TrainNov'].iloc[11]    
    
    
    if isOnline:
        
        # arruma os valores de 2020
        res = {}
        res['month'] = list(np.arange(1,15))
        res[metodo + ' (offline)'] = np.concatenate([np.zeros(2)*np.nan,df_method['2020 (offline)-TrainDec'].values])
        res[metodo + ' (online)'] = np.concatenate([np.zeros(2)*np.nan,df_method['2020-TrainDec'].values])
        #res['Covid-19 (online)'] = np.concatenate([np.zeros(2)*np.nan,df_method['2020 (Covid-19)-TrainDec'].values])
        #res['Elections (online)'] = np.concatenate([np.zeros(2)*np.nan,df_method['2020 (elections)-TrainDec'].values])
        df_method = pd.DataFrame(res)
            
    else:
        
        # arruma os valores de 2020
        res = {}
        res['month'] = list(np.arange(1,15))
        res[metodo] = np.concatenate([np.zeros(2)*np.nan,df_method['2020 (offline)-TrainDec'].values])
        #res['Covid-19'] = np.concatenate([np.zeros(2)*np.nan,df_method['2020 (Covid-19) (offline)-TrainDec'].values])
        #res['Elections'] = np.concatenate([np.zeros(2)*np.nan,df_method['2020 (elections) (offline)-TrainDec'].values])
        df_method = pd.DataFrame(res)
        
    return df_method, baseline

        

if __name__ == "__main__":

    # Resultados informados pelo usuário (F-measure Macro)
    resultados = {
        'Perceptron': [
            0.5064102564102564, 0.39488592864798655, 0.30900896269257794,
            0.2792484015243485, 0.2902939714820903, 0.318907066959015,
            0.28595722351249186, 0.2846623563218391, 0.3254309562692796,
            0.3566808526594049, 0.37652550155745546, 0.2788889548341148,
            0.27125851269662554, 0.25255382681125255
        ],
        'SGDClassifier': [
            0.3976318325760705, 0.35275813295615277, 0.28140026126351486,
            0.2405786586917612, 0.2650077911016908, 0.27641379310344827,
            0.2543448744896174, 0.27010960818463403, 0.2751145379371997,
            0.3273521969174143, 0.3498037701859358, 0.25073288968713664,
            0.23217251391688704, 0.22300254093732355
        ],
        'PassiveAggressiveClassifier': [
            0.5064102564102564, 0.39488592864798655, 0.3149919519371739,
            0.2758226358445922, 0.2910862523843142, 0.3319672131147541,
            0.29146429967063026, 0.2875355719977234, 0.3175905606971345,
            0.35771154794617066, 0.3839758125472411, 0.27747844827586204,
            0.26581058335128505, 0.24965587159531374
        ],
        'MultinomialNB': [
            0.48423224285293254, 0.6135895152054969, 0.5258929131881238,
            0.5159410893734833, 0.5097352752438896, 0.49186463480631404,
            0.5345706831119544, 0.5307603496618836, 0.5273278393482997,
            0.44467127075822727, 0.3698477921175994, 0.49070945945945943,
            0.5430820642516209, 0.5081559505522639
        ]
    }

    from pathlib import Path
    result_dir = Path(__file__).resolve().parents[2] / 'result' / 'comparacao_modelos'
    result_dir.mkdir(parents=True, exist_ok=True)
    pathFig = str(result_dir / 'resultados_4_modelos.png')
    annotateDist = 0.005

    # Mantém o mesmo formato esperado por plota_offline():
    # baseline = Nov/Dez 2019; res = Jan-Dez 2020.
    methodsList = []

    for metodo, valores in resultados.items():
        valores = np.asarray(valores, dtype=float)

        if len(valores) != 14:
            raise ValueError(f'{metodo}: esperado vetor com 14 resultados, recebido {len(valores)}.')

        baseline = np.full(14, np.nan)
        baseline[0] = valores[0]
        baseline[1] = valores[1]

        res = pd.DataFrame({
            'month': np.arange(1, 15),
            metodo: np.concatenate([np.array([np.nan, np.nan]), valores[2:]])
        })

        methodsList.append([metodo, res, baseline])

    plt.figure(figsize=(10, 8))
    plota_offline(methodsList, annotateDist=annotateDist)
    plt.savefig(pathFig, format='png', bbox_inches='tight', dpi=500)
    print(f'Figura salva em: {pathFig}')

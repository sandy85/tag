"""
API pour définir les tags pertinents à une question posée
"""

from flask import Flask, render_template, flash, request
import pickle
import numpy as np
import pandas as pd
import os
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer

app = Flask(__name__)

@app.route('/tags')

"""
Chargement du modèle et du dictionnaire de stopwords
"""
#chargement des stopwords mis à jour
stopwords = pickle.load(open('stopwords.pkl','rb'))

#chargement du modèle
modele = pickle.load(open('model_rf30.pkl','rb'))

#chargement de la liste des tags
cols = pickle.load(open('tags_rf30.pkl', 'rb'))

"""
Déclaration des fonctions nécessaires au nettoyage du texte 
"""
def question():
	return render_template('question.html')

#lemmatisation des mots
def stemming(text):
    stemmer = EnglishStemmer()
    stemtext = ""
    for word in text.split():
        stem = stemmer.stem(word)
        stemtext += stem
        stemtext += " "
    stemtext = stemtext.strip()
    return stemtext

@app.route('/calcul', methods=['POST'])

"""
Fonction transformant le texte en mots et application du modèle pour déterminer les thèmes abordés dans la question
"""
def calcul():
	# récupération de la question posée dans le formulaire HTML 
	quest = request.values.get('question')
	print(quest)
    
    # question en minuscules
    quest = quest.lower()
    
    # Gestion des noms particuliers
    dico = {'c#':'csharp',
        'c\+\+':'cpplus',
        '\.net':'dotnet',
        'internet-explorer-7':'internetexplorerseven',
        'language\-agnostic': 'languageagnostic',
        'sql-server' : 'sqlserver'
        }
    for key in dico.keys():
        quest=quest.replace(key, dico[key]) 
    
    #suppression des balises
    cleanr = re.compile('<.*?>')
    quest = re.sub(cleanr, ' ', str(quest))
    
    #suppression de la ponctuation
    quest = re.sub(r'[?|!|\'|"|#]',r' ',quest)
    quest = re.sub(r'[.|,|)|(|\|/]',r' ',quest)
    quest = quest.strip()
    quest = quest.replace("\n"," ")
    
    #on ne conserve que les caractères alphanumériques
    quest = quest.replace('[^\w\s]', ' ')
    
    #suppression des stopwords
    quest1  = ' '.join([word for word in quest.split() if word not in stopwords])

    #lemmatisation 
    quest1 = stemming(quest1)
    
	return resultat(quest1,tags)

@app.route('/resultat')

"""
Fonction qui retourne le résultat du modèle
"""
def resultat(parameter1,parameter2):	
	return render_template('resultat.html', valeur_une = parameter1,valeur_deux=parameter2)

if __name__ == "__main__":
    app.run()
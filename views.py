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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
app = Flask(__name__)

path = "P:\openclassrooms\P6_categorie_question\API\\tag"

"""
Chargement du modèle et du dictionnaire de stopwords
"""

#chargement des stopwords mis à jour
stopwords = pickle.load(open(path+'\stopwords.pkl','rb'))

#chargement du modèle
modele = pickle.load(open(path+'\model_REGLOG30.pkl','rb'))

#chargement de la liste des tags
cols = pickle.load(open(path+'\\tags_REGLOG30.pkl','rb'))

@app.route('/')


def tag():
    return render_template('question.html')

"""
Déclaration des fonctions nécessaires au nettoyage du texte
"""


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



"""
Fonction transformant le texte en mots et application du modèle pour déterminer les thèmes abordés dans la question
"""

@app.route('/calcul', methods=['POST'])

def calcul():
    # récupération de la question posée dans le formulaire HTML
    quest = request.values.get('question')

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
    quest = re.sub(r'[?|!|"|#]',r' ',quest)
    print('quest 1 :',quest )
    quest = re.sub(r'[.|,|)|(|\\|:|/]',r' ',quest)
    quest = quest.strip()
    quest = quest.replace("\\n"," ")
    
    #on ne conserve que les caractères alphanumériques
    quest = quest.replace('[^\w\s]', ' ')
    print('quest apres nettoyage:',quest )
    #suppression des stopwords
    quest1  = ' '.join([word for word in quest.split() if word not in stopwords])

    #lemmatisation
    quest1 = stemming(quest1)
    print('quest apres stemming:',quest1)

    #application du modèle Régression Logistique

    quest1 = pd.Series(quest1)
    print(quest1)

    pred=modele.predict_proba(quest1)
    print(pred)

    #conversion de la matrice en liste
    liste = [item for sublist in pred.tolist() for item in sublist]

    #création df
    df = list(zip(cols,liste))
    df = pd.DataFrame(df,columns = ['tag','proba'])

    #conservation des tags dont la proba est supérieure au meilleur seuil(0.28)
    df = df[df['proba'] > 0.28]
    # tri par proba décroissante
    df.sort_values(by='proba',ascending=False)
    res=df.to_html()
    return render_template("resultat.html",question=quest,table=res)



if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)
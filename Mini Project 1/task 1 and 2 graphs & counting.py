import math
import os
import pandas as pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == '__main__':
    # names of the classes
    x = ['business', 'entertainment', 'politics', 'sport', 'tech']

    # find total number of files for each class
    y = [
        len([name for name in os.listdir('BBC/business')]),
        len([name for name in os.listdir('BBC/entertainment')]),
        len([name for name in os.listdir('BBC/politics')]),
        len([name for name in os.listdir('BBC/sport')]),
        len([name for name in os.listdir('BBC/tech')]),
    ]

    # plot the graph
    plt.bar(x, y)

    plt.savefig('BBC-distribution.pdf')

    drugs = pandas.read_csv('drug200.csv')

    print(drugs['Drug'])

    classes = {
        'drugA': 0,
        'drugB': 0,
        'drugC': 0,
        'drugX': 0,
        'drugY': 0,
    }

    # iterate through and tally drug class instances
    for drugType in drugs['Drug']:
        classes[drugType] += 1

    plt.clf()

    plt.bar(classes.keys(), classes.values())

    plt.savefig('drug-distribution.pdf')

    # for i in range(len(drugs['Sex'])):
    #    print(drugs['Sex'][i])

    # BBC text occurrence counting

    bbc = datasets.load_files('BBC', encoding="latin1")

    # These are separate folders as I cannot seem to get datasets.load_files to load anything if it's pointed at a
    # directory containing files. (IE: 'BBC/business' contains files directly and doesn't work.)
    # as a workaround, I have copies of the dataset (with an extra folder layer to allow this to work properly
    # if someone knows how to make this work without the duplicates, I'm happy to change it. I could have modified our
    # bbc dataset directly to add these extra folders, but that would potentially interfere with your existing parts
    business = datasets.load_files('business', encoding="latin1")
    entertainment = datasets.load_files("entertainment", encoding="latin1")
    politics = datasets.load_files("politics", encoding="latin1")
    sport = datasets.load_files("sport", encoding="latin1")
    tech = datasets.load_files("tech", encoding="latin1")

    bbcVectorizer = CountVectorizer()
    businessVectorizer = CountVectorizer()
    entertainmentVectorizer = CountVectorizer()
    politicsVectorizer = CountVectorizer()
    sportVectorizer = CountVectorizer()
    techVectorizer = CountVectorizer()

    bbcFit = bbcVectorizer.fit_transform(bbc.data)
    businessFit = businessVectorizer.fit_transform(business.data)
    entertainmentFit = entertainmentVectorizer.fit_transform(entertainment.data)
    politicsFit = politicsVectorizer.fit_transform(politics.data)
    sportFit = sportVectorizer.fit_transform(sport.data)
    techFit = techVectorizer.fit_transform(tech.data)

    # sum up the occurences across texts within all classes
    bbcArray = bbcFit.toarray().sum(axis=0)

    # create a dictionary with the frequency of each word
    bbcDict = {}
    for i, word in enumerate(bbcVectorizer.vocabulary_.keys()):
        bbcDict[word] = bbcArray[i]

    businessArray = businessFit.toarray().sum(axis=0)

    businessDict = {}
    for i, word in enumerate(businessVectorizer.vocabulary_.keys()):
        businessDict[word] = businessArray[i]

    entertainmentArray = entertainmentFit.toarray().sum(axis=0)

    entertainmentDict = {}
    for i, word in enumerate(entertainmentVectorizer.vocabulary_.keys()):
        entertainmentDict[word] = entertainmentArray[i]

    politicsArray = politicsFit.toarray().sum(axis=0)

    politicsDict = {}
    for i, word in enumerate(politicsVectorizer.vocabulary_.keys()):
        politicsDict[word] = politicsArray[i]

    sportArray = sportFit.toarray().sum(axis=0)

    sportDict = {}
    for i, word in enumerate(sportVectorizer.vocabulary_.keys()):
        sportDict[word] = sportArray[i]

    techArray = techFit.toarray().sum(axis=0)

    techDict = {}
    for i, word in enumerate(techVectorizer.vocabulary_.keys()):
        techDict[word] = techArray[i]

    wordTokens = 0

    corpusSingle = 0

    businessZero = 0
    entertainmentZero = 0
    politicsZero = 0
    sportZero = 0
    techZero = 0

    businessTokens = 0
    entertainmentTokens = 0
    politicsTokens = 0
    sportTokens = 0
    techTokens = 0

    for word in bbcVectorizer.vocabulary_.keys():
        # Count Total tokens
        wordTokens += bbcDict[word]

        if bbcDict[word] == 1:
            corpusSingle += 1

        # if class contains a word, count the total tokens, else count it as an absence of that word
        if businessDict.__contains__(word):
            businessTokens += businessDict[word]
        else:
            businessZero += 1

        if entertainmentDict.__contains__(word):
            entertainmentTokens += entertainmentDict[word]
        else:
            entertainmentZero += 1

        if politicsDict.__contains__(word):
            politicsTokens += politicsDict[word]
        else:
            politicsZero += 1

        if sportDict.__contains__(word):
            sportTokens += sportDict[word]
        else:
            sportZero += 1

        if techDict.__contains__(word):
            techTokens += techDict[word]
        else:
            techZero += 1

    print("Total Vocabulary Size: " + str(len(bbcDict.values())))
    print("Total Tokens: " + str(wordTokens))
    print("Total tokens in business: " + str(businessTokens))
    print("Total tokens in entertainment: " + str(entertainmentTokens))
    print("Total tokens in politics: " + str(politicsTokens))
    print("Total tokens in sport: " + str(sportTokens))
    print("Total tokens in tech: " + str(techTokens))
    print("Single occurrence in the entire corpus: " +
          str(corpusSingle) + " / " + str(round(100 * corpusSingle/len(bbcDict.values()), 2)) + "%")
    print("Words in corpus vocabulary but not in business vocabulary: " +
          str(businessZero) + " / " + str(round(100 * businessZero/len(bbcDict.values()), 2)) + "%")
    print("Words in corpus vocabulary but not in entertainment vocabulary: " +
          str(entertainmentZero) + " / " + str(round(100 * (entertainmentZero/len(bbcDict.values())), 2)) + "%")
    print("Words in corpus vocabulary but not in politics vocabulary: " +
          str(politicsZero) + " / " + str(round(100 * politicsZero/len(bbcDict.values()), 2)) + "%")
    print("Words in corpus vocabulary but not in sport vocabulary: " +
          str(sportZero) + " / " + str(round(100 * sportZero/len(bbcDict.values()), 2)) + "%")
    print("Words in corpus vocabulary but not in tech vocabulary: " +
          str(techZero) + " / " + str(round(100 * techZero/len(bbcDict.values()), 2)) + "%")
    print("Fave Word - cahoot: Natural log = " + str(round(math.log(bbcDict['cahoot']/len(bbcDict.values())), 2)))
    print("Fave Word - garfield: Natural log = " + str(round(math.log(bbcDict['garfield'] / len(bbcDict.values())), 2)))

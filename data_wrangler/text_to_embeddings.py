import sys
sys.path.append('/datadrive/nlp/jasper/w2v/godin/word2vec_twitter_model/')
from word2vecReader import Word2Vec

if __name__ == "__main__":

    model_path = "/datadrive/nlp/jasper/w2v/godin/word2vec_twitter_model/word2vec_twitter_model.bin"
    print("Loading the model, this can take some time...")
    model = Word2Vec.load_word2vec_format(model_path, binary=True)
    print(("The vocabulary size is: "+str(len(model.vocab))))
    print("Vector for 'Shubham': " + str(model['Shubham']))
    print("Embedding dimension: " + str(len(model['Shubham'])))
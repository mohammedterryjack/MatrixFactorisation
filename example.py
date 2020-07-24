############   NATIVE IMPORTS  ###########################
from json import dumps 
############ INSTALLED IMPORTS ###########################
from spacy import load
############   LOCAL IMPORTS   ###########################
from src.matrix_factorisation import MatrixFactorisationClassifier
##########################################################

spacy_word_embeddings = load('en_core_web_sm')

classifier = MatrixFactorisationClassifier(
    path_to_sparse_matrix="sparse_matrix.csv",
    embedding_method=lambda word:spacy_word_embeddings(word).vector
)

results = classifier.predict(
    new_data_to_classify=[
        "i like you",
        "how are you",
        "how do i get to Chancery Lane"
    ]
)

print(dumps(results,indent=4))
############   NATIVE IMPORTS  ###########################
from typing import List, Dict, Tuple
############ INSTALLED IMPORTS ###########################
from pandas import read_csv, DataFrame
from numpy.linalg import pinv 
############   LOCAL IMPORTS   ###########################
##########################################################

class MatrixFactorisationClassifier:
    """
    Matrix Factorisation is used to convert a sparse matrix into a dense matrix
    in this way the output labels (columns) are embedded into the same space as the input data (rows)
    the embedded labels can then be compared directly to the embedded input data for classification
    an advantage is that this classifier can handle multiclass classification tasks
    """
    def __init__(
        self,
        path_to_sparse_matrix:str,
        embedding_method:callable
    ) -> None:

        self.embedding_method = embedding_method
        self.dense_output_vectors = self.get_dense_output_vectors_via_matrix_factorisation(
            sparse_matrix=read_csv(path_to_sparse_matrix,index_col='input'),
        )

    def predict(self, new_data_to_classify:List[str], top_n:int=3) -> Dict[str,List[Tuple[str,str]]]:
        """
        dot product of dense vectors obtains dense matrix of predicted relations between input and outputs
        the highest relating labels are returned for each input data 
        """
        new_dense_input_vectors = self.get_dense_input_vectors_using_a_known_embedding_method(
            input_labels=new_data_to_classify,
            known_embedding_method = self.embedding_method   
        )
        dense_prediction_matrix = new_dense_input_vectors.T @ self.dense_output_vectors
        dense_prediction_matrix_normalised = dense_prediction_matrix.div(
            dense_prediction_matrix.sum(axis=1), 
            axis=0
        )
        return dict(
            map(
                lambda new_data: (
                    new_data,
                    list(
                        dense_prediction_matrix_normalised.T[new_data].sort_values(
                            ascending=False
                        ).items()
                    )[:top_n]
                ),
                dense_prediction_matrix_normalised.index
            )
        )

    def get_dense_output_vectors_via_matrix_factorisation(
        self,
        sparse_matrix:DataFrame, 
    ) -> Tuple[List[float],List[float]]:
    
        dense_input_vectors = MatrixFactorisationClassifier.get_dense_input_vectors_using_a_known_embedding_method(
            input_labels=sparse_matrix.index,
            known_embedding_method = self.embedding_method   
        )
        return MatrixFactorisationClassifier.get_compatible_dense_output_vectors_using_pseudoinverse(
            dense_input_vectors=dense_input_vectors,
            sparse_matrix=sparse_matrix
        )

    @staticmethod
    def get_dense_input_vectors_using_a_known_embedding_method(
        input_labels:List[str],
        known_embedding_method:callable
    ) -> DataFrame:

        return DataFrame(
            dict(
                map(
                    lambda input_label: (
                        input_label,
                        known_embedding_method(input_label)
                    ),
                    input_labels
                )
            )
        )

    @staticmethod
    def get_compatible_dense_output_vectors_using_pseudoinverse(
        dense_input_vectors:List[float],
        sparse_matrix:DataFrame
    ) -> DataFrame:

        dense_output_vectors = DataFrame(
            pinv(dense_input_vectors.T) @ sparse_matrix
        )
        dense_output_vectors.columns = sparse_matrix.columns
        return dense_output_vectors


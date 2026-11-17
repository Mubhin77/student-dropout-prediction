
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


from src.config import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES
)


def build_preprocessor():
    """
    Build preprocessing transformations.

    Categorical features:
        OneHotEncoder

    Numerical features:
        StandardScaler
    """

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore"
    )

    numerical_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_transformer,
                CATEGORICAL_FEATURES
            ),
            (
                "numerical",
                numerical_transformer,
                NUMERICAL_FEATURES
            )
        ],
        remainder="drop"
    )

    return preprocessor

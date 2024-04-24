"""
environment.py
Ian Kollipara <ikollipara2@huskers.unl.edu>

Environment Manager for Maize-ML.
"""

# Imports
from typing import Generator, Self, Union
import pandas as pd
import numpy as np


class Environment:
    """
    Environment.

    This class defines the environment manager for Maize-ML. The
    environment manager is responsible for loading and managing
    the data for the Maize-ML models.

    Maize-ML is structured around the sourceEnvironments of the data, which
    are used to structure the weather, genotype, and phenotype data for training,
    validation, and testing. The environment manager is responsible for loading
    and managing the lists of sourceEnvironments.
    """

    def __init__(self, sourceEnvironments: list[str]) -> None:
        """ Initialize the Environment.

        Args:
            sourceEnvironments (list[str]): The list of sourceEnvironments to load.
        """
        self.sourceEnvironments = sourceEnvironments


    def shuffle(self) -> Self:
        """ Shuffle the data.

        This method shuffles the data in the environment manager.
        """
        np.random.shuffle(self.data)

        return self

    def split(self, *, splits: Union[tuple[float, float], tuple[float, float, float]]) -> Self:
        """
        Split the data.

        This method splits the data in the environment manager into training,
        validation, and testing sets. The splits are specified by the splits
        parameter. If the splits parameter has two elements, the data is split
        into training and test sets. If the splits parameter has three elements,
        the data is split into training, validation, and test sets. These splits
        are available as the 'train', 'val', and 'test' attributes of the
        environment manager.

        Args:
            splits (Union[tuple[float, float], tuple[float, float, float]]): The splits for the data.

        Returns:
            Self: The environment manager.
        """

        if len(splits) == 2:
            self.train, self.test = self._split_data(splits)
        elif len(splits) == 3:
            self.train, self.val, self.test = self._split_data(splits)

        return self

    def _split_data(self, splits: tuple[float, float, float]) -> tuple[pd.DataFrame]:
        """ Split the data.

        This method splits the data in the environment manager into training,
        validation, and testing sets. The splits are specified by the splits
        parameter. If the splits parameter has two elements, the data is split
        into training and test sets. If the splits parameter has three elements,
        the data is split into training, validation, and test sets.

        Args:
            splits (tuple[float, float, float]): The splits for the data.

        Returns:
            tuple[pd.DataFrame]: The training, validation, and testing sets.
        """

        if len(splits) == 2:
            train_idx = int(splits[0] * len(self.data))
            return self.data[:train_idx], self.data[train_idx:]
        elif len(splits) == 3:
            train_idx = int(splits[0] * len(self.data))
            val_idx = int(splits[1] * len(self.data))
            return self.data[:train_idx], self.data[train_idx:val_idx], self.data[val_idx:]


    @classmethod
    def load_from_path(cls, path: str) -> 'Environment':
        """ Load the data.


        This method loads the data for the environment manager from the given path.

        Args:
            path (str): The path to the data.

        Returns:
            Environment: The environment manager.

        """

        with open(path, 'r') as f:
            return cls(f.readlines())


    def save_to_path(self, path: str) -> None:
        """ Save the data.

        This method saves the data for the environment manager to the given path.

        Args:
            path (str): The path to save the data to.
        """

        with open(path, 'w') as f:
            f.writelines(self.sourceEnvironments)



    def __iter__(self) -> Generator[tuple[str, pd.DataFrame], None, None]:
        """ Iterate over the data.

        This method allows the environment manager to be used as an iterator.

        Yields:
            tuple[str, pd.DataFrame]: The sourceEnvironment and data.
        """
        for env in self.sourceEnvironments:
            yield env, self.data[env]

    def __getitem__(self, key: str) -> pd.DataFrame:
        """ Get the data for a sourceEnvironment.

        Args:
            key (str): The sourceEnvironment to get the data for.

        Returns:
            pd.DataFrame: The data for the sourceEnvironment.
        """

        return self.data[key]

    def __len__(self) -> int:
        """ Get the number of sourceEnvironments.

        Returns:
            int: The number of sourceEnvironments.
        """

        return len(self.sourceEnvironments)

    def __repr__(self) -> str:
        """ Get the string representation of the environment manager.

        Returns:
            str: The string representation of the environment manager.
        """

        return f'Environment({self.sourceEnvironments})'

    def __str__(self) -> str:
        """ Get the string representation of the environment manager.

        Returns:
            str: The string representation of the environment manager.
        """

        return f'Environment({self.sourceEnvironments})'

    def __contains__(self, key: str) -> bool:
        """ Check if a sourceEnvironment is in the environment manager.


        Args:
            key (str): The sourceEnvironment to check for.

        Returns:
            bool: True if the sourceEnvironment is in the environment manager, False otherwise.
        """

        return key in self.sourceEnvironments

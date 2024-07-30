import torch
import torch.utils.data as data_utils
import pandas as pd
from transformers import (
    DataCollatorWithPadding,
    RobertaTokenizer,
)
from transformers import PreTrainedTokenizer
from datasets import Dataset  # huggingface datasets
from typing import *

from problems.prompting import PromptBuilder

from datasets.utils.logging import disable_progress_bar

disable_progress_bar()


class DataProcessor:
    """
    Base class for all Bayesian optimization datasets (always regression)
    """

    def __init__(
        self,
        prompt_builder: PromptBuilder,
        num_outputs: int,
        tokenizer: PreTrainedTokenizer,
    ):
        self.prompt_builder = prompt_builder
        self.num_outputs = num_outputs
        self.tokenizer = tokenizer
        # To be defined in subclasses
        self.target_col = None
        self.obj_str = None

    def get_dataloader(
        self,
        pandas_dataset: pd.DataFrame,
        batch_size=16,
        max_seq_len=512,
        shuffle=False,
        append_eos=True,
    ) -> data_utils.DataLoader:
        dataset = Dataset.from_pandas(pandas_dataset)

        def tokenize(row):
            prompt = self.prompt_builder.get_prompt(row["SMILES"], self.obj_str)
            if append_eos:
                prompt += self.tokenizer.eos_token
            out = self.tokenizer(prompt, truncation=True, max_length=max_seq_len)
            out["labels"] = self._get_targets(row)
            return out

        dataset = dataset.map(
            tokenize, remove_columns=self._get_columns_to_remove(), num_proc=4
        )

        return data_utils.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=DataCollatorWithPadding(self.tokenizer),
        )

    def _get_targets(self, row: Union[pd.Series, dict]) -> torch.Tensor:
        """
        Arguments:
        ----------
        row: pd.Series containing one entry or a dictionary
            A single row of the raw dataset.

        Returns:
        --------
        targets: torch.Tensor
            Regression target(s). Shape (self.num_outputs,).
        """
        if isinstance(self.target_col, list):
            return [row[col] for col in self.target_col]
        else:
            return [row[self.target_col]]

    def _get_columns_to_remove(self) -> List[str]:
        """
        Returns:
        --------
        cols: list of strs
            Columns to remove from the dataset
        """
        raise NotImplementedError


class RedoxDataProcessor(DataProcessor):
    """
    RangeIndex: 1407 entries, 0 to 1406
    Data columns (total 7 columns):
    #   Column                 Non-Null Count  Dtype
    --  ------                 --------------  -----
    0   Entry Number           1407 non-null   int64
    1   File Name              1407 non-null   object
    2   SMILES                 1407 non-null   object
    3   IUPAC Name             1407 non-null   object
    4   Ered                   1407 non-null   float64
    5   HOMO                   1407 non-null   float64
    6   Gsol                   1407 non-null   float64
    7   Absorption Wavelength  1407 non-null   float64
    dtypes: float64(4), int64(1), object(2)
    memory usage: 77.1+ KB

    Objective: Minimize Ered (secondary objective: minimize Gsol)
    """

    def __init__(self, prompt_builder, tokenizer):
        super().__init__(
            prompt_builder=prompt_builder, num_outputs=1, tokenizer=tokenizer
        )
        self.target_col = "Ered"
        self.obj_str = "redox potential"

    def _get_columns_to_remove(self) -> List[str]:
        return [
            "Entry Number",
            "File Name",
            "SMILES",
            "IUPAC Name",
            "HOMO",
            "Ered",
            "Gsol",
            "Absorption Wavelength",
        ]


class SolvationDataProcessor(DataProcessor):
    """
    RangeIndex: 1407 entries, 0 to 1406
    Data columns (total 7 columns):
    #   Column                 Non-Null Count  Dtype
    --  ------                 --------------  -----
    0   Entry Number           1407 non-null   int64
    1   File Name              1407 non-null   object
    2   SMILES                 1407 non-null   object
    3   IUPAC Name             1407 non-null   object
    4   Ered                   1407 non-null   float64
    5   HOMO                   1407 non-null   float64
    6   Gsol                   1407 non-null   float64
    7   Absorption Wavelength  1407 non-null   float64
    dtypes: float64(4), int64(1), object(2)
    memory usage: 77.1+ KB

    Objective: Minimize Ered (secondary objective: minimize Gsol)
    """

    def __init__(self, prompt_builder, tokenizer):
        super().__init__(
            prompt_builder=prompt_builder, num_outputs=1, tokenizer=tokenizer
        )
        self.target_col = "Gsol"
        self.obj_str = "solvation energy"

    def _get_columns_to_remove(self) -> List[str]:
        return [
            "Entry Number",
            "File Name",
            "SMILES",
            "IUPAC Name",
            "HOMO",
            "Ered",
            "Gsol",
            "Absorption Wavelength",
        ]


class KinaseDockingDataProcessor(DataProcessor):
    """
    Three datasets (10k, 50k, HTS) with same structure.

    RangeIndex:
        10k: 10,449 entries, 0 to 10448
        50k: 49,706 entries, 0 to 49,705
        HTS: 2,104,318 entries, 0 to 2,104,317

    Data columns (total 2 columns):
    #   Column                 Dtype
    --  ------                 -----
    0   SMILES                 object
    1   score                  float64

    dtypes: float64(1), object(1)

    Objective: Minimize the score
    """

    def __init__(self, prompt_builder, tokenizer):
        super().__init__(
            prompt_builder=prompt_builder, num_outputs=1, tokenizer=tokenizer
        )
        self.target_col = "score"
        self.obj_str = "docking score"

    def _get_targets(self, row: Union[pd.Series, dict]) -> List[float]:
        return [row["score"]]

    def _get_columns_to_remove(self) -> List[str]:
        return ["SMILES", "score"]


class AmpCDockingDataProcessor(DataProcessor):
    """
    RangeIndex: 96,214,206 entries, 0 to 96,214,205

    Data columns (total 2 columns):
    #   Column                 Dtype
    --  ------                 -----
    0   SMILES                 object
    1   score                  float64

    dtypes: float64(1), object(1)

    Objective: Minimize the score
    """

    def __init__(self, prompt_builder, tokenizer):
        super().__init__(
            prompt_builder=prompt_builder, num_outputs=1, tokenizer=tokenizer
        )
        self.target_col = "score"
        self.obj_str = "docking score"

    def _get_columns_to_remove(self) -> List[str]:
        return ["SMILES", "score"]


class D4DockingDataProcessor(DataProcessor):
    """
    RangeIndex: 116,241,184 entries, 0 to 116,241,183

    Data columns (total 2 columns):
    #   Column                 Dtype
    --  ------                 -----
    0   SMILES                 object
    1   score                  float64

    dtypes: float64(1), object(1)

    Objective: Minimize the score
    """

    def __init__(self, prompt_builder, tokenizer):
        super().__init__(
            prompt_builder=prompt_builder, num_outputs=1, tokenizer=tokenizer
        )
        self.target_col = "score"
        self.obj_str = "docking score"

    def _get_columns_to_remove(self) -> List[str]:
        return ["SMILES", "score"]


class PhotovoltaicsPCEDataProcessor(DataProcessor):
    """
    RangeIndex: 2,320,648 entries, 0 to 2,232,647

    Data columns (total 2 columns):
    #   Column                 Dtype
    --  ------                 -----
    0   SMILES                 object
    1   pce                    float64

    dtypes: float64(1), object(1)

    Objective: Maximize the PCE
    """

    def __init__(self, prompt_builder, tokenizer):
        super().__init__(
            prompt_builder=prompt_builder, num_outputs=1, tokenizer=tokenizer
        )
        self.target_col = "pce"
        self.obj_str = "power conversion efficiency"

    def _get_columns_to_remove(self) -> List[str]:
        return ["SMILES", "pce"]


class LaserEmitterDataProcessor(DataProcessor):
    """
    RangeIndex: 182,858 entries, 0 to 182,857

    Data columns (total 2 columns):
    #   Column                              Dtype
    --  ------                              -----
    0   SMILES                              object
    1   Fluorescence Oscillator Strength    float64
    2   Electronic Gap                      float64

    dtypes: float64(2), object(1)

    Objective: Maximize the Fluorescence Oscillator Strength (secondary objective: maximize the Electronic Gap)
    """

    def __init__(self, prompt_builder, tokenizer):
        super().__init__(
            prompt_builder=prompt_builder, num_outputs=1, tokenizer=tokenizer
        )
        self.target_col = "Fluorescence Oscillator Strength"
        self.obj_str = "fluorescence oscillator strength"

    def _get_columns_to_remove(self) -> List[str]:
        return ["SMILES", "Fluorescence Oscillator Strength", "Electronic Gap"]


class PhotoswitchDataProcessor(DataProcessor):
    """
    RangeIndex: 392 entries, 0 to 391

    Data columns (total 2 columns):
    #   Column                              Dtype
    --  ------                              -----
    0   SMILES                              object
    1   Pi-Pi* Transition Wavelength        float64

    dtypes: float64(1), object(1)

    Objective: Maximize the Pi–Pi* Transition Wavelength of the E isomer
    """

    def __init__(self, prompt_builder, tokenizer):
        super().__init__(
            prompt_builder=prompt_builder, num_outputs=1, tokenizer=tokenizer
        )
        self.target_col = "Pi-Pi* Transition Wavelength"
        self.obj_str = "Pi-Pi* Transition Wavelength"

    def _get_columns_to_remove(self) -> List[str]:
        return ["SMILES", "Pi-Pi* Transition Wavelength"]


class MultiRedoxDataProcessor(DataProcessor):
    """
    RangeIndex: 1407 entries, 0 to 1406
    Data columns (total 7 columns):
    #   Column                 Non-Null Count  Dtype
    --  ------                 --------------  -----
    0   Entry Number           1407 non-null   int64
    1   File Name              1407 non-null   object
    2   SMILES                 1407 non-null   object
    3   IUPAC Name             1407 non-null   object
    4   Ered                   1407 non-null   float64
    5   HOMO                   1407 non-null   float64
    6   Gsol                   1407 non-null   float64
    7   Absorption Wavelength  1407 non-null   float64
    dtypes: float64(4), int64(1), object(2)
    memory usage: 77.1+ KB

    Objective: Minimize Ered, minimize Gsol
    """

    def __init__(self, prompt_builder, tokenizer):
        super().__init__(
            prompt_builder=prompt_builder, num_outputs=2, tokenizer=tokenizer
        )
        self.target_col = ["Ered", "Gsol"]
        self.obj_str = "redox potential and solvation energy"

    def _get_columns_to_remove(self) -> List[str]:
        return [
            "Entry Number",
            "File Name",
            "SMILES",
            "HOMO",
            "Ered",
            "Gsol",
            "Absorption Wavelength",
        ]


class MultiLaserDataProcessor(DataProcessor):
    """
    Data columns (total 2 columns):
    #   Column                              Dtype
    --  ------                              -----
    0   SMILES                              object
    1   Fluorescence Oscillator Strength    float64
    2   Electronic Gap                      float64

    dtypes: float64(2), object(1)

    Objective: Maximize the Fluorescence Oscillator Strength (secondary objective: maximize the Electronic Gap)
    """

    def __init__(self, prompt_builder, tokenizer):
        super().__init__(
            prompt_builder=prompt_builder, num_outputs=2, tokenizer=tokenizer
        )
        self.target_col = ["Fluorescence Oscillator Strength", "Electronic Gap"]
        self.obj_str = "fluorescence oscillator strength and electronic gap"

    def _get_columns_to_remove(self) -> List[str]:
        return ["SMILES", "Fluorescence Oscillator Strength", "Electronic Gap"]


if __name__ == "__main__":
    tok = RobertaTokenizer.from_pretrained("roberta-base")
    df = pd.read_csv("data/redox_mer.csv")

    dset = RedoxDataProcessor(tokenizer=tok)
    dataloader = dset.get_dataloader(df)

    for data in dataloader:
        # print(data.keys()); input()
        print(data.input_ids.shape, data.attention_mask.shape, data.targets.shape)
        input()

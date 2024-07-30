class PromptBuilder:
    """
    Base class for building prompts. To be derived per dataset.

    Parameters:
    -----------
    kind: str
        The name of the prompt type
    """

    def __init__(self, kind: str):
        self.kind = kind

    def get_prompt(self, smiles_str: str, obj_str: str) -> str:
        """
        Given a SMLIES string, create a prompt string.

        Parameters:
        -----------
        smiles_str: str
            The SMILES string of a molecule.

        obj_str: str
            The textual name of the objective property. E.g. "redox potential" or "solvation energy"
        """
        if self.kind == "naive":
            return f"Predict the {obj_str} of the following SMILES: {smiles_str}!"
        elif self.kind == "completion":
            return f"The estimated {obj_str} of the molecule {smiles_str} is: "
        elif self.kind == "single-number":
            return f"Answer with just numbers without any further explanation! What is the estimated {obj_str} of the molecule with the SMILES string {smiles_str}?"
        elif self.kind == "just-smiles":
            return smiles_str
        else:
            return NotImplementedError

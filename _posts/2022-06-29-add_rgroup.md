---
title: Adding R-groups to molecules in RDKit
date: 2022-06-29 00:00:00 Z
layout: post
excerpt: A tutorial on how to attach R-groups to molecules. Along the way we will
  learning how to edit molecules in RDKit, how to utilize atom map numbers and wildcard
  atoms.
toc: false
author: David Huang
featured_image: https://qph.cf2.quoracdn.net/main-qimg-7381df4d4e273e439b123b82f887545b
---

In this example, we will be using `Chem.RWMol` to stitch compounds together to recreate Table 2 in "Structure-Aided Design, Synthesis, and Biological Evaluation of Potent and Selective Non-Nucleoside Inhibitors Targeting Protein Arginine Methyltransferase 5".

{% include image.html
   style="width: 60%; height: auto;"
   description=""
   url="https://oloren-blogcontent.s3.us-east-2.amazonaws.com/rgroup/fig1.jpeg"
    %}

# Creating the base molecule
First, we'll draw the base molecules using the wildcard symbol * for the R and X groups, with atom map numbers such that R and X are *:1 and *:2 respectively.
```python
from  rdkit import Chem, DataStructs
from rdkit import Chem

m = Chem.MolFromSmiles("[*:1]C1=C([*:2]=C(C2=CC=C(OCC3=CC=CC=C3)C=C2)N4)C4=NC=N1")
```

{% include image.html
   style="width: 60%; height: auto;"
   description=""
   url="https://oloren-blogcontent.s3.us-east-2.amazonaws.com/rgroup/fig2.png"
    %}


# Function for addition of groups
Let's write a function for using a dictionary, mapping, which maps atom map number to the group to be added, to add groups onto a Chem.Mol, m. Code explanation is in comments as well as below the code block.

```python
from rdkit import Chem, DataStructs
def add_groups(m, mapping):
    # Written by David Huang, Oloren AI

    # Loop over atoms until there are no wildcard atoms
    while True:

        # Find wildcard atom if available, otherwise exit
        a = None
        for a_ in m.GetAtoms():
            if a_.GetAtomicNum() == 0:
                a = a_
                break
        if a is None:
            break

        # Get appropriate group to substitute in from mapping
        group = mapping[a.GetAtomMapNum()]

        if group.GetNumAtoms() == 1:

            # Directly substitute atom in, if single atom group
            a.SetAtomicNum(group.GetAtomWithIdx(0).GetAtomicNum())
            a.SetAtomMapNum(0)
        else:

            # Set wildcard atoms to having AtomMapNum 1000 for tracking
            a.SetAtomMapNum(1000)

            for a_ in group.GetAtoms():
                if a_.GetAtomicNum() == 0:
                    a_.SetAtomMapNum(1000)

            # Put group and base molecule together and make it editable
            m = Chem.CombineMols(m, group)
            m = Chem.RWMol(m)

            # Find using tracking number the atoms to merge in new molecule
            a1 = None
            a2 = None
            bonds = []
            for a in m.GetAtoms():
                if a.GetAtomMapNum() == 1000:
                    if a1 is None:
                        a1 = a
                    else:
                        a2 = a

            # Find atoms to bind together based on atoms to merge
            b1 = a1.GetBonds()[0]
            start = (b1.GetBeginAtomIdx() if b1.GetEndAtomIdx() == a1.GetIdx()
                else b1.GetEndAtomIdx())

            b2 = a2.GetBonds()[0]
            end = (b2.GetBeginAtomIdx() if b2.GetEndAtomIdx() == a2.GetIdx()
                else b2.GetEndAtomIdx())

            # Add the connection and remove original wildcard atoms
            m.AddBond(start, end, order=Chem.rdchem.BondType.SINGLE)
            m.RemoveAtom(a1.GetIdx())
            m.RemoveAtom(a2.GetIdx())

    return m
```

Our key strategy in this function is to define attachment points with the Atom Map Number in both the molecule and in the mapping dictionary. To make the additions, we loop over all the atoms in the molecule and check if they are a wildcard atom (atomic number 0) and if they are we make the appropriate additions.

Note, we do this in a while loop where we loop until we can't find a wildcard atom, because we cannot edit a molecule within a for a in m.GetAtoms() loop without throwing a RuntimeError: Sequence modified during iteration. Exception.

To make these additions, we first give the wildcard atoms in the molecule and the group to be added an Atom Map Number of 1000` to keep track of it. Then we use Chem.CombineMols to combine the molecule and group together in one object and then we use Chem.RWMol to make it edittable. We then re-find those tracked atoms and then find the atoms which they are bonded to. We join those atoms together and remove the tracked wildcard atoms.


# Recreating the entire table

Now, to recreate the table we will write a CSV file with the R and X groups as SMILES and load it in as a pd.DataFrame.

```python
import pandas as pd
df = pd.read_csv("table2_raw.csv")
```

{% include image.html
   style="width: 60%; height: auto;"
   description=""
   url="https://oloren-blogcontent.s3.us-east-2.amazonaws.com/rgroup/fig3.jpeg"
    %}

We can then input the rows of this table into our previous function. We can also convert these to SMILES.

```python
df["mol"] = [add_groups(Chem.MolFromSmiles("[*:1]C1=C([*:2]=C(C2=CC=C(OCC3=CC=CC=C3)C=C2)N4)C4=NC=N1"),
    {1: Chem.MolFromSmiles(r["R"]),
    2: Chem.MolFromSmiles(r["X"])}) for i, r in df.iterrows()]
df["smi"] = [Chem.MolToSmiles(m) for m in df["mol"]]
```

# Our resulting table!
Let's visualize our result with `Draw.MolsToGrid`

```python
from rdkit.Chem import Draw
Draw.MolsToGridImage(df["mol"], molsPerRow=4, useSVG=True)
```

{% include image.html
   style="width: 60%; height: auto;"
   description=""
   url="https://oloren-blogcontent.s3.us-east-2.amazonaws.com/rgroup/fig4.jpeg"
    %}


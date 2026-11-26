import pandas as pd
import pytest
from neuroseg.data import validate, EXPECTED

def test_patient_leakage_rejected(tmp_path):
    rows=[]
    index=0
    patient_index=0
    for split,(n,p) in EXPECTED.items():
        for i in range(n):
            rows.append({"filename":f"{index}.png","patient_group":f"p{patient_index+i%p}","split":split})
            index+=1
        patient_index+=p
    # Keep total number of patients unchanged but move one slice across groups.
    rows[1]["patient_group"]=rows[EXPECTED["train"][0]]["patient_group"]
    pd.DataFrame(rows).to_csv(tmp_path/"dataset_manifest.csv",index=False)
    with pytest.raises(AssertionError): validate(tmp_path)

import pandas as pd
#import numpy as np
import re
import os
import argparse

def clean_html_and_noise(text) :
    if not isinstance (text, str) :
        return None
    
    # pour enlever les html tags des critiques
    text= re.sub (r"<.*?>", " " , text)

    # replacer multiple whitespace avec une seul
    text= re.sub (r"\s+", " " , text).strip()

    return text

def preprocess(input_path , output_path): # main fct
    print(f"[+] Loading {input_path}")
    df= pd.read_csv(input_path, dtype=str, keep_default_na=False, na_values=['']) 

    if "review_content" not in df.columns:

        raise ValueError("Column 'review_content' is not found in the CSV entered") #cas ou nous utilisons any other csv with no review_content column

    # selectionner review_content column ou ya les critiques et la netoyyer
    names_series= df["review_content"].replace('', pd.NA).dropna().astype(str)
    names_series= names_series.map(clean_html_and_noise).dropna().reset_index(drop=True)

    out_df = pd.DataFrame({"critique" : names_series})
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False)

    print(f"[+] Saved the cleaned CSV to {output_path} (number of rows is: {len(out_df)})")

if __name__ == "__main__":
    files= argparse.ArgumentParser()
    files.add_argument("--input" , required=True ) # entrer le csv input, example fightclub_critiques.csv
    files.add_argument("--output" , default=None ) # sortie du csv clean, example fightclub_clean.csv
    
    args= files.parse_args()
    input_path= args.input
    if args.output :
        output_path= args.output

    else:
        base= os.path.splitext(os.path.basename(input_path))[0]
        output_path= f"data/{base}_clean.csv"
    #main appel
    preprocess(input_path, output_path)

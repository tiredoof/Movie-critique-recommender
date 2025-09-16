import os
import argparse
import pandas as pd
import joblib
#from scipy import sparse
#from nltk.corpus import stopwords
import numpy as np
from sentence_transformers import SentenceTransformer

def build (clean_csv, model_dire="models" , model_name= "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" ) : # main fct
  
    print( f"[+] Loading embedding model: {model_name}") 
    model=SentenceTransformer (model_name) # elle telecharge et charge le modele d’embedding SentenceTransformers SBERT from Hugging Face lib

    df= pd.read_csv(clean_csv, dtype=str).fillna("")
    critiques= df["critique"].astype(str).tolist() #cleared CSV files only have 1 column named "critique"

    print( "[+] Encoding the files' critiques into embeddings ... " )
    embeddings= model.encode( critiques, batch_size =32 , show_progress_bar = True , convert_to_numpy= True )
   
    # Partie creation du dossier models + chemin pour embeddings, metadonnees etc
    os.makedirs(model_dire , exist_ok=True)
    base= os.path.splitext(os.path.basename(clean_csv))[0].replace("_clean", "" )
    vec_chem= os.path.join(model_dire , f"{base}_encoder.joblib" ) 
    X_chem= os.path.join(model_dire ,  f"{base}_X.npy") #results embeddings chemin
    meta_chem= os.path.join(model_dire , f"{base}_meta.joblib" )

    print(f" [+] Saving encoder reference in {vec_chem} ")
    joblib.dump(model_name , vec_chem) 
    print(f" [+] Saving embeddings in { X_chem } ")
    np.save( X_chem, embeddings)
    print(f" [+] Saving meta in { meta_chem} ")
    joblib.dump({"n_docs": embeddings.shape[0], "csv": clean_csv} , meta_chem)
    print("[+] Done. ")

    return vec_chem ,X_chem, meta_chem


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument( "--clean_csv" , required=True)
    parser.add_argument("--model_dir" , default = "models") # ou to store models
    parser.add_argument("--model_name" , default = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    args = parser.parse_args()
    build(args.clean_csv, args.model_dir, args.model_name)

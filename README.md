> [!NOTE]  
> Installation mit conda wird empfohlen. `conda env create -f environment.yml`

<div>
  <img width="45%" height="45%" alt="panda" src="https://github.com/user-attachments/assets/9041cf71-fc52-4b9a-ab5d-3f65b19ef181" />
  <img width="45%" height="45%" alt="panda-x-tiger" src="https://github.com/user-attachments/assets/67ab5313-2f67-4392-8aa5-8e4fda3270c6" />
</div>


## Image Recommender – Big Data Engineering
### Überblick
Im Rahmen des Moduls *Big Data Engineering* haben wir ein Content-Based *Image Recommender System* entwickelt. Ziel ist es, für ein Eingabebild die ähnlichsten Bilder aus einem Datensatz effizient zu finden.


## Installation/Usage
Download des [Index](https://fhd-my.sharepoint.com/:u:/r/personal/richard_bihlmeier_study_hs-duesseldorf_de/Documents/4.%20Semester/ImageRecommender/ImageIDX.faiss?csf=1&web=1&e=JptL0n), 
dem [KMeans-Modell](https://fhd-my.sharepoint.com/:u:/r/personal/richard_bihlmeier_study_hs-duesseldorf_de/Documents/4.%20Semester/ImageRecommender/sift_kmeans.faiss?csf=1&web=1&e=KfwcAU) 
sowie der [Datenbank](https://fhd-my.sharepoint.com/:u:/r/personal/richard_bihlmeier_study_hs-duesseldorf_de/Documents/4.%20Semester/ImageRecommender/ImageDB.db?csf=1&web=1&e=DWCz0K) \
Anpassung der Pfade in `GUI.py` \
Drag'n'Drop Image onto GUI and enjoy! 😜

## Features
- Kombination aus:
  - **Farbhistogrammen**
  - **DINO Embeddings**
  - **BOVW mit SIFT/ORB-Descriptors**
- Zusammenführung zu einem gewichteten Feature-Vektor
- **FAISS** für schnelle Ähnlichkeitssuche mit:
  - **Cosine Similarity** (durch L2-Normalisierung + Inner Product Index)  
- **SQLite** zur Verwaltung der Pfade & Metadaten  

## Pipeline
1. Feature-Extraktion & Speicherung in SQLite  
2. Aufbau eines FAISS-Index (normalisierte Vektoren)  
3. Query: Extraktion → Normalisierung → Suche im Index  
4. Rückgabe der Top-k ähnlichen Bilder
5. Anzeige in GUI

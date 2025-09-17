import os
from typing import Optional, Union
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# === Création de l'app ===
app = FastAPI(
    title="🎬 Movie Critique Recommender",
    description="UI + API pour recommander des critiques similaires avec SBERT ✨",
    version="3.6",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Films disponibles
AVAILABLE = {
    "interstellar": {"clean_csv": "data/interstellar_clean.csv"},
    "fightclub": {"clean_csv": "data/fightclub_clean.csv"},
}

MODEL_DIR = "models"

# === Chargement du modèle et données ===
def ensure_model_loaded(film):
    if film not in AVAILABLE:
        raise HTTPException(status_code=404, detail=f"Film '{film}' non disponible.")
    
    if "encoder" not in AVAILABLE[film]:
        base = film
        vec_path = os.path.join(MODEL_DIR, f"{base}_encoder.joblib")
        X_path = os.path.join(MODEL_DIR, f"{base}_X.npy")
        meta_path = os.path.join(MODEL_DIR, f"{base}_meta.joblib")

        if not os.path.exists(vec_path) or not os.path.exists(X_path):
            raise HTTPException(status_code=500, detail=f"Modèle pour '{film}' absent. Lancez build_index.py.")
        
        model_name = joblib.load(vec_path)
        AVAILABLE[film]["encoder"] = SentenceTransformer(model_name)
        AVAILABLE[film]["X"] = np.load(X_path)
        AVAILABLE[film]["meta"] = joblib.load(meta_path)
        df = pd.read_csv(AVAILABLE[film]["clean_csv"], dtype=str).fillna("")
        AVAILABLE[film]["df"] = df.reset_index(drop=True)
    
    return AVAILABLE[film]

def top_sorted(sim_array):
    idx = np.argsort(-sim_array)
    return idx, sim_array[idx]

# === UI principale ===
@app.get("/", response_class=HTMLResponse)
def home(
    film: str = Query("interstellar"),
    text: str = Query(""),
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=50),
    limit: Optional[Union[int, str]] = Query(None)
):
    results_html = ""
    pagination_html = ""
    filter_form_html = ""
    total_results = 0

    # Normalisation de limit
    if isinstance(limit, str):
        if limit.strip() == "":
            limit = None
        else:
            try:
                limit = int(limit)
            except ValueError:
                limit = None

    if text.strip() != "":
        # Charger modèle + données
        m = ensure_model_loaded(film)
        vect = m["encoder"].encode([text], convert_to_numpy=True)
        X = m["X"]
        sims = cosine_similarity(vect, X).flatten()

        # Trier tous les résultats
        idxs, scores = top_sorted(sims)
        df = m["df"]
        total_results = len(idxs)

        # Appliquer filtre (limit)
        if limit is not None:
            idxs = idxs[:limit]
            scores = scores[:limit]
            total_results = len(idxs)

        # Pagination
        start = (page - 1) * per_page
        end = start + per_page
        idxs_page = idxs[start:end]
        scores_page = scores[start:end]

        for i, s in zip(idxs_page, scores_page):
            critique_text = df.iloc[int(i)]["critique"]
            sim_percent = f"{s * 100:.1f}%"
            short_text = critique_text[:200] + ("..." if len(critique_text) > 200 else "")
            
            # Déterminer la couleur en fonction du score de similarité
            if s >= 0.8:
                score_color = "text-green-600"
            elif s >= 0.6:
                score_color = "text-yellow-600"
            elif s >= 0.4:
                score_color = "text-orange-600"
            else:
                score_color = "text-red-600"
                
            results_html += f"""
            <div class="bg-white rounded-xl shadow-md p-6 mb-4 hover:shadow-lg transition-shadow duration-300 result-item">
                <div class="flex justify-between items-start mb-3">
                    <span class="text-sm text-gray-500">ID: {i}</span>
                    <span class="text-lg font-semibold {score_color}">Similarité de {sim_percent}</span>
                </div>
                <div class="text-gray-800 critique-cell break-words overflow-wrap-anywhere">
                    <p class="short-text">{short_text}</p>
                    <p class="full-text hidden mt-2">{critique_text}</p>
                    {"<button class='toggle-btn mt-2 text-blue-500 hover:text-blue-700 font-medium text-sm transition-colors duration-200'>Voir plus →</button>" if len(critique_text) > 200 else ""}
                </div>
            </div>
            """

        total_pages = (total_results + per_page - 1) // per_page
        if (limit is None or limit > per_page) and total_pages > 1:
            pagination_html = '<div class="flex space-x-2 mt-8 justify-center">'
            window = 2
            start_page = max(1, page - window)
            end_page = min(total_pages, page + window)

            # Bouton précédent
            if page > 1:
                pagination_html += f"""
                <form method="get" class="inline">
                    <input type="hidden" name="film" value="{film}">
                    <input type="hidden" name="text" value='{text.replace("'", "&#39;")}'>
                    <input type="hidden" name="page" value="{page-1}">
                    <input type="hidden" name="per_page" value="{per_page}">
                    {"<input type='hidden' name='limit' value='" + str(limit) + "'>" if limit else ""}
                    <button type="submit" class="px-4 py-2 rounded-lg bg-gray-200 hover:bg-gray-300 transition-colors duration-200 text-gray-800">
                        <i class="fas fa-chevron-left"></i>
                    </button>
                </form>
                """
            
            if start_page > 1:
                pagination_html += f"""
                <form method="get" class="inline">
                    <input type="hidden" name="film" value="{film}">
                    <input type="hidden" name="text" value='{text.replace("'", "&#39;")}'>
                    <input type="hidden" name="page" value="1">
                    <input type="hidden" name="per_page" value="{per_page}">
                    {"<input type='hidden' name='limit' value='" + str(limit) + "'>" if limit else ""}
                    <button type="submit" class="px-4 py-2 rounded-lg bg-gray-200 hover:bg-gray-300 transition-colors duration-200 text-gray-800">1</button>
                </form>
                """
                if start_page > 2:
                    pagination_html += "<span class='px-2 py-2 text-gray-800'>...</span>"
            
            for p in range(start_page, end_page + 1):
                active = "bg-blue-500 text-white hover:bg-blue-600" if p == page else "bg-gray-200 hover:bg-gray-300 text-gray-800"
                pagination_html += f"""
                <form method="get" class="inline">
                    <input type="hidden" name="film" value="{film}">
                    <input type="hidden" name="text" value='{text.replace("'", "&#39;")}'>
                    <input type="hidden" name="page" value="{p}">
                    <input type="hidden" name="per_page" value="{per_page}">
                    {"<input type='hidden' name='limit' value='" + str(limit) + "'>" if limit else ""}
                    <button type="submit" class="px-4 py-2 rounded-lg {active} transition-colors duration-200">{p}</button>
                </form>
                """
            
            if end_page < total_pages:
                if end_page < total_pages - 1:
                    pagination_html += "<span class='px-2 py-2 text-gray-800'>...</span>"
                pagination_html += f"""
                <form method="get" class="inline">
                    <input type="hidden" name="film" value="{film}">
                    <input type="hidden" name="text" value='{text.replace("'", "&#39;")}'>
                    <input type="hidden" name="page" value="{total_pages}">
                    <input type="hidden" name="per_page" value="{per_page}">
                    {"<input type='hidden' name='limit' value='" + str(limit) + "'>" if limit else ""}
                    <button type="submit" class="px-4 py-2 rounded-lg bg-gray-200 hover:bg-gray-300 transition-colors duration-200 text-gray-800">{total_pages}</button>
                </form>
                """
            
            # Bouton suivant
            if page < total_pages:
                pagination_html += f"""
                <form method="get" class="inline">
                    <input type="hidden" name="film" value="{film}">
                    <input type="hidden" name="text" value='{text.replace("'", "&#39;")}'>
                    <input type="hidden" name="page" value="{page+1}">
                    <input type="hidden" name="per_page" value="{per_page}">
                    {"<input type='hidden' name='limit' value='" + str(limit) + "'>" if limit else ""}
                    <button type="submit" class="px-4 py-2 rounded-lg bg-gray-200 hover:bg-gray-300 transition-colors duration-200 text-gray-800">
                        <i class="fas fa-chevron-right"></i>
                    </button>
                </form>
                """
            
            pagination_html += "</div>"

        # Formulaire de filtre
        filter_form_html = f"""
        <div class="bg-blue-50 p-4 rounded-xl mb-6">
            <form method="get" class="flex flex-col md:flex-row gap-4 items-center">
                <input type="hidden" name="film" value="{film}">
                <input type="hidden" name="text" value='{text.replace("'", "&#39;")}'>
                <input type="hidden" name="page" value="1">
                <input type="hidden" name="per_page" value="{per_page}">
                <label class="text-gray-700 font-medium">Limiter le nombre de résultats :</label>
                <input type="number" name="limit" value="{limit if limit else ''}" min="1" class="border rounded-lg p-2 w-32 focus:ring-2 focus:ring-blue-300 focus:outline-none text-gray-800" placeholder="ex: 6">
                <button type="submit" class="bg-green-500 text-white rounded-lg px-4 py-2 hover:bg-green-600 transition-colors duration-200"> Appliquer </button>
                {"<a href='/?film=" + film + "&text=" + text.replace("'", "%27") + "&page=1&per_page=" + str(per_page) + "' class='bg-red-500 text-white rounded-lg px-4 py-2 hover:bg-red-600 transition-colors duration-200'>Supprimer le filtre</a>" if limit else ""}
            </form>
        </div>
        """
    else:
        results_html = """
        <div class="text-center py-12">
            <div class="inline-block p-4 bg-blue-50 rounded-full mb-4">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-12 w-12 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                </svg>
            </div>
            <h3 class="text-xl font-semibold text-gray-700">Entrez une critique pour voir les résultats</h3>
            <p class="text-gray-500 mt-2">Découvrez des critiques similaires à la vôtre</p>
        </div>
        """

    # Définir les couleurs d'arrière-plan selon le film sélectionné
    header_bg = "bg-gradient-to-r from-purple-500 via-blue-500 to-teal-500"  # Défaut
    button_bg = "bg-gradient-to-r from-purple-500 to-teal-500"
    if film == "interstellar":
        header_bg = "bg-gradient-to-r from-blue-700 via-purple-700 to-gray-900"
        button_bg = "bg-gradient-to-r from-blue-700 to-gray-900"
    elif film == "fightclub":
        header_bg = "bg-gradient-to-r from-yellow-500 via-red-600 to-gray-800"
        button_bg = "bg-gradient-to-r from-yellow-500 to-gray-800"

    # === HTML complet ===
    html_content = f"""
    <html>
    <head>
        <title>🎬 Recommandeur de critiques de films</title>
        <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
            
            body {{
                font-family: 'Poppins', sans-serif;
                background-color: #f8fafc;
            }}
            
            .film-card {{
                transition: all 0.3s ease;
            }}
            
            .film-card:hover {{
                transform: translateY(-5px);
            }}
            
            .critique-cell {{
                transition: all 0.3s ease;
            }}
            
            /* Animation pour le bouton de pagination */
            .page-btn {{
                transition: all 0.2s ease;
            }}
            
            /* Effet de focus pour les champs de formulaire */
            .focus-effect:focus {{
                box-shadow: 0 0 0 3px rgba(66, 153, 225, 0.5);
            }}
            
            /* Animation d'entrée pour les résultats */
            @keyframes fadeIn {{
                from {{ opacity: 0; transform: translateY(10px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            
            .result-item {{
                animation: fadeIn 0.5s ease forwards;
            }}
            
            /* Empêche les mots très longs de déborder */
            .break-words {{
                word-break: break-word;
                overflow-wrap: anywhere;
            }}
            
            .overflow-wrap-anywhere {{
                overflow-wrap: anywhere;
            }}
            
            /* Style pour les formulaires de pagination */
            .pagination-form {{
                display: inline;
            }}
        </style>
        <script>
        document.addEventListener("DOMContentLoaded", () => {{
            // Fonctionnalité pour développer/réduire les critiques
            document.querySelectorAll(".toggle-btn").forEach(btn => {{
                btn.addEventListener("click", () => {{
                    const container = btn.parentElement;
                    const shortText = container.querySelector(".short-text");
                    const fullText = container.querySelector(".full-text");
                    
                    if (shortText.classList.contains("hidden")) {{
                        shortText.classList.remove("hidden");
                        fullText.classList.add("hidden");
                        btn.innerHTML = "Voir plus →";
                    }} else {{
                        shortText.classList.add("hidden");
                        fullText.classList.remove("hidden");
                        btn.innerHTML = "Voir moins ↑";
                    }}
                }});
            }});
            
            // Animation pour les éléments de résultat
            const observerOptions = {{
                root: null,
                rootMargin: '0px',
                threshold: 0.1
            }};
            
            const observer = new IntersectionObserver((entries) => {{
                entries.forEach(entry => {{
                    if (entry.isIntersecting) {{
                        entry.target.style.opacity = "1";
                        entry.target.style.transform = "translateY(0)";
                    }}
                }});
            }}, observerOptions);
            
            document.querySelectorAll('.result-item').forEach(item => {{
                item.style.opacity = "0";
                item.style.transform = "translateY(20px)";
                item.style.transition = "opacity 0.5s ease, transform 0.5s ease";
                observer.observe(item);
            }});
        }});
        </script>
    </head>
    <body class="min-h-screen {header_bg}">
        <div class="max-w-6xl mx-auto px-4 py-8">
            <header class="text-center mb-12">
                <h1 class="text-5xl md:text-6xl font-bold mb-4 flex items-center justify-center text-white">
                    <i class="fas fa-film mr-4"></i> Recommandeur de Critiques
                </h1>
                <p class="text-2xl opacity-90 text-white">Découvrez des critiques similaires à la vôtre grâce à l'IA</p>
            </header>
            
            <main class="bg-white rounded-2xl shadow-2xl p-6 md:p-8">
                <form method="get" class="space-y-6 mb-8">
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div>
                            <label class="block text-lg font-medium mb-2 text-gray-700">
                                <i class="fas fa-star mr-2"></i> Sélectionnez un film
                            </label>
                            <select name="film" class="w-full border border-gray-300 rounded-xl p-3 focus:ring-2 focus:ring-blue-300 focus:outline-none text-gray-800">
                                <option value="interstellar" {"selected" if film=="interstellar" else ""}>Interstellar</option>
                                <option value="fightclub" {"selected" if film=="fightclub" else ""}>Fight Club</option>
                            </select>
                        </div>
                        
                        <div>
                            <label class="block text-lg font-medium mb-2 text-gray-700">
                                <i class="fas fa-list-ol mr-2"></i> Résultats par page
                            </label>
                            <input type="number" name="per_page" value="{per_page}" min="1" max="50" class="w-full border border-gray-300 rounded-xl p-3 focus:ring-2 focus:ring-blue-300 focus:outline-none text-gray-800">
                        </div>
                    </div>
                    
                    <div>
                        <label class="block text-lg font-medium mb-2 text-gray-700">
                            <i class="fas fa-pencil-alt mr-2"></i> Votre critique
                        </label>
                        <textarea name="text" placeholder="Écrivez votre critique ici..." class="w-full border border-gray-300 rounded-xl p-4 focus:ring-2 focus:ring-blue-300 focus:outline-none text-gray-800" rows="4">{text}</textarea>
                    </div>
                    
                    <button type="submit" class="w-full {button_bg} text-white rounded-xl py-4 px-6 font-semibold text-lg hover:opacity-90 transition-all duration-300 transform hover:-translate-y-1 shadow-lg">
                        <i class="fas fa-search mr-2"></i> Trouver des critiques similaires
                    </button>
                </form>
                
                {"<hr class='border-gray-200 my-6'>" if text.strip() else ""}
                
                {"<h2 class='text-2xl font-bold mb-6 flex items-center text-gray-800'><i class='fas fa-list mr-3'></i> Résultats</h2>" if text.strip() else ""}
                
                {filter_form_html if text.strip() else ""}
                
                <div class="space-y-4">
                    {results_html}
                </div>
                
                {pagination_html}
            </main>
            
            <footer class="text-center mt-12 text-white">
                <p>Movie Recommender by Maya BEN ABDELATIF</p>
            </footer>
        </div>
    </body>
    </html>
    """
    return html_content
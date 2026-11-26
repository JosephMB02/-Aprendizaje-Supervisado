import os
import re
import requests
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Billboard Predictor", page_icon="🎵", layout="centered")
st.title("🎵 ¿Esta canción podría llegar al Top de Billboard?")
st.write("Escribe el nombre de una canción y su artista. La app consulta iTunes para recuperar información y utiliza el modelo Random Forest entrenado en la ADA 3.")

MODEL_PATH = "modelo_random_forest.joblib"
DATA_PATH = "billboard_limpio.csv"
FEATURES = ["durationSeg", "anioNacimiento", "mood", "tempo", "genre", "artist_type"]

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_catalog():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame()

def normalize(value):
    value = str(value or "").casefold().strip()
    return re.sub(r"[^a-z0-9áéíóúüñ]+", "", value)

def search_itunes(song, artist):
    """Busca metadatos y preview usando la API pública de iTunes Search."""
    term = f"{song} {artist}"
    response = requests.get(
        "https://itunes.apple.com/search",
        params={"term": term, "entity": "song", "limit": 10, "country": "MX"},
        timeout=15,
        headers={"User-Agent": "BillboardADA3/1.0"}
    )
    response.raise_for_status()
    results = response.json().get("results", [])
    if not results:
        return None

    artist_key = normalize(artist)
    song_key = normalize(song)
    # Preferir coincidencia de título y artista; si no, usar el primer resultado.
    ranked = sorted(
        results,
        key=lambda item: (
            normalize(item.get("trackName", "")) == song_key,
            artist_key in normalize(item.get("artistName", "")),
        ),
        reverse=True,
    )
    return ranked[0]

def find_historical_row(catalog, song, artist):
    if catalog.empty:
        return None
    required = {"title", "artist"}
    if not required.issubset(catalog.columns):
        return None
    song_key, artist_key = normalize(song), normalize(artist)
    matches = catalog[
        catalog["title"].map(normalize).eq(song_key)
        & catalog["artist"].map(normalize).str.contains(re.escape(artist_key), na=False)
    ]
    return None if matches.empty else matches.iloc[0]

with st.form("song_form"):
    song_name = st.text_input("Nombre de la canción", placeholder="Ej. Blinding Lights")
    artist_name = st.text_input("Nombre del artista", placeholder="Ej. The Weeknd")
    submitted = st.form_submit_button("Consultar y predecir")

if submitted:
    if not song_name.strip() or not artist_name.strip():
        st.warning("Escribe tanto el nombre de la canción como el artista.")
        st.stop()

    if not os.path.exists(MODEL_PATH):
        st.error("No se encontró el modelo. Ejecuta todas las celdas de la notebook y agrega modelo_random_forest.joblib al repositorio.")
        st.stop()

    try:
        with st.spinner("Buscando la canción..."):
            track = search_itunes(song_name.strip(), artist_name.strip())
    except requests.RequestException as error:
        st.error(f"No fue posible consultar iTunes en este momento: {error}")
        st.stop()

    if track is None:
        st.error("No se encontró una canción en iTunes con esos datos. Revisa el título y el artista.")
        st.stop()

    catalog = load_catalog()
    historical = find_historical_row(catalog, song_name, artist_name)

    # iTunes aporta duración, género, título/artista normalizados y preview.
    # Si la canción ya existe en el dataset, se reutilizan sus atributos históricos.
    # Para canciones nuevas, los campos que no ofrece iTunes quedan como NaN
    # y el pipeline los completa con los imputadores aprendidos durante el entrenamiento.
    duration = track.get("trackTimeMillis")
    duration_sec = float(duration) / 1000 if duration is not None else np.nan
    genre = track.get("primaryGenreName", np.nan)

    row = {
        "durationSeg": duration_sec,
        "anioNacimiento": np.nan,
        "mood": np.nan,
        "tempo": np.nan,
        "genre": genre,
        "artist_type": np.nan,
    }

    if historical is not None:
        for feature in FEATURES:
            if feature in historical.index and pd.notna(historical[feature]):
                row[feature] = historical[feature]

    input_df = pd.DataFrame([row], columns=FEATURES)
    model = load_model()
    prediction = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]
    classes = list(model.named_steps["clasificador"].classes_)
    top_probability = float(probabilities[classes.index(1)]) if 1 in classes else 0.0

    st.subheader("Información encontrada")
    col1, col2 = st.columns([1, 2])
    with col1:
        artwork = track.get("artworkUrl100")
        if artwork:
            st.image(artwork, use_container_width=True)
    with col2:
        st.write(f"**Canción:** {track.get('trackName', song_name)}")
        st.write(f"**Artista:** {track.get('artistName', artist_name)}")
        st.write(f"**Álbum:** {track.get('collectionName', 'No disponible')}")
        st.write(f"**Género:** {track.get('primaryGenreName', 'No disponible')}")
        st.write(f"**Duración:** {round(duration_sec)} segundos" if pd.notna(duration_sec) else "**Duración:** No disponible")

    preview = track.get("previewUrl")
    if preview:
        st.write("**Preview de la canción:**")
        st.audio(preview)
    else:
        st.info("iTunes no proporcionó un preview para esta canción.")

    st.subheader("Resultado del modelo")
    if prediction == 1:
        st.success("El modelo predice: **Top de Billboard (clase 1)**")
    else:
        st.info("El modelo predice: **No Top de Billboard (clase 0)**")
    st.metric("Probabilidad estimada de clase Top", f"{top_probability:.1%}")
    st.progress(min(max(top_probability, 0.0), 1.0))

    if historical is not None:
        st.caption("Se encontraron atributos de esta canción en el catálogo histórico utilizado en la ADA.")
    else:
        st.warning(
            "Esta canción no aparece en el catálogo histórico. iTunes solo aporta algunos metadatos; "
            "los atributos no disponibles (mood, tempo, tipo y año de nacimiento) se imputan con los "
            "valores aprendidos durante el entrenamiento. Por eso esta predicción es orientativa y "
            "tiene incertidumbre adicional."
        )

    st.caption(
        "Proyecto académico: el modelo utiliza un histórico pequeño (2004–2015). "
        "La probabilidad no es una garantía ni una predicción oficial de Billboard."
    )

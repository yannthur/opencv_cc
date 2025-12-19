import streamlit as st
from processor import ImageProcessor
from PIL import Image
import io

# --- 1. CONFIGURATION & CSS ---
st.set_page_config(page_title="ProVision Studio", layout="wide", page_icon="💠")

def local_css():
    st.markdown("""
    <style>
        /* Thème Sombre Professionnel */
        .stApp { background-color: #0E1117; }
        
        /* Sidebar plus élégante */
        [data-testid="stSidebar"] {
            background-color: #161B22;
            border-right: 1px solid #30363D;
        }

        /* En-têtes */
        h1, h2, h3 { color: #E6EDF3; font-family: 'Segoe UI', sans-serif; }
        
        /* Conteneurs personnalisés */
        .css-1r6slb0 { border: 1px solid #30363D; border-radius: 10px; padding: 20px; }
        
        /* Boutons */
        .stButton>button {
            background-color: #238636;
            color: white;
            border: none;
            border-radius: 6px;
            font-weight: 600;
        }
        .stButton>button:hover { background-color: #2EA043; }
        
        /* Images arrondies */
        img { border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    </style>
    """, unsafe_allow_html=True)

local_css()

# --- 2. GESTION DE L'ÉTAT (PERSISTANCE) ---
# Cette fonction assure que toutes les clés existent dès le début
def init_session_state():
    defaults = {
        # Base
        'rotation': 0, 'grayscale': False, 'clahe': False,
        # Flou
        'blur': False, 'blur_intensity': 5,
        'bilateral': False, 'bilateral_d': 9, 'bilateral_sigma': 75,
        # Art
        'reduce_colors': False, 'k_colors': 8, 'cartoon_mode': False,
        'edges': False, 'edge_t1': 50, 'edge_t2': 150,
        # IA
        'detect_faces': False, 'detect_eyes': False, 'scan_codes': False,
        # Couleurs
        'color_extract': False, 'show_mask': False, 'tint_active': False, 'tint_color': '#3366ff',
        'hsv_lower': [0,0,0], 'hsv_upper': [179,255,255],
        # Binaire
        'binary': False, 'binary_otsu': True, 'binary_thresh': 127,
        # UI
        'workflow_mode': 'Base'
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# --- 3. UI PRINCIPALE ---
def main():
    # --- HEADER ---
    with st.sidebar:
        st.title("💠 ProVision")
        st.caption("Studio de Traitement d'Image v3.0")
        
        # Upload
        uploaded_file = st.file_uploader("Importer une source", type=['jpg', 'png', 'jpeg'])
        
        if not uploaded_file:
            st.warning("En attente d'image...")
            return

        st.markdown("---")
        st.markdown("### 🎛️ Paramètres")
        
        # Menu de navigation (Persistant grâce à key)
        mode = st.radio(
            "Espace de travail",
            ["Base", "Retouche & Art", "Laboratoire IA", "Colorimétrie", "Utilitaire"],
            key="workflow_mode", # Lien avec session_state
            label_visibility="collapsed"
        )
        
        # Bouton Reset
        if st.button("↺ Réinitialiser les filtres"):
            for key in st.session_state.keys():
                if key != 'workflow_mode': # On ne reset pas le mode actuel
                    del st.session_state[key]
            st.rerun()

    # Chargement Image
    original_image = ImageProcessor.load_image(uploaded_file)

    # --- 4. PANNEAU DE CONTRÔLE DYNAMIQUE ---
    # Ici, on utilise key="nom_du_reglage" pour que Streamlit sauvegarde 
    # automatiquement la valeur dans st.session_state
    
    with st.sidebar:
        st.markdown(f"#### Mode : {st.session_state.workflow_mode}")
        
        if st.session_state.workflow_mode == "Base":
            with st.expander("Géométrie", expanded=True):
                st.selectbox("Rotation", [0, 90, 180, 270], key="rotation")
            
            with st.expander("Lumière & Contraste", expanded=True):
                st.checkbox("Boost Contraste (CLAHE)", key="clahe")
                st.checkbox("Niveaux de gris", key="grayscale")

        elif st.session_state.workflow_mode == "Retouche & Art":
            with st.expander("Lissage", expanded=True):
                st.checkbox("Lissage Bilatéral (Peau)", key="bilateral")
                if st.session_state.bilateral:
                    st.slider("Diamètre", 5, 20, 9, key="bilateral_d")
                    st.slider("Force", 10, 150, 75, key="bilateral_sigma")
                
                st.checkbox("Flou Gaussien", key="blur")
                if st.session_state.blur:
                    st.slider("Intensité", 3, 21, 5, step=2, key="blur_intensity")

            with st.expander("Créatif", expanded=True):
                st.checkbox("Effet Cartoon", key="cartoon_mode")
                st.checkbox("Réduction couleurs (Poster)", key="reduce_colors")
                if st.session_state.reduce_colors:
                    st.slider("Niveaux de couleur", 2, 16, 8, key="k_colors")
                
                st.checkbox("Détection Contours", key="edges")
                if st.session_state.edges:
                    st.slider("Seuil Min", 0, 255, 50, key="edge_t1")
                    st.slider("Seuil Max", 0, 255, 150, key="edge_t2")

        elif st.session_state.workflow_mode == "Laboratoire IA":
            st.info("Détection par Haar Cascades")
            st.checkbox("Détecter Visages", key="detect_faces")
            st.checkbox("Détecter Yeux", key="detect_eyes", disabled=not st.session_state.detect_faces)

        elif st.session_state.workflow_mode == "Colorimétrie":
            tab1, tab2 = st.tabs(["Extraction", "Teinte"])
            with tab1:
                st.checkbox("Activer Extraction Couleur", key="color_extract")
                if st.session_state.color_extract:
                    col_preset = st.selectbox("Préréglage", ["Bleu", "Vert", "Rouge", "Manuel"])
                    st.checkbox("Voir masque uniquement", key="show_mask")
                    
                    # Logique pour mettre à jour les sliders HSV selon le preset
                    # Note : Streamlit gère mal la mise à jour forcée des sliders sans rerun, 
                    # mais pour l'extraction simple, on assigne les valeurs au state
                    if col_preset == "Bleu":
                        st.session_state.hsv_lower = [100, 50, 50]
                        st.session_state.hsv_upper = [140, 255, 255]
                    elif col_preset == "Vert":
                        st.session_state.hsv_lower = [40, 40, 40]
                        st.session_state.hsv_upper = [80, 255, 255]
                    elif col_preset == "Rouge":
                        st.session_state.hsv_lower = [0, 70, 50]
                        st.session_state.hsv_upper = [10, 255, 255]
                    
                    if col_preset == "Manuel":
                        # Sliders manuels qui écrivent dans le state
                        st.text("Mode manuel (Editez le code pour les sliders fins)")

            with tab2:
                st.checkbox("Activer Teinte Monochrome", key="tint_active")
                st.color_picker("Couleur de teinte", key="tint_color")

        elif st.session_state.workflow_mode == "Utilitaire":
            st.checkbox("Scanner Codes (QR/Barre)", key="scan_codes")
            st.markdown("---")
            st.checkbox("Mode Binaire (N&B Pur)", key="binary")
            if st.session_state.binary:
                st.checkbox("Seuil Automatique (Otsu)", key="binary_otsu")
                if not st.session_state.binary_otsu:
                    st.slider("Seuil manuel", 0, 255, 127, key="binary_thresh")

    # --- 5. TRAITEMENT DU PIPELINE ---
    # On passe st.session_state (qui agit comme un dictionnaire) au processeur
    with st.spinner('Calcul en cours...'):
        final_image, debug_data = ImageProcessor.process_pipeline(original_image, st.session_state)

    # --- 6. AFFICHAGE (LAYOUT SPLIT VIEW PAR DÉFAUT) ---
    col_orig, col_res = st.columns(2)
    
    with col_orig:
        st.markdown("<h3 style='text-align: center; color: #8b949e;'>ORIGINAL</h3>", unsafe_allow_html=True)
        st.image(original_image, use_container_width=True)

    with col_res:
        st.markdown("<h3 style='text-align: center; color: #58a6ff;'>RÉSULTAT</h3>", unsafe_allow_html=True)
        # Gestion auto des channels pour affichage correct
        channels = "GRAY" if len(final_image.shape) == 2 else "RGB"
        st.image(final_image, use_container_width=True, channels=channels)

    # --- 7. FOOTER & INFOS ---
    if debug_data:
        st.info(f"Données détectées : {len(debug_data)}")
        for d in debug_data:
            st.code(d)

    st.markdown("---")
    # Export centré
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        res_pil = Image.fromarray(final_image)
        buf = io.BytesIO()
        res_pil.save(buf, format="PNG")
        st.download_button(
            label="📥 TÉLÉCHARGER LE RÉSULTAT HD",
            data=buf.getvalue(),
            file_name="provision_export.png",
            mime="image/png",
            use_container_width=True
        )

if __name__ == "__main__":
    main()
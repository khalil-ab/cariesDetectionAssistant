"""
Demo Streamlit : detection des caries (U-Net) + assistant de traitement (RAG).

Lancement : streamlit run app/app.py
"""

import os
import sys

import cv2
import numpy as np
import streamlit as st

# Permet d'importer le package src/ quand on lance via streamlit
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import predict

st.set_page_config(page_title="Assistant caries", layout="wide")
st.title("Detection des caries + assistant de traitement")
st.caption("Aide a la lecture — ne remplace pas un diagnostic dentaire.")


@st.cache_resource
def _charger_modele():
    if predict.modele_disponible():
        return predict.charger_modele()
    return None


@st.cache_resource
def _charger_assistant():
    from src.rag.assistant import AssistantCaries
    return AssistantCaries()


col_g, col_d = st.columns(2)

with col_g:
    st.subheader("1. Radio panoramique")
    fichier = st.file_uploader("Charger une radio (png/jpg)", type=["png", "jpg", "jpeg"])

    detection = None
    if fichier is not None:
        donnees = np.frombuffer(fichier.read(), np.uint8)
        image = cv2.imdecode(donnees, cv2.IMREAD_GRAYSCALE)
        modele = _charger_modele()

        if modele is None:
            st.warning("Modele non entraine (models/unet_caries.pth absent). "
                       "Lance d'abord l'entrainement en ligne, puis depose le .pth.")
            st.image(image, caption="Radio", clamp=True, use_container_width=True)
        else:
            masque, stats = predict.segmenter(image, modele)
            apercu = predict.superposer(image, masque)
            st.image(apercu, caption="Caries surlignees", use_container_width=True)
            st.metric("Zones cariees detectees", stats["nb_zones"])
            st.metric("Surface cariee", f"{stats['surface_pct']} %")
            detection = stats
            st.session_state["detection"] = detection

with col_d:
    st.subheader("2. Assistant de traitement")
    detection = st.session_state.get("detection")
    question = st.text_input(
        "Pose une question",
        value="Quels traitements sont possibles pour ces caries ?",
    )
    if st.button("Demander a l'assistant"):
        with st.spinner("Recherche et redaction..."):
            assistant = _charger_assistant()
            res = assistant.repondre(question, detection=detection)
        st.markdown(res["reponse"])
        st.caption("Sources : " + ", ".join(res["sources"]))

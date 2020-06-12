from os import remove

import streamlit as st
from AlgorithmType import *
from ProcessImage import ProcessImage
from skimage.io import imread

__switch = {
    "podstawowe - Meijering": AlgorithmType.SIMPLE2, "podstawowe - progowanie Sauvola": AlgorithmType.SIMPLE,
    "KNN": AlgorithmType.KNN,
    "drzewo decyzyjne": AlgorithmType.TREE, "drzewo decyzyjne 2": AlgorithmType.TREE2
}

if __name__ == "__main__":
    st.title("Wykrywanie naczyń dna siatkówki oka")
    algo = st.sidebar.radio("Wybierz rodzaj przetwarzania:",
                            (
                                "podstawowe - Meijering", "podstawowe - progowanie Sauvola", "drzewo decyzyjne",
                                "drzewo decyzyjne 2"))
    # st.sidebar.write("Opcje")
    extension = "jpg"  # st.sidebar.radio("Wybierz rozszerzenie pliku:", ("jpg", "png", "ppm"))

    # st.write("""
    # #
    # """)

    image = st.file_uploader("Podaj obraz zawierający siatkówkę oka")

    if image is not None:
        with open("temp." + extension, "bw") as f:
            f.write(image.read())
        image = imread("temp." + extension)
        remove("temp." + extension)

        st.write("Wybrany obraz")
        st.image(image, use_column_width=True)

        mask = ProcessImage.get_mask(image)

        progress0 = st.progress(0)
        slot0 = st.empty()

        gray = ProcessImage.preprocesing(image, (slot0, []), mask, progress0)
        st.write("Wstępne przetworzenie:")
        st.image(gray, use_column_width=True)

        al_type = __switch.get(algo)

        progress = st.progress(0)
        slot1 = st.empty()

        process = ProcessImage(al_type)
        result = process.process(gray, mask, origin=image, stream=(slot1, []), progress=progress)
        progress.progress(100)

        st.write("Wynik")
        st.image(result, use_column_width=True)

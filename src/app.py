import io
from pathlib import Path

import streamlit as st
from PIL import Image

from styles import STYLE_PRESETS
from makeover import RoomMakeoverModel
from utils import load_image

# --- Streamlit page config ---
st.set_page_config(
    page_title="AR Home Makeover",
    page_icon="🏠",
    layout="wide",
)

# --- Sidebar: Model settings / info ---
st.sidebar.title("⚙️ Settings")

if "model" not in st.session_state:
    with st.spinner("Loading AI model (first time only)..."):
        st.session_state.model = RoomMakeoverModel()

model = st.session_state.model

st.title("🏠 AR Home Makeover")
st.write(
    "Upload a photo of your room and let AI restyle it in different interior design styles."
)

# --- File uploader ---
uploaded_file = st.file_uploader(
    "Upload a room photo (JPG/PNG)",
    type=["jpg", "jpeg", "png"],
)

# --- Style selection ---
style_names = list(STYLE_PRESETS.keys())
style_labels = [
    f"{STYLE_PRESETS[name]['emoji']} {name}" for name in style_names
]
style_choice = st.selectbox("Choose a style", style_labels, index=0)
style_name = style_names[style_labels.index(style_choice)]

st.sidebar.subheader("Generation controls")

strength = st.sidebar.slider(
    "Strength (how much to change the image)",
    min_value=0.2,
    max_value=0.9,
    value=0.6,
    step=0.05,
    help="Higher = more change, lower = more similar to original.",
)

guidance_scale = st.sidebar.slider(
    "Guidance scale (how strongly to follow the prompt)",
    min_value=3.0,
    max_value=15.0,
    value=7.5,
    step=0.5,
)

steps = st.sidebar.slider(
    "Inference steps",
    min_value=10,
    max_value=50,
    value=30,
    step=5,
)

seed_input = st.sidebar.number_input(
    "Seed (optional, -1 for random)",
    min_value=-1,
    max_value=2**31 - 1,
    value=-1,
    step=1,
)

# --- Main content ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Original room")
    if uploaded_file is not None:
        try:
            input_image = load_image(uploaded_file)
            st.image(input_image, use_column_width=True)
        except Exception as e:
            st.error(f"Could not load image: {e}")
            input_image = None
    else:
        st.info("Upload an image to begin.")
        input_image = None

with col2:
    st.subheader(f"Restyled: {style_name}")
    generate_button = st.button("✨ Restyle my room")

    if generate_button:
        if input_image is None:
            st.warning("Please upload an image first.")
        else:
            seed = None if seed_input == -1 else int(seed_input)

            with st.spinner("AI is restyling your room..."):
                try:
                    output_image = model.restyle_room(
                        image=input_image,
                        style_name=style_name,
                        strength=float(strength),
                        guidance_scale=float(guidance_scale),
                        num_inference_steps=int(steps),
                        seed=seed,
                    )
                    st.image(output_image, use_column_width=True)

                    # Prepare downloadable image
                    buf = io.BytesIO()
                    output_image.save(buf, format="PNG")
                    byte_im = buf.getvalue()

                    st.download_button(
                        label="⬇️ Download restyled image",
                        data=byte_im,
                        file_name=f"room_{style_name.replace(' ', '_').lower()}.png",
                        mime="image/png",
                    )
                except Exception as e:
                    st.error(f"Something went wrong during generation: {e}")

# --- Footer ---
st.markdown("---")
st.caption(
    "Built with ❤️ using Stable Diffusion, Diffusers, and Streamlit. "
    "For portfolio/demo purposes only."
)

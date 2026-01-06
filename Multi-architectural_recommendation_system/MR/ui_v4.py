import os
import base64
import time
import streamlit as st

from part2_localsearch.local_search import predict_architecture_style
from part2_localsearch.local_top import local_top_api
from part3_onlinesearch.online_multi_mod import (
    predict_image_style,
    search_similar_images,
    compare_with_similar_images,
)

# ========== Config ==========
dataset_root = "../dataset/data_clean"
model_path   = "../part1_model/best_grid_50vit_deep.pth"
data_dir     = dataset_root

# ========== Session Init ==========
state = st.session_state
for k, v in {
    "fav_image_path": None,
    "favorites":      [],
    "results":        None,     # style and local_search
    "cached_online":  None,     # cache of online part
    "run_online":     False,    # weather refresh online search
    "last_upload":    None,     # last time uploaded bytes
    "location_style": "",
    "period":         "",
    # "top_k":          6,
}.items():
    if k not in state:
        state[k] = v

# ========== Utility ==========
def image_download_link(img_path, label="Download"):
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return (
        f'<a href="data:file/jpg;base64,{b64}" '
        f'download="{img_path}" style="text-decoration:none;">{label}</a>'
    )

# ---------- cache 1: classify_and_retrieve ----------
@st.cache_data(show_spinner=False)
def classify_and_retrieve(image_bytes: bytes):
    t0 = time.perf_counter()
    tmp_path = "../images/temp_uploaded.jpg"
    with open(tmp_path, "wb") as f:
        f.write(image_bytes)

    style, _, conf = predict_architecture_style(tmp_path)
    local_list     = local_top_api(tmp_path)
    rt             = time.perf_counter() - t0

    return dict(tmp_path=tmp_path, style=style, conf=conf,
                local_list=local_list, runtime=rt)

# ---------- cache 2: online part ----------
@st.cache_data(show_spinner=False)
def retrieve_online(fav_path: str, location: str, period: str, top_k: int):
    t0 = time.perf_counter()
    style, feat = predict_image_style(fav_path, model_path, data_dir)
    cands       = search_similar_images(style, location=location,
                                        period=period, num=10)
    online_res  = compare_with_similar_images(feat, cands, top_k=top_k)
    rt          = time.perf_counter() - t0
    return online_res, rt

# ---------- callback of Favourite button ----------
def set_favorite(path: str):
    state.fav_image_path = path
    state.favorites.append(path)
    state.run_online = True        # trigger online search

# ========== Page Config ==========
st.set_page_config(page_title="Architectural Style Retrieval", layout="wide")
st.title("🏛 Architectural Style Retrieval System")
st.markdown("Upload an image to classify its architectural style and retrieve similar buildings.")

# ---- Sidebar ----
with st.sidebar:
    st.header("Optional Textual Input for Online Search")

    st.text_input(
        "Location / Style (optional)",
        key="location_style",
        placeholder="default: World, any Architectural Style",
        on_change=lambda: state.update(run_online=True),
    )
    st.text_input(
        "Time Period (optional)",
        key="period",
        placeholder="default: daytime",
        on_change=lambda: state.update(run_online=True),
    )
    top_k = st.number_input(
        "Top K Results",
        min_value=1, max_value=10,
        value=6,  # default value
        step=1,
        key="top_k",
        on_change=lambda: state.update(run_online=True),
    )

loc   = state.location_style.strip() or "World"
per   = state.period.strip()         or "daytime"
top_k = state.top_k

# ---- Upload ----
user_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if user_image:
    img_bytes = user_image.getvalue()

    # ① show imme
    st.image(img_bytes, width=500, caption="Uploaded Image")

    # ② if new re-classify and search
    if img_bytes != state.last_upload:
        state.last_upload = img_bytes
        with st.spinner("Classifying image and retrieving local matches…"):
            state.results = classify_and_retrieve(img_bytes)
        state.run_online = False
        state.fav_image_path = None
        state.cached_online  = None

    res = state.results

    # ---- show results----
    st.success(f"✅ Predicted Style: **{res['style']}**")
    st.info   (f"Confidence Score: **{res['conf']*100:.2f}%**")
    st.caption(f"🕒 Runtime (Classification + Local): {res['runtime']:.2f} s")
    st.markdown("---")

    # ---- Local Top-10 ----
    st.subheader("🔍 Top 10 Similar Local Images")
    cols = st.columns(5)
    for i, item in enumerate(res["local_list"][:10]):
        fixed_path = os.path.join(dataset_root, item["path"]).replace("\\", "/")
        with cols[i % 5]:
            st.image(fixed_path, width=160)
            st.markdown(
                f"<div style='text-align:center;font-size:0.85em;'>"
                f"<b>{item['style']}</b><br/>Sim: {item['similarity']:.2f}"
                f"</div>", unsafe_allow_html=True
            )
            st.markdown(
                f"<div style='text-align:center;'>{image_download_link(fixed_path,'⬇️ Download')}</div>",
                unsafe_allow_html=True,
            )
            st.button(
                f"⭐ Favorite {i+1}", key=f"fav_{i}",
                on_click=set_favorite, args=(fixed_path,),
            )

    st.markdown("---")

else:
    st.info("Please upload an image to start.")

# ---- Online Retrieval ----
if state.fav_image_path:
    st.subheader(f"🌐 Top {top_k} Similar Online Images")

    if state.run_online or state.cached_online is None:
        with st.spinner("Querying online search engine…"):
            state.cached_online = retrieve_online(
                state.fav_image_path, loc, per, top_k
            )
        state.run_online = False

    # show cache results
    online_res, rt_online = state.cached_online
    cols = st.columns(2)
    seen, idx = set(), 0
    for url, src, sim in online_res:
        if src in seen: continue
        seen.add(src)
        with cols[idx % 2]:
            st.image(url, width=300, caption=f"Sim {sim:.2f}")
            st.markdown(f"[Source Link]({src})")
        idx += 1
        if idx >= top_k: break
    st.caption(f"🕒 Runtime (Online retrieval): {rt_online:.2f} s")

st.markdown("---")

# ---- Favorite History ----
if state.favorites:
    st.subheader("📂 Favorite History")
    cols = st.columns(5)
    for j, fav in enumerate(state.favorites):
        with cols[j % 5]:
            st.image(fav, width=160, caption=f"Fav #{j+1}")
            st.markdown(image_download_link(fav, "Download Again"), unsafe_allow_html=True)

st.caption("Authors 🧑‍💻 © 2025 Zixuan Yu, Xinyue Du, Ziyin Hu, Jiawei Yang")
st.image("usydlogo.png",width=100)

# Wardrobe Assistant - Streamlit app (version 2: colour-coordinated bottoms + event-aware tops)
# Needs data.zip (made by Section 12 of the Colab notebook) in the same folder as this file.
import io
import json
import os
import random
import re
import zipfile

import numpy as np
import streamlit as st

DATA_ZIP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data (3).zip")
REQUIRED_CONFIG_KEYS = ["EVENT_TOP_ALLOWED", "COLOUR_PAIRING", "BOTTOM_COLOUR_ALIASES", "NEUTRAL_BOTTOMS"]


# ----------------------------------------------------------------------------
# Data loading (runs once, then cached)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_data():
    with zipfile.ZipFile(DATA_ZIP) as z:
        cfg = json.loads(z.read("config.json"))
        items = json.loads(z.read("index.json"))
        embs = np.load(io.BytesIO(z.read("embeddings.npy"))).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9

    b_paths, b_folders, b_colours = [], [], []
    for folder, entries in items["bottom"].items():
        for e in entries:
            b_paths.append(e["path"]); b_folders.append(folder); b_colours.append(e["colour"])

    return {
        "cfg": cfg,
        "paths": [it["path"] for it in items["top"]],
        "folders": np.array([it["folder"] for it in items["top"]]),
        "colors": np.array([it["colors"] for it in items["top"]], dtype=np.float32),
        "embs": embs,
        "b_paths": b_paths,
        "b_folders": b_folders,
        "b_colours": b_colours,
    }


@st.cache_data(max_entries=300, show_spinner=False)
def read_image(path):
    with zipfile.ZipFile(DATA_ZIP) as z:
        return z.read(path)


# ----------------------------------------------------------------------------
# Top wear logic (same as the Colab notebook)
# ----------------------------------------------------------------------------
def parse_description(text, cfg):
    t = text.lower().replace("t shirt", "t-shirt").replace("tshirt", "t-shirt")
    colour = None
    for word in sorted(cfg["COLOR_SYNONYMS"], key=len, reverse=True):
        if re.search(r"\b" + re.escape(word) + r"\b", t):
            colour = cfg["COLOR_SYNONYMS"][word]
            break
    garment_folders = None
    for word in sorted(cfg["GARMENT_KEYWORDS"], key=len, reverse=True):
        if re.search(r"\b" + re.escape(word) + r"\b", t):
            garment_folders = cfg["GARMENT_KEYWORDS"][word]
            break
    return colour, garment_folders


def resolve_top_folders(cfg, garment_folders, event):
    # returns (folders to search, conflict message or None, folders suitable for the event)
    top_folders = cfg["TOP_FOLDERS"]
    allowed = [f for f in cfg["EVENT_TOP_ALLOWED"].get(event, top_folders) if f in top_folders] or list(top_folders)
    if garment_folders is None:
        return allowed, None, allowed
    wanted = [f for f in garment_folders if f in top_folders]
    ok = [f for f in wanted if f in allowed]
    if ok:
        return ok, None, allowed
    msg = (f"'{' / '.join(wanted)}' does not suit the {event} event "
           f"(suitable top wear: {', '.join(allowed)}).")
    return None, msg, allowed


def predict_top_wear(data, description, event, k=3, exclude=None):
    cfg = data["cfg"]
    color_names = cfg["COLOR_NAMES"]
    colour, garment_folders = parse_description(description, cfg)
    folders, conflict, allowed = resolve_top_folders(cfg, garment_folders, event)
    info = {"colour": colour, "folders": folders, "conflict": conflict,
            "blocked": False, "substituted": False}

    if conflict:
        if cfg.get("CONFLICT_MODE", "substitute") == "block":
            info["blocked"] = True
            return [], info
        folders = allowed                      # substitute: search the suitable folders instead
        info["folders"] = folders
        info["substituted"] = True

    ids = np.where(np.isin(data["folders"], folders))[0]
    if len(ids) == 0:
        return [], info

    centroid = data["embs"][ids].mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-9
    sim = data["embs"][ids] @ centroid
    sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-9)

    if colour and colour in color_names:
        score = 0.7 * data["colors"][ids, color_names.index(colour)] + 0.3 * sim
    else:
        score = sim

    order = list(np.argsort(-score))
    best = score[order[0]]
    variety = max(1, int(cfg.get("TOP_VARIETY", 1)))
    margin = float(cfg.get("VARIETY_MARGIN", 0.15))
    near_best = [o for o in order[:variety] if score[o] >= best - margin]
    if colour and colour in color_names:        # every candidate must still show the requested colour
        ci = color_names.index(colour)
        same_colour = [o for o in near_best if int(np.argmax(data["colors"][ids[o]])) == ci]
        near_best = same_colour or [order[0]]
    if exclude:                                 # "shuffle top wear": avoid showing the same item again
        others = [o for o in near_best if data["paths"][ids[o]] != exclude]
        if others:
            near_best = others
        else:
            order = [o for o in order if data["paths"][ids[o]] != exclude] or order
            near_best = order[:1]
    chosen = random.choice(near_best)
    ranked = [chosen] + [o for o in order[:k + 1] if o != chosen][:k - 1]

    results = [
        {
            "path": data["paths"][ids[o]],
            "score": float(score[o]),
            "colour": color_names[int(np.argmax(data["colors"][ids[o]]))],
            "folder": str(data["folders"][ids[o]]),
        }
        for o in ranked
    ]
    return results, info


# ----------------------------------------------------------------------------
# Bottom wear logic: event rules + colour pairing
# ----------------------------------------------------------------------------
def expand_colours(cfg, colours):
    out = []
    for c in colours:
        for a in cfg["BOTTOM_COLOUR_ALIASES"].get(c, [c]):
            if a not in out:
                out.append(a)
    return out


def suggest_bottom_wear(data, event, top_colour=None, exclude=None):
    cfg = data["cfg"]
    folders = cfg["EVENT_BOTTOM_RULES"][event]
    in_event = [i for i, f in enumerate(data["b_folders"]) if f in folders]
    if exclude and len(in_event) > 1:
        in_event = [i for i in in_event if data["b_paths"][i] != exclude]
    if not in_event:
        return None

    paired = expand_colours(cfg, cfg["COLOUR_PAIRING"].get(top_colour, []))
    tiers = []
    if paired:
        tiers.append(("matched", paired))
    tiers.append(("neutral", cfg["NEUTRAL_BOTTOMS"]))

    for tier, wanted in tiers:
        pool = [i for i in in_event if data["b_colours"][i] in wanted]
        if pool:
            i = random.choice(pool)
            return {"path": data["b_paths"][i], "folder": data["b_folders"][i],
                    "colour": data["b_colours"][i], "tier": tier, "paired": paired}

    i = random.choice(in_event)                 # last resort
    return {"path": data["b_paths"][i], "folder": data["b_folders"][i],
            "colour": data["b_colours"][i], "tier": "any", "paired": paired}


def pairing_note(bottom, top_colour, event):
    if bottom["tier"] == "matched":
        return f"Colour pairing: {top_colour} top + {bottom['colour']} bottom"
    if bottom["tier"] == "neutral":
        return (f"No {'/'.join(bottom['paired'])} bottom wear is available for a {event} event, "
                f"so a neutral colour ({bottom['colour']}) was used.")
    return "No suitable colour was available for this event, so the bottom wear was picked at random."


# ----------------------------------------------------------------------------
# Small helper so the app works on both new and older Streamlit versions
# ----------------------------------------------------------------------------
def stretch(fn, *args, **kwargs):
    try:
        return fn(*args, width="stretch", **kwargs)
    except Exception:
        return fn(*args, use_container_width=True, **kwargs)


# ----------------------------------------------------------------------------
# User interface
# ----------------------------------------------------------------------------
def build_outfit(data, description, event, exclude_top=None, exclude_bottom=None):
    results, info = predict_top_wear(data, description, event, exclude=exclude_top)
    outfit = {"description": description, "event": event, "results": results, "info": info, "bottom": None}
    if results:
        outfit["bottom"] = suggest_bottom_wear(data, event, top_colour=results[0]["colour"], exclude=exclude_bottom)
    return outfit


def main():
    st.set_page_config(page_title="Wardrobe Assistant", page_icon="👗", layout="centered")
    st.title("👗 Wardrobe Assistant")
    st.caption("Describe your top wear, pick an event, and get a colour-matched outfit.")

    if not os.path.exists(DATA_ZIP):
        st.error("data.zip not found. Run Section 12 in the Colab notebook and upload data.zip next to app.py.")
        st.stop()

    data = load_data()
    if any(k not in data["cfg"] for k in REQUIRED_CONFIG_KEYS):
        st.error("This data.zip was made with an older notebook. Re-run Sections 3-7b and the new Section 12 "
                 "in Colab, then upload the new data.zip.")
        st.stop()
    events = list(data["cfg"]["EVENT_BOTTOM_RULES"].keys())

    with st.form("search"):
        description = st.text_input("Top wear", placeholder="e.g. white t shirt, green shirt")
        event = st.selectbox("Event type", events)
        submitted = stretch(st.form_submit_button, "Get outfit")

    if submitted:
        if not description.strip():
            st.warning("Please describe your top wear first.")
        else:
            st.session_state["outfit"] = build_outfit(data, description, event)

    outfit = st.session_state.get("outfit")
    if not outfit:
        return

    info = outfit["info"]
    if info["conflict"]:
        st.warning(info["conflict"])
    if info["blocked"]:
        st.error("Please change the top wear or the event type and try again.")
        return
    if not outfit["results"]:
        st.error("No matching top wear found in the collection.")
        return
    if info["substituted"]:
        what = " / ".join(info["folders"])
        colour_txt = f" in {info['colour']}" if info["colour"] else ""
        st.info(f"Showing the best matching {what}{colour_txt} instead.")
    elif info["colour"] is None:
        st.info("No colour recognised in your description, showing the most typical match.")

    best = outfit["results"][0]
    bottom = outfit["bottom"]
    st.subheader(f"{outfit['event'].title()} outfit")
    c1, c2 = st.columns(2)
    with c1:
        stretch(st.image, read_image(best["path"]))
        st.markdown(f"**Top wear**  \n{best['folder'].title()} · {best['colour']}")
    with c2:
        if bottom:
            stretch(st.image, read_image(bottom["path"]))
            st.markdown(f"**Bottom wear**  \n{bottom['folder'].title()} · {bottom['colour']}")
        else:
            st.warning("No bottom wear images found for this event.")
    if bottom:
        st.caption(pairing_note(bottom, best["colour"], outfit["event"]))

    b1, b2 = st.columns(2)
    with b1:
        if stretch(st.button, "Shuffle top wear"):
            st.session_state["outfit"] = build_outfit(data, outfit["description"], outfit["event"],
                                                      exclude_top=best["path"])
            st.rerun()
    with b2:
        if bottom and stretch(st.button, "Shuffle bottom wear"):
            outfit["bottom"] = suggest_bottom_wear(data, outfit["event"], top_colour=best["colour"],
                                                   exclude=bottom["path"])
            st.rerun()

    if len(outfit["results"]) > 1:
        with st.expander("Other top wear matches"):
            cols = st.columns(len(outfit["results"]) - 1)
            for col, r in zip(cols, outfit["results"][1:]):
                with col:
                    stretch(st.image, read_image(r["path"]))
                    st.caption(f"{r['colour']} · score {r['score']:.2f}")


if __name__ == "__main__":
    main()

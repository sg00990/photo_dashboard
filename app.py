"""
📸 Photo EXIF Dashboard
Drag in a folder (as a .zip) or select a batch of photos and get a personalized
shooting-habits dashboard: gear usage, aperture/focal length/ISO/shutter histograms,
time-of-day + calendar heatmap, GPS map, exposure-triangle scatter, and plain-English
"AI insights" derived from your own data.

Run with:  streamlit run app.py
"""

import io
import os
import tempfile
import zipfile
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import ExifTags, Image

# Optional HEIC support (iPhone photos). Falls back gracefully if not installed.
try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIC_SUPPORTED = True
except Exception:
    HEIC_SUPPORTED = False

IMAGE_EXTS = {".jpg", ".jpeg", ".tif", ".tiff", ".png"}
if HEIC_SUPPORTED:
    IMAGE_EXTS |= {".heic", ".heif"}

st.set_page_config(page_title="Photo EXIF Dashboard", page_icon="📸", layout="wide")

# ----------------------------------------------------------------------------
# EXIF EXTRACTION
# ----------------------------------------------------------------------------

def _get_raw_exif(img):
    """Return a dict of {tag_name: value} for an opened PIL Image."""
    exif_data = {}
    try:
        exif = img.getexif()
    except Exception:
        exif = None
    if not exif:
        return exif_data

    for tag_id, value in exif.items():
        tag = ExifTags.TAGS.get(tag_id, tag_id)
        exif_data[tag] = value

    # Grab the "Exif IFD" sub-block, which holds FNumber, ISO, LensModel, etc.
    try:
        ifd = exif.get_ifd(ExifTags.IFD.Exif)
        for tag_id, value in ifd.items():
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            exif_data[tag] = value
    except Exception:
        pass

    # GPS sub-block
    try:
        gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
        if gps_ifd:
            gps_data = {}
            for key, val in gps_ifd.items():
                name = ExifTags.GPSTAGS.get(key, key)
                gps_data[name] = val
            exif_data["GPSInfo"] = gps_data
    except Exception:
        pass

    return exif_data


def _to_float(value):
    """Coerce IFDRational / tuple-fraction / plain number EXIF values to float."""
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        try:
            return float(value[0]) / float(value[1])
        except Exception:
            return None


def _deg_to_decimal(dms, ref):
    try:
        d, m, s = [float(x) for x in dms]
        dec = d + m / 60.0 + s / 3600.0
        if ref in ("S", "W"):
            dec = -dec
        return dec
    except Exception:
        return None


def _shutter_label(exposure_seconds):
    if exposure_seconds is None or exposure_seconds <= 0:
        return None
    if exposure_seconds >= 1:
        return f"{exposure_seconds:.1f}s"
    denom = round(1 / exposure_seconds)
    return f"1/{denom}"


def extract_row(filename, exif):
    make = str(exif.get("Make", "")).strip().strip("\x00")
    model = str(exif.get("Model", "")).strip().strip("\x00")
    if make and model:
        # Many manufacturers (Canon, Nikon...) repeat the make inside the model
        # string already (e.g. Model="Canon EOS R50"), so avoid "Canon Canon EOS R50".
        camera = model if model.lower().startswith(make.lower()) else f"{make} {model}"
    else:
        camera = model or make
    camera = camera.strip()
    if not camera:
        camera = "Unknown Camera"

    lens = exif.get("LensModel") or exif.get("LensSpecification") or exif.get("LensMake")
    lens = str(lens).strip().strip("\x00") if lens else "Unknown Lens"

    focal_length = _to_float(exif.get("FocalLength"))
    focal_35mm = _to_float(exif.get("FocalLengthIn35mmFilm"))
    fnumber = _to_float(exif.get("FNumber"))

    iso = exif.get("ISOSpeedRatings") or exif.get("PhotographicSensitivity") or exif.get("ISO")
    if isinstance(iso, (tuple, list)):
        iso = iso[0] if len(iso) else None
    try:
        iso = int(iso) if iso is not None else None
    except Exception:
        iso = None

    exposure_time = _to_float(exif.get("ExposureTime"))

    dt_raw = exif.get("DateTimeOriginal") or exif.get("DateTime")
    dt = None
    if dt_raw:
        try:
            dt = datetime.strptime(str(dt_raw).strip("\x00"), "%Y:%m:%d %H:%M:%S")
        except Exception:
            dt = None

    lat, lon = None, None
    gps_info = exif.get("GPSInfo")
    if gps_info:
        try:
            lat = _deg_to_decimal(gps_info.get("GPSLatitude"), gps_info.get("GPSLatitudeRef"))
            lon = _deg_to_decimal(gps_info.get("GPSLongitude"), gps_info.get("GPSLongitudeRef"))
        except Exception:
            pass

    return {
        "filename": filename,
        "camera": camera,
        "lens": lens,
        "focal_length": focal_length,
        "focal_35mm": focal_35mm if focal_35mm else focal_length,
        "fnumber": fnumber,
        "iso": iso,
        "exposure_time": exposure_time,
        "shutter_label": _shutter_label(exposure_time),
        "datetime": dt,
        "lat": lat,
        "lon": lon,
        "has_gps": lat is not None and lon is not None,
    }


def load_images_from_uploads(uploaded_files):
    """Accepts a list of Streamlit UploadedFile objects (images and/or one or
    more .zip archives) and returns a list of (filename, raw_bytes) tuples.

    Important: we deliberately keep the ORIGINAL file bytes rather than
    decoding+re-encoding through PIL, because re-saving a PIL Image without
    explicitly forwarding its `exif` bytes silently strips all EXIF metadata.
    """
    images = []

    for uf in uploaded_files:
        name = uf.name
        ext = os.path.splitext(name)[1].lower()

        if ext == ".zip":
            try:
                with zipfile.ZipFile(io.BytesIO(uf.read())) as zf:
                    for member in zf.namelist():
                        m_ext = os.path.splitext(member)[1].lower()
                        if m_ext in IMAGE_EXTS and not member.startswith("__MACOSX"):
                            try:
                                data = zf.read(member)
                                # Validate it's actually openable; keep raw bytes.
                                Image.open(io.BytesIO(data)).verify()
                                images.append((os.path.basename(member), data))
                            except Exception:
                                continue
            except zipfile.BadZipFile:
                st.warning(f"Could not open '{name}' as a zip file — skipping.")
        elif ext in IMAGE_EXTS:
            try:
                data = uf.read()
                Image.open(io.BytesIO(data)).verify()
                images.append((name, data))
            except Exception:
                continue

    return images


@st.cache_data(show_spinner=False)
def build_dataframe(_images_key, images_data):
    """images_data: list of (filename, raw_bytes). Cached on a hashable key."""
    rows = []
    for filename, raw_bytes in images_data:
        try:
            img = Image.open(io.BytesIO(raw_bytes))
            exif = _get_raw_exif(img)
            rows.append(extract_row(filename, exif))
        except Exception:
            continue
    df = pd.DataFrame(rows)
    return df


# ----------------------------------------------------------------------------
# CHART / STAT HELPERS
# ----------------------------------------------------------------------------

def pct(n, total):
    return 0.0 if total == 0 else 100.0 * n / total


def fnumber_label(f):
    if f is None or (isinstance(f, float) and np.isnan(f)):
        return None
    # Common aperture stops for tidy bucketing/labels
    return f"f/{f:g}"


def bucket_iso(iso):
    if iso is None or (isinstance(iso, float) and np.isnan(iso)):
        return None
    for edge, label in [
        (100, "≤100"), (200, "101-200"), (400, "201-400"),
        (800, "401-800"), (1600, "801-1600"), (3200, "1601-3200"),
        (6400, "3201-6400"),
    ]:
        if iso <= edge:
            return label
    return "6400+"


ISO_BUCKET_ORDER = ["≤100", "101-200", "201-400", "401-800", "801-1600", "1601-3200", "3201-6400", "6400+"]


def calendar_heatmap_fig(df):
    dff = df.dropna(subset=["datetime"]).copy()
    if dff.empty:
        return None
    dff["date"] = dff["datetime"].dt.floor("D")
    counts = dff.groupby("date").size().reset_index(name="count")

    start = counts["date"].min()
    end = counts["date"].max()
    grid_start = start - pd.Timedelta(days=start.weekday())  # back up to Monday
    all_days = pd.date_range(grid_start, end)

    full = pd.DataFrame({"date": all_days})
    full = full.merge(counts, on="date", how="left")
    full["count"] = full["count"].fillna(0)
    full["week"] = (full["date"] - grid_start).dt.days // 7
    full["weekday"] = full["date"].dt.weekday

    n_weeks = int(full["week"].max()) + 1
    z = np.full((7, n_weeks), np.nan)
    text = np.full((7, n_weeks), "", dtype=object)
    for _, row in full.iterrows():
        w, wd = int(row["week"]), int(row["weekday"])
        z[wd][w] = row["count"]
        text[wd][w] = f"{row['date'].strftime('%b %d, %Y')}<br>{int(row['count'])} photo(s)"

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            text=text,
            hoverinfo="text",
            colorscale="Greens",
            showscale=True,
            xgap=3,
            ygap=3,
            colorbar=dict(title="Photos"),
        )
    )
    fig.update_yaxes(
        tickvals=list(range(7)),
        ticktext=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        autorange="reversed",
    )
    # Month tick labels along the top
    month_ticks, month_labels = [], []
    seen_months = set()
    for _, row in full.iterrows():
        key = (row["date"].year, row["date"].month)
        if key not in seen_months:
            seen_months.add(key)
            month_ticks.append(int(row["week"]))
            month_labels.append(row["date"].strftime("%b '%y"))
    fig.update_xaxes(tickvals=month_ticks, ticktext=month_labels, tickangle=0)
    fig.update_layout(height=260, margin=dict(l=40, r=20, t=30, b=20))
    return fig


def golden_hour_flag(hour):
    return (hour in (6, 7, 8)) or (hour in (17, 18, 19, 20))


# ----------------------------------------------------------------------------
# AI-STYLE INSIGHTS (rule-based, computed from the user's own data)
# ----------------------------------------------------------------------------

def generate_insights(df):
    insights = []
    total = len(df)
    if total == 0:
        return insights

    # Lens dominance
    lens_counts = df["lens"].value_counts()
    if len(lens_counts) >= 1:
        top_lens, top_n = lens_counts.index[0], lens_counts.iloc[0]
        insights.append(f"You shot **{pct(top_n, total):.0f}%** of your photos with your **{top_lens}**.")

    # Focal length range concentration (e.g. 35-50mm)
    fl = df["focal_35mm"].dropna()
    if len(fl) >= 5:
        band_counts = {
            "14–24mm (ultra-wide)": ((fl >= 14) & (fl <= 24)).sum(),
            "24–35mm (wide)": ((fl > 24) & (fl <= 35)).sum(),
            "35–50mm (standard)": ((fl > 35) & (fl <= 50)).sum(),
            "50–85mm (short tele)": ((fl > 50) & (fl <= 85)).sum(),
            "85–200mm (tele)": ((fl > 85) & (fl <= 200)).sum(),
            "200mm+ (super-tele)": (fl > 200).sum(),
        }
        top_band, top_band_n = max(band_counts.items(), key=lambda kv: kv[1])
        if top_band_n > 0:
            insights.append(f"**{pct(top_band_n, len(fl)):.0f}%** of your shots fall in the **{top_band}** range.")
        mode_fl = fl.round(0).mode()
        if not mode_fl.empty:
            insights.append(f"Your single most-used focal length is **{int(mode_fl.iloc[0])}mm**.")

    # Aperture
    fn = df["fnumber"].dropna()
    if len(fn) >= 5:
        rounded = fn.round(1)
        mode_ap = rounded.mode()
        if not mode_ap.empty:
            mode_val = mode_ap.iloc[0]
            share = pct((rounded == mode_val).sum(), len(rounded))
            insights.append(f"Your favorite aperture is **f/{mode_val:g}**, used in **{share:.0f}%** of shots.")
        wide_open_share = pct((fn <= 2.0).sum(), len(fn))
        if wide_open_share >= 30:
            insights.append(f"You shoot wide open (f/2 or faster) **{wide_open_share:.0f}%** of the time — you clearly love shallow depth of field.")
        narrow_share = pct((fn >= 8).sum(), len(fn))
        if narrow_share >= 25:
            insights.append(f"You stop down to f/8 or narrower **{narrow_share:.0f}%** of the time, suggesting a lot of landscape or group work.")

    # ISO
    iso = df["iso"].dropna()
    if len(iso) >= 5:
        avg_iso = iso.mean()
        insights.append(f"Your average ISO is **{avg_iso:.0f}**.")
        low_share = pct((iso <= 200).sum(), len(iso))
        if low_share >= 60:
            insights.append(f"**{low_share:.0f}%** of your shots are at ISO 200 or below — you mostly shoot in good light.")
        high_share = pct((iso > 3200).sum(), len(iso))
        if high_share > 0:
            insights.append(f"You shoot above ISO 3200 in **{high_share:.0f}%** of photos.")
        else:
            insights.append("You almost never push above ISO 3200.")

    # Shutter speed
    et = df["exposure_time"].dropna()
    if len(et) >= 5:
        fast_share = pct((et <= 1 / 500).sum(), len(et))
        slow_share = pct((et >= 1 / 30).sum(), len(et))
        if fast_share >= 30:
            insights.append(f"**{fast_share:.0f}%** of your shots are 1/500s or faster — looks like you shoot a fair amount of action or wildlife.")
        if slow_share >= 30:
            insights.append(f"**{slow_share:.0f}%** of your shots are 1/30s or slower, which fits landscapes, tripod work, or low light.")

    # Time of day / golden hour
    dt = df["datetime"].dropna()
    if len(dt) >= 5:
        hours = dt.dt.hour
        golden = hours.apply(golden_hour_flag)
        golden_share = pct(golden.sum(), len(hours))
        if golden_share >= 40:
            insights.append(f"**{golden_share:.0f}%** of your photos are taken during golden hour (early morning or early evening) — you clearly chase the light.")
        night_share = pct(((hours >= 21) | (hours <= 5)).sum(), len(hours))
        if night_share >= 20:
            insights.append(f"**{night_share:.0f}%** of your shots happen at night.")

        weekday_names = dt.dt.day_name()
        weekend_share = pct(weekday_names.isin(["Saturday", "Sunday"]).sum(), len(weekday_names))
        if weekend_share >= 55:
            insights.append(f"Most of your photography happens on weekends (**{weekend_share:.0f}%** of shots).")
        elif weekend_share <= 20:
            insights.append(f"You mostly shoot on weekdays — only **{weekend_share:.0f}%** of your photos are from weekends.")

    # Camera body diversity
    n_cameras = df["camera"].nunique()
    if n_cameras > 1:
        insights.append(f"You shot with **{n_cameras} different camera bodies** across this set.")

    # GPS coverage
    gps_share = pct(df["has_gps"].sum(), total)
    if gps_share >= 50:
        insights.append(f"**{gps_share:.0f}%** of your photos are geotagged — most of your shoots are trackable on a map.")
    elif gps_share == 0:
        insights.append("None of your photos have GPS data, so no map for this batch — that's normal for many cameras without built-in GPS.")

    return insights


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------

st.title("📸 Photo EXIF Dashboard")
st.caption("Drop in a folder of photos as a **.zip**, or select multiple images directly. Everything runs locally in this session — nothing is uploaded anywhere else.")

with st.sidebar:
    st.header("Import")
    uploaded_files = st.file_uploader(
        "Drag a .zip of photos, or select images",
        type=["zip", "jpg", "jpeg", "tif", "tiff", "png"] + (["heic", "heif"] if HEIC_SUPPORTED else []),
        accept_multiple_files=True,
        help="Browsers can't drag whole folders into a file input, so zip your folder first for the easiest import.",
    )
    if not HEIC_SUPPORTED:
        st.caption("ℹ️ Install `pillow-heif` to support iPhone .HEIC files.")
    st.markdown("---")
    st.caption("EXIF data (camera settings, timestamps, GPS) is read directly from your files in-browser/session memory — nothing is sent to any AI or third party.")

if not uploaded_files:
    st.info("👈 Upload a .zip of your photo folder or select images to get started.")
    st.markdown(
        """
        **What you'll get:**
        - 🎯 Gear usage: camera body, lens breakdown, favorite focal length
        - 🔘 Aperture, focal length, ISO, and shutter speed histograms
        - 🕒 Time-of-day patterns and a GitHub-style shooting calendar
        - 🗺️ A map of every geotagged photo
        - 📈 An ISO vs. aperture exposure-triangle scatter plot
        - 🤖 Plain-English insights about your own shooting habits
        """
    )
    st.stop()

with st.spinner("Reading files..."):
    images = load_images_from_uploads(uploaded_files)

if not images:
    st.error("No readable images were found in your upload. Make sure your zip contains .jpg/.jpeg/.tif/.png files (or .heic if pillow-heif is installed).")
    st.stop()

# `images` is already a list of (filename, raw_bytes) — no re-encoding needed,
# which is what keeps EXIF metadata intact.
images_data = images
cache_key = f"{len(images_data)}-{sum(len(b) for _, b in images_data)}"

with st.spinner("Extracting EXIF metadata..."):
    df = build_dataframe(cache_key, images_data)

if df.empty:
    st.error("Images were found, but none contained readable EXIF metadata.")
    st.stop()

n_total = len(df)
n_with_exif = df[["camera", "focal_length", "fnumber", "iso"]].notna().any(axis=1).sum()
st.success(f"Loaded **{n_total}** photos — {n_with_exif} with usable EXIF data.")

tabs = st.tabs(["🛠️ Gear", "🔘 Aperture", "📏 Focal Length", "🎛️ ISO", "⚡ Shutter Speed",
                "🕒 Time of Day", "📅 Calendar", "🗺️ Map", "📈 Exposure Triangle", "🤖 AI Insights"])

# ---------------- GEAR ----------------
with tabs[0]:
    st.subheader("Gear")
    col1, col2 = st.columns(2)

    with col1:
        cam_counts = df["camera"].value_counts()
        st.markdown("**Camera body**")
        for cam, n in cam_counts.items():
            st.write(f"- {cam} — {pct(n, n_total):.0f}% ({n} photos)")

    with col2:
        lens_counts = df["lens"].value_counts()
        st.markdown("**Lens usage**")
        for lens, n in lens_counts.items():
            st.write(f"- {lens} — {pct(n, n_total):.0f}% ({n} photos)")

    lens_df = lens_counts.rename_axis("lens").reset_index(name="count")
    fig_lens = px.bar(
        lens_df,
        x="count", y="lens", orientation="h", title="Photos per lens",
    )
    fig_lens.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_lens, use_container_width=True)

    fl_mode = df["focal_35mm"].dropna()
    if not fl_mode.empty:
        favorite_fl = fl_mode.round(0).mode()
        if not favorite_fl.empty:
            st.metric("Favorite focal length", f"{int(favorite_fl.iloc[0])}mm")

    st.markdown("**Zoom range usage per lens**")
    zoom_rows = []
    for lens, group in df.groupby("lens"):
        fls = group["focal_35mm"].dropna()
        if fls.empty:
            continue
        zoom_rows.append({
            "Lens": lens,
            "Min focal length": f"{fls.min():.0f}mm",
            "Max focal length": f"{fls.max():.0f}mm",
            "Most-used": f"{fls.round(0).mode().iloc[0]:.0f}mm" if not fls.round(0).mode().empty else "—",
            "Shots": len(group),
        })
    if zoom_rows:
        st.dataframe(pd.DataFrame(zoom_rows), use_container_width=True, hide_index=True)

# ---------------- APERTURE ----------------
with tabs[1]:
    st.subheader("Aperture")
    fn = df["fnumber"].dropna()
    if fn.empty:
        st.warning("No aperture (FNumber) data found in these photos.")
    else:
        fig = px.histogram(fn.round(1), nbins=30, labels={"value": "f-number"}, title="Aperture distribution")
        fig.update_layout(xaxis_title="Aperture (f/)", yaxis_title="Number of photos", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        mode_ap = fn.round(1).mode()
        if not mode_ap.empty:
            share = pct((fn.round(1) == mode_ap.iloc[0]).sum(), len(fn))
            st.info(f"You shoot at **f/{mode_ap.iloc[0]:g}** almost **{share:.0f}%** of the time.")

# ---------------- FOCAL LENGTH ----------------
with tabs[2]:
    st.subheader("Focal Length")
    fl = df["focal_35mm"].dropna()
    if fl.empty:
        st.warning("No focal length data found in these photos.")
    else:
        fig = px.histogram(fl, nbins=40, labels={"value": "mm"}, title="Focal length distribution (35mm equivalent)")
        fig.update_layout(xaxis_title="Focal length (mm)", yaxis_title="Number of photos", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        top3 = fl.round(0).value_counts().head(3)
        st.markdown("**Top 3 focal lengths**")
        for val, n in top3.items():
            st.write(f"- {int(val)}mm — {pct(n, len(fl)):.0f}%")

# ---------------- ISO ----------------
with tabs[3]:
    st.subheader("ISO")
    iso = df["iso"].dropna()
    if iso.empty:
        st.warning("No ISO data found in these photos.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Average ISO", f"{iso.mean():.0f}")
        col2.metric("Above ISO 3200", f"{pct((iso > 3200).sum(), len(iso)):.0f}%")
        col3.metric("At base ISO (≤200)", f"{pct((iso <= 200).sum(), len(iso)):.0f}%")

        bucketed = iso.apply(bucket_iso)
        order = [b for b in ISO_BUCKET_ORDER if b in bucketed.unique()]
        fig = px.histogram(bucketed, category_orders={"value": order}, title="ISO distribution")
        fig.update_layout(xaxis_title="ISO range", yaxis_title="Number of photos", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ---------------- SHUTTER SPEED ----------------
with tabs[4]:
    st.subheader("Shutter Speed")
    et = df["exposure_time"].dropna()
    if et.empty:
        st.warning("No shutter speed data found in these photos.")
    else:
        labels = et.apply(_shutter_label)
        # Order labels by actual speed (fast -> slow)
        order_df = pd.DataFrame({"label": labels, "value": et}).drop_duplicates("label").sort_values("value", ascending=False)
        fig = px.histogram(labels, category_orders={"value": order_df["label"].tolist()}, title="Shutter speed distribution")
        fig.update_layout(xaxis_title="Shutter speed", yaxis_title="Number of photos", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        col1.metric("1/500s or faster", f"{pct((et <= 1/500).sum(), len(et)):.0f}%")
        col2.metric("1/30s or slower", f"{pct((et >= 1/30).sum(), len(et)):.0f}%")

# ---------------- TIME OF DAY ----------------
with tabs[5]:
    st.subheader("Time of Day")
    dt = df["datetime"].dropna()
    if dt.empty:
        st.warning("No timestamp data found in these photos.")
    else:
        hours = dt.dt.hour + dt.dt.minute / 60.0
        fig = px.histogram(hours, nbins=48, title="What time of day do you shoot?")
        fig.update_layout(xaxis_title="Hour of day", yaxis_title="Number of photos", showlegend=False,
                           xaxis=dict(tickmode="array", tickvals=list(range(0, 25, 3)),
                                      ticktext=[f"{h%24:02d}:00" for h in range(0, 25, 3)]))
        st.plotly_chart(fig, use_container_width=True)

        golden_share = pct(dt.dt.hour.apply(golden_hour_flag).sum(), len(dt))
        st.info(f"**{golden_share:.0f}%** of your photos are taken during golden hour (6–9 AM or 5–9 PM).")

# ---------------- CALENDAR ----------------
with tabs[6]:
    st.subheader("Calendar Heatmap")
    fig = calendar_heatmap_fig(df)
    if fig is None:
        st.warning("No timestamp data found in these photos.")
    else:
        st.plotly_chart(fig, use_container_width=True)
        dt = df["datetime"].dropna()
        n_days = dt.dt.floor("D").nunique()
        st.caption(f"You took photos on **{n_days}** different days in this batch.")

# ---------------- MAP ----------------
with tabs[7]:
    st.subheader("Map")
    geo = df.dropna(subset=["lat", "lon"])
    if geo.empty:
        st.warning("No GPS data found in these photos.")
    else:
        st.map(geo.rename(columns={"lat": "latitude", "lon": "longitude"})[["latitude", "longitude"]])
        st.caption(f"{len(geo)} of {n_total} photos are geotagged ({pct(len(geo), n_total):.0f}%).")

# ---------------- EXPOSURE TRIANGLE ----------------
with tabs[8]:
    st.subheader("Exposure Triangle")
    trip = df.dropna(subset=["iso", "fnumber", "exposure_time"]).copy()
    if trip.empty:
        st.warning("Not enough ISO / aperture / shutter speed data to plot the exposure triangle.")
    else:
        trip["shutter_seconds"] = trip["exposure_time"]
        fig = px.scatter(
            trip, x="fnumber", y="iso", color="shutter_seconds",
            color_continuous_scale="Viridis",
            labels={"fnumber": "Aperture (f/)", "iso": "ISO", "shutter_seconds": "Shutter (s)"},
            hover_data=["filename"],
            title="ISO vs. Aperture, colored by shutter speed",
        )
        fig.update_yaxes(type="log")
        st.plotly_chart(fig, use_container_width=True)

# ---------------- AI INSIGHTS ----------------
with tabs[9]:
    st.subheader("🤖 AI Insights")
    st.caption("Plain-English observations, computed directly from your photos' metadata.")
    insights = generate_insights(df)
    if not insights:
        st.info("Not enough data yet to generate insights — try uploading more photos.")
    else:
        for line in insights:
            st.markdown(f"- {line}")

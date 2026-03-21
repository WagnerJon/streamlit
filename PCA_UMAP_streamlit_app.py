import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import zipfile
import threading
import socket
import http.server
import urllib.parse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import silhouette_score
import umap
import matplotlib.pyplot as plt
import seaborn as sns

# Visualization
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, CustomJS, Div
from bokeh.layouts import row
from bokeh.transform import factor_cmap, linear_cmap
from bokeh.palettes import Turbo256, Viridis256, Category10_10

# -----------------------------------------------------------------------------
# 1. Julia-Compatible Constants
# -----------------------------------------------------------------------------

COLUMNS_TO_KEEP = [
    "experiment", "chip", "position", "timepoint", "z_plane", 
    "mean_intensity", "area", "eccentricity", "extent", "orientation", 
    "perimeter", "perimeter_crofton", "solidity", "equivalent_diameter_area", 
    "euler_number", "feret_diameter_max", "axis_major_length", "axis_minor_length", 
    "circularity", "rfp_mean_intensity", "rfp_max_intensity", "rfp_min_intensity", 
    "rfp_overlap_area", "rfp_overlap_fraction", "gfp_mean_intensity", "gfp_max_intensity", 
    "gfp_min_intensity", "gfp_overlap_area", "gfp_overlap_fraction", "dist_to_nearest_gfp", 
    "rfp_mean_prob", "rfp_max_prob", "rfp_min_prob", "gfp_mean_prob", 
    "gfp_max_prob", "gfp_min_prob", "confidence",
    "crop_path", "crop_zip", "crop_filename", "object_uid"
]

CATEGORICAL_COLS = ["experiment", "chip", "position", "z_plane"]

PCA_EXCLUDE_COLS = CATEGORICAL_COLS + [
    "timepoint", "crop_path", "crop_zip", "crop_filename", "object_uid", 
    "mask_path", "track_id", "group", "class_id", "label"
]

# -----------------------------------------------------------------------------
# 2. Background Image Server
# -----------------------------------------------------------------------------
PORT = 8503 

class ZipCache:
    _cache = {}
    @classmethod
    def get_file(cls, zip_path, filename):
        if zip_path not in cls._cache:
            if os.path.exists(zip_path):
                try: cls._cache[zip_path] = zipfile.ZipFile(zip_path, 'r')
                except: return None
            else: return None
        try: return cls._cache[zip_path].read(filename)
        except:
            try:
                cls._cache[zip_path] = zipfile.ZipFile(zip_path, 'r')
                return cls._cache[zip_path].read(filename)
            except: return None

class ImageRequestHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args): pass
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        img_data = None
        try:
            if 'zip' in params and 'file' in params:
                img_data = ZipCache.get_file(params['zip'][0], params['file'][0])
            if img_data:
                self.send_response(200)
                self.send_header('Content-type', 'image/png')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(img_data)
                return
        except: pass
        self.send_response(404)
        self.end_headers()

def start_server():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if sock.connect_ex(('127.0.0.1', PORT)) != 0:
        threading.Thread(target=lambda: http.server.HTTPServer(('', PORT), ImageRequestHandler).serve_forever(), daemon=True).start()
    sock.close()

start_server()

# -----------------------------------------------------------------------------
# 3. Data Loading & Preprocessing
# -----------------------------------------------------------------------------

st.set_page_config(layout="wide", page_title="Protist Interactive UMAP")

@st.cache_data
def load_data_from_folder(folder_path):
    if not os.path.isdir(folder_path): 
        st.error(f"Folder not found: {folder_path}"); return pd.DataFrame()
    files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not files: st.warning("No CSVs found."); return pd.DataFrame()
    
    dfs = []
    for f in files:
        try:
            t = pd.read_csv(f)
            if "experiment" not in t.columns: 
                t["experiment"] = os.path.basename(f).replace(".csv", "")
            dfs.append(t)
        except: pass
        
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def load_data_from_uploads(uploaded_files):
    dfs = []
    for f in uploaded_files:
        try:
            f.seek(0)
            t = pd.read_csv(f)
            # Use filename as experiment name if not present
            if "experiment" not in t.columns:
                t["experiment"] = f.name.replace(".csv", "")
            dfs.append(t)
        except: pass
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def get_numeric_columns(df_in):
    if df_in.empty: return []
    df = df_in.copy()
    
    potential_pca_cols = [c for c in df.columns if c not in PCA_EXCLUDE_COLS]
    numeric_cols = []
    for c in potential_pca_cols:
        # Check if already numeric or convertible
        if pd.api.types.is_numeric_dtype(df[c]):
             numeric_cols.append(c)
        else:
             # Try convert
             s = pd.to_numeric(df[c], errors='coerce')
             if s.notna().sum() > 0: # If at least some are numbers
                 numeric_cols.append(c)
    return numeric_cols

def prepare_features(df, cols_to_use):
    """
    Helper to subset, cleanup, imputation and scale features.
    """
    if df.empty or not cols_to_use: return None, None

    df_sub = df.copy()
    
    # Imputation (Basic specific logic)
    if "dist_to_nearest_gfp" in df_sub.columns:
        df_sub["dist_to_nearest_gfp"] = df_sub["dist_to_nearest_gfp"].fillna(1500.0)
        df_sub["dist_to_nearest_gfp"] = df_sub["dist_to_nearest_gfp"].replace([np.inf, -np.inf], 1500.0)

    # Convert to numeric
    for c in cols_to_use:
        df_sub[c] = pd.to_numeric(df_sub[c], errors='coerce')

    # Handle Inf -> NaN
    df_sub[cols_to_use] = df_sub[cols_to_use].replace([np.inf, -np.inf], np.nan)
    
    # Drop Missing
    df_clean = df_sub.dropna(subset=cols_to_use).reset_index(drop=True)
    
    if df_clean.empty:
        return None, None
        
    # Extract
    features = df_clean[cols_to_use].values
    
    # Variance Filter
    selector = VarianceThreshold(threshold=0.0)
    try:
        features = selector.fit_transform(features)
    except ValueError:
        return None, None # All zero variance

    # Scale
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    return df_clean, features_scaled

# -----------------------------------------------------------------------------
# 4. Visualization (Bokeh)
# -----------------------------------------------------------------------------

def bokeh_plot(df, x_col, y_col, color_col):
    def get_url(row):
        base = f"http://localhost:{PORT}/?"
        if pd.notna(row.get("crop_zip")) and pd.notna(row.get("crop_filename")):
            z_abs = os.path.abspath(os.path.join(st.session_state.data_folder, str(row["crop_zip"])))
            f_enc = urllib.parse.quote(str(row['crop_filename']))
            z_enc = urllib.parse.quote(z_abs)
            return f"{base}zip={z_enc}&file={f_enc}"
        return ""

    df["_img_url"] = df.apply(get_url, axis=1)

    if not pd.api.types.is_numeric_dtype(df[color_col]):
        df[color_col] = df[color_col].astype(str)
        factors = sorted(df[color_col].unique())
        pal = Category10_10 if len(factors) <= 10 else list(Turbo256) * (len(factors)//256 + 1)
        mapper = factor_cmap(color_col, palette=pal, factors=factors)
    else:
        mapper = linear_cmap(color_col, palette=Viridis256, low=df[color_col].min(), high=df[color_col].max())

    source = ColumnDataSource(df)

    p = figure(tools="pan,wheel_zoom,box_select,reset", active_scroll="wheel_zoom", 
               title=f"Projection ({len(df)} objects)",
               height=750, sizing_mode="stretch_width")
    
    p.scatter(x_col, y_col, source=source, 
              size=6, fill_alpha=0.7, line_color=None,
              color=mapper)

    tooltip_div = Div(text="""
        <div style="background:#fcfcfc; padding:10px; border:1px solid #ddd; width:220px; font-family:sans-serif; height:730px; overflow-y:auto;">
            <h4 style="margin-top:0; color:#333;">Cell Inspector</h4>
            <p style="color:#777; font-size:12px;">Hover over points to see nearest neighbors.</p>
        </div>
    """, width=250)

    js_code = """
        const data = source.data;
        const x = data[x_col];
        const y = data[y_col];
        const urls = data['_img_url'];
        const uids = data['object_uid'];
        const exps = data['experiment'];
        
        const mx = cb_obj.x;
        const my = cb_obj.y;
        
        let candidates = [];
        const n = x.length;
        
        for (let i = 0; i < n; i++) {
            const dx = x[i] - mx;
            const dy = y[i] - my;
            const d2 = dx*dx + dy*dy;
            candidates.push({idx: i, dist: d2});
        }
        
        candidates.sort((a, b) => a.dist - b.dist);
        const top5 = candidates.slice(0, 5);
        
        let html = '<div style="font-family:sans-serif; padding:5px;">';
        html += '<h4 style="margin:0 0 10px 0; border-bottom:1px solid #ccc; font-size:14px;">Nearest Objects</h4>';
        
        for (let k=0; k<top5.length; k++) {
            const i = top5[k].idx;
            const url = urls[i];
            
            html += '<div style="display:flex; align-items:center; margin-bottom:8px; border-bottom:1px solid #eee; padding-bottom:4px;">';
            
            if (url) {
                html += `<img src="${url}" style="width:60px; height:60px; object-fit:contain; background:black; border:1px solid #ccc; margin-right:8px;">`;
            } else {
                html += `<div style="width:60px; height:60px; background:#eee; margin-right:8px; display:flex; align-items:center; justify-content:center; font-size:10px;">No Img</div>`;
            }
            
            html += '<div>';
            html += `<div style="font-weight:bold; font-size:11px;">${uids[i]}</div>`;
            html += `<div style="font-size:10px; color:#555;">${exps[i]}</div>`;
            if (data['DBSCAN']) {
                html += `<div style="font-size:10px; color:blue;">Cluster: ${data['DBSCAN'][i]}</div>`;
            }
            html += '</div></div>';
        }
        html += '</div>';
        div.text = html;
    """

    callback = CustomJS(args=dict(source=source, div=tooltip_div, x_col=x_col, y_col=y_col), code=js_code)
    p.js_on_event('mousemove', callback)
    
    return row(p, tooltip_div)


# -----------------------------------------------------------------------------
# 5. Modular Tabs
# -----------------------------------------------------------------------------

def umap_dbscan_tab(df_subset, all_numeric_cols):
    st.markdown("### UMAP & DBSCAN")
    
    # --- Feature Selection Source ---
    use_pca_features = st.checkbox("Use features from 'PCA Exploration' tab?", value=False)
    
    if use_pca_features:
        if "pca_selected_features" in st.session_state and st.session_state.pca_selected_features:
             features_to_use = st.session_state.pca_selected_features
             st.info(f"Using {len(features_to_use)} features from PCA tab: {', '.join(features_to_use[:5])}...")
        else:
             st.warning("No features selected in PCA tab. Falling back to default.")
             features_to_use = [c for c in COLUMNS_TO_KEEP if c in all_numeric_cols]
    else:
         features_to_use = [c for c in COLUMNS_TO_KEEP if c in all_numeric_cols]
         st.info(f"Using default features ({len(features_to_use)} vars)")
    
    # --- Modular Controls ---
    c1, c2, c3 = st.columns(3)
    
    # PCA Step
    with c1:
        st.markdown("**1. PCA**")
        n_comp = st.number_input("Comp", 2, 50, 10, key="main_pca_n")
        if st.button("Run PCA", key="btn_main_pca"):
            st.spinner("Running PCA...")
            
            # Preparation
            df_clean, feats_scaled = prepare_features(df_subset, features_to_use)
            if df_clean is not None:
                pca = PCA(n_components=n_comp).fit_transform(feats_scaled)
                st.session_state.main_df = df_clean
                st.session_state.main_pca_res = pca
                st.session_state.main_dbscan_res = None # Reset downstream
                st.session_state.main_umap_res = None
                st.session_state.main_feats_used = features_to_use
                st.rerun()
            else:
                st.error("Prep failed")

    # DBSCAN Step
    with c2:
        st.markdown("**2. DBSCAN**")
        eps = st.number_input("Eps", 0.1, 10.0, 1.0, 0.1, key="main_dbscan_eps")
        if st.button("Run DBSCAN", key="btn_main_dbscan"):
             if "main_pca_res" in st.session_state and st.session_state.main_pca_res is not None:
                 pca_data = st.session_state.main_pca_res
                 lbls = DBSCAN(eps=eps, min_samples=10).fit_predict(pca_data)
                 st.session_state.main_df["DBSCAN"] = [str(l) if l!=-1 else "Noise" for l in lbls]
                 st.session_state.main_dbscan_res = True
                 st.rerun()
             else:
                 st.warning("Run PCA first")

    # UMAP Step
    with c3:
        st.markdown("**3. UMAP**")
        n_neigh = st.number_input("Neighbors", 5, 200, 15, key="main_umap_n")
        if st.button("Run UMAP", key="btn_main_umap"):
             if "main_pca_res" in st.session_state and st.session_state.main_pca_res is not None:
                 pca_data = st.session_state.main_pca_res
                 red = umap.UMAP(n_neighbors=n_neigh, min_dist=0.1, n_components=2)
                 u_res = red.fit_transform(pca_data)
                 st.session_state.main_umap_res = u_res
                 st.rerun()
             else:
                 st.warning("Run PCA first")
                 
    # --- Plotting ---
    if "main_umap_res" in st.session_state and st.session_state.main_umap_res is not None:
        st.divider()
        df_viz = st.session_state.main_df
        coords = st.session_state.main_umap_res
        
        df_viz["UMAP1"] = coords[:, 0]
        df_viz["UMAP2"] = coords[:, 1]
        
        cols = [c for c in df_viz.columns if pd.api.types.is_numeric_dtype(df_viz[c]) and c not in ["UMAP1", "UMAP2"]]
        opts = ["experiment"]
        if "DBSCAN" in df_viz.columns: opts.append("DBSCAN")
        opts += cols
        
        c_mode = st.selectbox("Color By", opts, key="main_color_sel")
        p = bokeh_plot(df_viz, "UMAP1", "UMAP2", c_mode)
        
        # Plot using streamlit-bokeh (as requested)
        try:
            from streamlit_bokeh import st_bokeh_chart
            st_bokeh_chart(p, use_container_width=True)
        except ImportError:
            st.error("Please install streamlit-bokeh: pip install streamlit-bokeh")
            # Fallback to st.write just in case, though likely to fail if st.bokeh_chart is gone
            st.write(p)


def pca_exploration_tab(df_subset, all_numeric_cols):
    st.markdown("### PCA Exploration")
    
    # Feature Selection (Local)
    sel_features = st.multiselect(
        "Select Features for PCA:", 
        options=all_numeric_cols, 
        default=all_numeric_cols[:10],
        key="pca_selected_features" # Shared Key!
    )
    
    if st.button("Run Exploration Analysis", key="btn_exp_pca"):
        if not sel_features:
            st.error("Select features")
            return
            
        df_clean, feats_scaled = prepare_features(df_subset, sel_features)
        
        if df_clean is not None:
            pca = PCA() # Full
            pca.fit(feats_scaled)
            st.session_state.exp_pca_model = pca
            st.session_state.exp_feats_scaled = feats_scaled
            st.session_state.exp_df = df_clean
            st.session_state.exp_feat_names = sel_features
        else:
            st.error("Prep failed")

    if "exp_pca_model" in st.session_state:
        pca = st.session_state.exp_pca_model
        feats = st.session_state.exp_feats_scaled
        df_exp = st.session_state.exp_df
        feat_names = st.session_state.exp_feat_names
        
        # 1. Scree
        st.subheader("1. Variance Explained")
        var_expl = pca.explained_variance_ratio_ * 100
        cum_var = np.cumsum(var_expl)
        
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6,4))
            ax.bar(range(1, len(var_expl)+1)[:30], var_expl[:30])
            ax.set_title("Scree Plot")
            st.pyplot(fig)
            
        with c2:
            fig, ax = plt.subplots(figsize=(6,4))
            ax.plot(range(1, len(var_expl)+1)[:30], cum_var[:30], '-o')
            ax.axhline(90, c='r', ls='--')
            ax.set_title("Cumulative Variance")
            st.pyplot(fig)
            
        # 2. 2D Scatter
        st.subheader("2. PCA Data Projection")
        pca_coords = pca.transform(feats)
        
        c3, c4, c5 = st.columns(3)
        px = c3.number_input("PC X", 1, len(var_expl), 1)
        py = c4.number_input("PC Y", 1, len(var_expl), 2)
        
        c_opts = ["experiment"] + [c for c in df_exp.columns if pd.api.types.is_numeric_dtype(df_exp[c])]
        c_sel = c5.selectbox("Color", c_opts)
        
        plot_df = pd.DataFrame({
            "x": pca_coords[:, px-1],
            "y": pca_coords[:, py-1],
            "col": df_exp[c_sel] if c_sel in df_exp.columns else 0
        })
        
        fig, ax = plt.subplots()
        if pd.api.types.is_numeric_dtype(plot_df["col"]):
            sc = ax.scatter(plot_df.x, plot_df.y, c=plot_df.col, cmap='viridis', s=5, alpha=0.6)
            plt.colorbar(sc)
        else:
            sns.scatterplot(data=plot_df, x="x", y="y", hue="col", s=10, alpha=0.6, ax=ax)
            
        ax.set_xlabel(f"PC{px}")
        ax.set_ylabel(f"PC{py}")
        st.pyplot(fig)
        
        # 3. Loadings
        st.subheader("3. Feature Loadings")
        loadings = pd.DataFrame(pca.components_.T, columns=[f"PC{i+1}" for i in range(len(var_expl))], index=feat_names)
        
        # Robust selection
        max_pc = min(10, len(var_expl))
        pc_list = [f"PC{i+1}" for i in range(max_pc)]
        
        # Check if previous selection is valid
        prev_idx = 0
        if "sel_loadings_pc" in st.session_state and st.session_state.sel_loadings_pc in pc_list:
             pass 
        
        target_pc = st.selectbox("Select PC", pc_list, key="sel_loadings_pc")
        
        top = loadings[target_pc].abs().sort_values(ascending=False).head(10)
        plot_l = loadings.loc[top.index, target_pc]
        
        fig, ax = plt.subplots(figsize=(8,5))
        colors = ['blue' if x>0 else 'red' for x in plot_l]
        plot_l.plot(kind='barh', color=colors, ax=ax)
        st.pyplot(fig)

        # 4. K-Means
        st.subheader("4. K-Means Exploration")
        if st.checkbox("Run K-Means (k=2..15)"):
            X = pca_coords[:, :5]
            if len(X) > 5000:
                idx = np.random.choice(len(X), 5000, replace=False)
                X = X[idx]
                st.info("Subsampled to 5000")
            
            wss, sils = [], []
            rng = range(2, 16)
            for k in rng:
                km = KMeans(n_clusters=k, n_init=10).fit(X)
                wss.append(km.inertia_)
                sils.append(silhouette_score(X, km.labels_))
                
            c_k1, c_k2 = st.columns(2)
            
            with c_k1:
                fig, ax = plt.subplots()
                ax.plot(rng, wss, 'o-')
                ax.set_title("Elbow (WSS)")
                st.pyplot(fig)
            with c_k2:
                fig, ax = plt.subplots()
                ax.plot(rng, sils, 'o-')
                ax.set_title("Silhouette")
                st.pyplot(fig)

# -----------------------------------------------------------------------------
# 6. Main App Structure
# -----------------------------------------------------------------------------

def main():
    st.sidebar.title("Protist Clustering")
    
    # 1. Load
    st.sidebar.markdown("### 1. Data Source")
    data_source = st.sidebar.radio("Source:", ["Folder", "Upload"], horizontal=True, label_visibility="collapsed")
    
    if "df_raw" not in st.session_state: st.session_state.df_raw = pd.DataFrame()
    
    if data_source == "Folder":
        default_path = "/Volumes/vangestel/Jonas/Vanessa_segmentation/Visualizations/Umap_data"
        data_folder = st.sidebar.text_input("Data Folder", value=default_path)
        st.session_state.data_folder = data_folder
        if st.sidebar.button("Load Folder"):
            st.session_state.df_raw = load_data_from_folder(data_folder)
            st.rerun()
    else:
        uploaded_files = st.sidebar.file_uploader("Upload CSVs", type="csv", accept_multiple_files=True)
        st.session_state.data_folder = ""
        if uploaded_files and st.sidebar.button("Load Files"):
            st.session_state.df_raw = load_data_from_uploads(uploaded_files)
            st.rerun()
            
    df = st.session_state.df_raw
    if df.empty:
        st.info("Load data.")
        return
        
    # 2. Filter
    st.sidebar.markdown("### 2. Filter")
    all_exps = sorted(df["experiment"].astype(str).unique())
    sel_exps = st.sidebar.multiselect("Experiments", all_exps, default=all_exps)
    if not sel_exps: return
    
    df_subset = df[df["experiment"].isin(sel_exps)].copy()
    st.sidebar.success(f"Analysing {len(df_subset)} items")
    
    # 3. Tabs
    t1, t2 = st.tabs(["UMAP & DBSCAN", "PCA Exploration"])
    
    all_numeric_cols = get_numeric_columns(df_subset)
    
    with t1:
        umap_dbscan_tab(df_subset, all_numeric_cols)
        
    with t2:
        pca_exploration_tab(df_subset, all_numeric_cols)

if __name__ == "__main__":
    main()
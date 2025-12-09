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

def preprocess_data(df_in):
    if df_in.empty: return None, []
    df = df_in.copy()

    # Imputation
    if "dist_to_nearest_gfp" in df.columns:
        df["dist_to_nearest_gfp"] = df["dist_to_nearest_gfp"].fillna(1500.0)
        df["dist_to_nearest_gfp"] = df["dist_to_nearest_gfp"].replace([np.inf, -np.inf], 1500.0)

    # Determine Numeric Columns
    # We use all potential numeric columns
    potential_pca_cols = [c for c in df.columns if c not in PCA_EXCLUDE_COLS]
    numeric_cols = []
    for c in potential_pca_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        if pd.api.types.is_numeric_dtype(df[c]):
             numeric_cols.append(c)
             
    if not numeric_cols:
        st.error(f"No numeric columns found.")
        return None, []

    # Handle Inf -> NaN
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    
    return df, numeric_cols

# -----------------------------------------------------------------------------
# 4. Visualization (Bokeh)
# -----------------------------------------------------------------------------

def prepare_features(df, cols_to_use):
    """
    Helper to subset, dropna, and scale features for a specific analysis.
    Returns: df_clean, features_scaled
    """
    # 1. Subset
    df_sub = df.dropna(subset=cols_to_use).reset_index(drop=True)
    
    if df_sub.empty:
        return None, None
        
    # 2. Extract
    features = df_sub[cols_to_use].values
    
    # 3. Variance Filter
    selector = VarianceThreshold(threshold=0.0)
    try:
        features = selector.fit_transform(features)
    except ValueError:
        return None, None # All zero variance

    # 4. Scale
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    return df_sub, features_scaled

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
               title=f"UMAP Projection ({len(df)} objects)",
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
# 5. Main App
# -----------------------------------------------------------------------------

def umap_dbscan_tab(df_full, numeric_cols):
    st.markdown("### UMAP & DBSCAN")
    
    # Default columns logic for UMAP tab
    # We try to use COLUMNS_TO_KEEP that are actually in the df
    default_cols = [c for c in COLUMNS_TO_KEEP if c in df_full.columns and c in numeric_cols]
    
    # If this default processing hasn't been done or cached, do it
    if "umap_feats" not in st.session_state or st.session_state.umap_feats is None:
         df_clean, feats = prepare_features(df_full, default_cols)
         if df_clean is None:
             st.error("Preprocessing for UMAP failed (0 rows after dropping NaN in default columns).")
             return
         st.session_state.umap_df = df_clean
         st.session_state.umap_feats = feats
    
    df_clean = st.session_state.umap_df
    feats = st.session_state.umap_feats
    
    c1, c2, c3 = st.columns(3)
    
    # PCA
    with c1:
        st.markdown("**1. PCA**")
        n_comp = st.number_input("Comp", 2, 50, 10, key="num_pca")
        if st.button("Run PCA", key="btn_pca"):
            st.session_state.pca_res = PCA(n_components=n_comp).fit_transform(feats)
            st.session_state.umap_res = None 
            st.rerun()

    # DBSCAN
    with c2:
        st.markdown("**2. DBSCAN**")
        eps = st.number_input("Eps", 0.1, 10.0, 1.0, 0.1, key="num_eps")
        if st.button("Run DBSCAN", key="btn_dbscan"):
            if st.session_state.pca_res is not None:
                lbls = DBSCAN(eps=eps, min_samples=10).fit_predict(st.session_state.pca_res)
                st.session_state.umap_df["DBSCAN"] = [str(l) if l!=-1 else "Noise" for l in lbls]
                st.session_state.dbscan_done = True
                st.rerun()
            else:
                st.warning("Run PCA first")

    # UMAP
    with c3:
        st.markdown("**3. UMAP**")
        n_neigh = st.number_input("Neighbors", 5, 200, 15, key="num_umap")
        if st.button("Run UMAP", key="btn_umap"):
            if st.session_state.pca_res is not None:
                red = umap.UMAP(n_neighbors=n_neigh, min_dist=0.1, n_components=2)
                st.session_state.umap_res = red.fit_transform(st.session_state.pca_res)
                st.rerun()
            else:
                st.warning("Run PCA first")

    # Plot
    if st.session_state.umap_res is not None:
        st.divider()
        coords = st.session_state.umap_res
        st.session_state.umap_df["UMAP1"] = coords[:, 0]
        st.session_state.umap_df["UMAP2"] = coords[:, 1]
        
        # Color Options
        opts = ["experiment"]
        if st.session_state.get("dbscan_done"): opts.insert(0, "DBSCAN")
        
        # Use columns from clean df
        viz_cols = [c for c in df_clean.columns if pd.api.types.is_numeric_dtype(df_clean[c])]
        exclude_viz = ["UMAP1", "UMAP2", "crop_path", "crop_zip"]
        opts += [c for c in viz_cols if c not in exclude_viz]
        
        c_mode = st.selectbox("Color by:", opts, key="sel_color")
        st.bokeh_chart(bokeh_plot(st.session_state.umap_df, "UMAP1", "UMAP2", c_mode), use_container_width=True)


def pca_exploration_tab(df_full, numeric_col_names):
    st.markdown("### PCA Exploration")
    
    # Feature Selection
    sel_features = st.multiselect(
        "Select Features for PCA:", 
        options=numeric_col_names, 
        default=numeric_col_names[:10], # Default to first 10 to be safe? Or just all? User wanted default.
        key="pca_feat_sel"
    )
    
    if st.button("Run Analysis", key="btn_run_pca_exp"):
        if not sel_features:
            st.error("Please select at least one feature.")
            return

        # Prepare data LOCALLY for this tab
        df_pca, feats_sub = prepare_features(df_full, sel_features)
        
        if df_pca is None or len(df_pca) == 0:
            st.error("Dropped all rows after selecting these features (too many missing values?).")
            return
            
        # Full PCA
        pca = PCA() # Full components
        pca.fit(feats_sub)
        
        st.session_state.pca_exp_model = pca
        st.session_state.pca_exp_feats = feats_sub
        st.session_state.pca_exp_feat_names = sel_features
        st.session_state.pca_exp_df = df_pca # Need this for plotting (matching indices)
    
    if "pca_exp_model" in st.session_state:
        pca = st.session_state.pca_exp_model
        feats_scaled = st.session_state.pca_exp_feats
        current_feat_names = st.session_state.get("pca_exp_feat_names", numeric_col_names)
        df_viz = st.session_state.get("pca_exp_df", df_full) # Fallback? No, should be there
        
        # 1. Variance Explained
        st.subheader("1. Variance Explained")
        c1, c2 = st.columns(2)
        
        var_expl = pca.explained_variance_ratio_ * 100
        cum_var = np.cumsum(var_expl)
        
        n_pcs = len(var_expl)
        pcs = range(1, n_pcs + 1)
        
        # Elbow Plot
        with c1:
            fig, ax = plt.subplots()
            ax.bar(pcs[:30], var_expl[:30], color='steelblue', alpha=0.7)
            ax.plot(pcs[:30], var_expl[:30], color='red', marker='o')
            ax.set_title("Scree Plot (First 30 PCs)")
            ax.set_xlabel("PC")
            ax.set_ylabel("Variance Explained (%)")
            st.pyplot(fig)
            
        # Cumulative
        with c2:
            fig, ax = plt.subplots()
            ax.plot(pcs[:30], cum_var[:30], marker='o', color='orange')
            ax.axhline(y=90, color='r', linestyle='--')
            ax.set_title("Cumulative Variance (First 30 PCs)")
            ax.set_xlabel("PC")
            ax.set_ylabel("Cumulative Variance (%)")
            st.pyplot(fig)
            
        st.write(f"PCs explaining >90% variance: {np.argmax(cum_var >= 90) + 1}")

        # 2. 2D Scatter
        st.subheader("2. PCA Data Projection")
        
        pcs_coords = pca.transform(feats_scaled)
        
        c_x, c_y, c_col = st.columns(3)
        x_pc = c_x.number_input("X Axis (PC)", 1, n_pcs, 1, key="pc_x")
        y_pc = c_y.number_input("Y Axis (PC)", 1, n_pcs, 2, key="pc_y")
        
        # Options for color - use columns from df_viz
        opts = ["experiment"] + [c for c in df_viz.columns if pd.api.types.is_numeric_dtype(df_viz[c])]
        color_by_pc = c_col.selectbox("Color By", opts, key="pc_color_by")
        
        # Create temp DF for plotting
        plot_df = pd.DataFrame({
            "x": pcs_coords[:, x_pc-1],
            "y": pcs_coords[:, y_pc-1],
            "color": df_viz[color_by_pc] if color_by_pc in df_viz.columns else np.zeros(len(df_viz))
        })
        
        fig, ax = plt.subplots(figsize=(8,6))
        if pd.api.types.is_numeric_dtype(plot_df["color"]):
            sc = ax.scatter(plot_df.x, plot_df.y, c=plot_df.color, cmap='viridis', alpha=0.6, s=10)
            plt.colorbar(sc, label=color_by_pc)
        else:
            sns.scatterplot(data=plot_df, x="x", y="y", hue="color", ax=ax, alpha=0.6, s=10)
            
        ax.set_xlabel(f"PC{x_pc} ({var_expl[x_pc-1]:.1f}%)")
        ax.set_ylabel(f"PC{y_pc} ({var_expl[y_pc-1]:.1f}%)")
        ax.set_title(f"PCA: PC{x_pc} vs PC{y_pc}")
        st.pyplot(fig)

        # 3. Loadings
        st.subheader("3. Feature Loadings (Top contributors)")
        
        try:
            loadings = pd.DataFrame(pca.components_.T, columns=[f"PC{i+1}" for i in range(n_pcs)], index=current_feat_names)
            
            target_pc = st.selectbox("Select PC to inspect loadings", [f"PC{i+1}" for i in range(min(10, n_pcs))], key="sel_loadings_pc")
            
            n_top = 10
            pc_loadings = loadings[target_pc].abs().sort_values(ascending=False).head(n_top)
            top_feats = pc_loadings.index
            
            # Get actual signed values for plotting
            plot_loadings = loadings.loc[top_feats, target_pc]
            
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ['blue' if x > 0 else 'red' for x in plot_loadings]
            plot_loadings.plot(kind='barh', ax=ax, color=colors)
            ax.set_title(f"Top {n_top} Loadings for {target_pc}")
            ax.set_xlabel("Loading Value")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error plotting loadings: {e}")
            import traceback
            st.text(traceback.format_exc())
            
        # 4. KMeans / Elbow for Clusters
        st.subheader("4. Optimal Cluster Number (K-Means)")
        
        if st.checkbox("Run K-Means Exploration (k=2..15)", key="run_kmeans"):
            # Subsample if large
            X_clust = pcs_coords[:, :5] # Use first 5 PCs as in R script
            if len(X_clust) > 5000:
                idx = np.random.choice(len(X_clust), 5000, replace=False)
                X_clust = X_clust[idx]
                st.info("Subsampled to 5000 points for speed.")
                
            wss = []
            sils = []
            k_range = range(2, 16)
            
            with st.spinner("Calculating K-Means..."):
                for k in k_range:
                    km = KMeans(n_clusters=k, n_init=10, random_state=42)
                    labels = km.fit_predict(X_clust)
                    wss.append(km.inertia_)
                    sils.append(silhouette_score(X_clust, labels))
            
            c_k1, c_k2 = st.columns(2)
            
            with c_k1:
                fig, ax = plt.subplots()
                ax.plot(k_range, wss, 'bo-')
                ax.set_xlabel("Number of clusters k")
                ax.set_ylabel("WSS")
                ax.set_title("Elbow Method (WSS)")
                st.pyplot(fig)
                
            with c_k2:
                fig, ax = plt.subplots()
                ax.plot(k_range, sils, 'go-')
                ax.set_xlabel("Number of clusters k")
                ax.set_ylabel("Silhouette Score")
                ax.set_title("Silhouette Method")
                st.pyplot(fig)

def main():
    st.sidebar.title("Protist Clustering")
    
    # Data Loading Section
    st.sidebar.markdown("### Data Source")
    data_source = st.sidebar.radio("Source:", ["Folder", "Upload"], horizontal=True)
    
    df_raw = pd.DataFrame()
    
    if data_source == "Folder":
        default_path = "/Volumes/vangestel/Jonas/Vanessa_segmentation/Visualizations/Umap_data"
        data_folder = st.sidebar.text_input("Data Folder", value=default_path)
        st.session_state.data_folder = data_folder # Store for image server
        
        if st.sidebar.button("Load Folder", key="btn_load_folder"):
            st.session_state.df_raw = load_data_from_folder(data_folder)
            st.session_state.data_source_mode = "Folder"
            st.session_state.pca_res = None
            st.session_state.umap_res = None
            st.session_state.sel_hash = None
            st.session_state.last_loaded_folder = data_folder
            # Clear ext PCA model if data changes
            if "pca_exp_model" in st.session_state: del st.session_state.pca_exp_model
            # Clear UMAP cache
            if "umap_feats" in st.session_state: del st.session_state.umap_feats
            st.rerun() 
            
    else: # Upload
        uploaded_files = st.sidebar.file_uploader("Upload CSVs", type="csv", accept_multiple_files=True)
        st.session_state.data_folder = "" 
        
        if uploaded_files:
            # Only load if button clicked
            if st.sidebar.button("Load Files", key="btn_load_files"):
                st.session_state.df_raw = load_data_from_uploads(uploaded_files)
                st.session_state.data_source_mode = "Upload"
                st.session_state.pca_res = None
                st.session_state.umap_res = None
                st.session_state.sel_hash = None
                # Clear ext PCA model if data changes
                if "pca_exp_model" in st.session_state: del st.session_state.pca_exp_model
                 # Clear UMAP cache
                if "umap_feats" in st.session_state: del st.session_state.umap_feats
                st.rerun() 

    if "df_raw" not in st.session_state or st.session_state.df_raw.empty:
        st.info("Please load data to begin.")
        return

    # Filter
    all_exps = sorted(st.session_state.df_raw["experiment"].astype(str).unique())
    sel_exps = st.sidebar.multiselect("Experiments", all_exps, default=all_exps)
    
    if not sel_exps: return

    # Processing - Now strictly mostly light-weight
    curr_hash = hash(tuple(sorted(sel_exps)))
    if st.session_state.get("sel_hash") != curr_hash: 
        with st.spinner("Preprocessing..."):
            subset = st.session_state.df_raw[st.session_state.df_raw["experiment"].isin(sel_exps)]
            df_full_processed, numeric_cols = preprocess_data(subset)
            
            st.session_state.df_full_processed = df_full_processed
            st.session_state.numeric_cols = numeric_cols
            st.session_state.sel_hash = curr_hash
            
            # Reset Tab 1 (UMAP) cache since data changed
            st.session_state.umap_feats = None
            
            # Clear ext PCA model if data changes
            if "pca_exp_model" in st.session_state: del st.session_state.pca_exp_model

    df_full = st.session_state.get("df_full_processed")
    numeric_cols = st.session_state.get("numeric_cols")
    
    if df_full is None: return 

    st.sidebar.success(f"Ready: {len(df_full)} objects (Total) | {len(numeric_cols)} numeric cols")

    # TABS
    tab1, tab2 = st.tabs(["UMAP & DBSCAN", "PCA Exploration"])
    
    with tab1:
        umap_dbscan_tab(df_full, numeric_cols)
        
    with tab2:
        pca_exploration_tab(df_full, numeric_cols)

if __name__ == "__main__":
    main()
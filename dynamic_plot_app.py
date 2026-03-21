import streamlit as st
import pandas as pd
import plotly.express as px
import os
import glob
import numpy as np

# -----------------------------------------------------------------------------
# 1. Page Config
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Dynamic CSV Plotter", layout="wide")
st.title("Dynamic CSV Plotter")
st.markdown("Import a CSV (up to 5GB), filter data dynamically, and plot with custom mathematical expressions for axes.")

# -----------------------------------------------------------------------------
# 2. Data Loading Module
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading CSV... This may take a while for large files.")
def load_data(file_or_path):
    try:
        # Use pyarrow engine which is much faster and memory efficient for large CSVs
        df = pd.read_csv(file_or_path, engine="pyarrow")
    except Exception:
        # Fallback to standard c engine if pyarrow fails or is missing
        if hasattr(file_or_path, 'seek'):
            file_or_path.seek(0)
        df = pd.read_csv(file_or_path, low_memory=False)
    return df

st.sidebar.header("1. Data Loading")
data_source = st.sidebar.radio("Data Source:", ["Upload File", "Local Folder"])

df = None
if data_source == "Upload File":
    st.sidebar.markdown("*Note: Ensure `.streamlit/config.toml` has `maxUploadSize` increased for files >200MB.*")
    uploaded_file = st.sidebar.file_uploader("Upload CSV (up to 5GB)", type=['csv'])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
else:
    # Optional folder loading since large files often time out web uploads
    default_path = "/Users/jonas/Documents/Master_thesis/streamlit/"
    folder_path = st.sidebar.text_input("Local Folder Path containing CSVs:", value=default_path)
    if os.path.isdir(folder_path):
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        if not csv_files:
            st.sidebar.warning("No CSV files found in directory.")
        else:
            selected_file = st.sidebar.selectbox("Select a CSV file:", [os.path.basename(f) for f in csv_files])
            if st.sidebar.button("Load Local File"):
                full_path = os.path.join(folder_path, selected_file)
                df = load_data(full_path)
    elif folder_path:
        st.sidebar.error("Invalid directory path.")

if st.sidebar.button("Clear Cache"):
    st.cache_data.clear()
    st.rerun()

# -----------------------------------------------------------------------------
# 3. Filtering and Plotting
# -----------------------------------------------------------------------------
if df is not None:
    st.success(f"Data loaded successfully! Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
    st.write("### Data Preview")
    st.dataframe(df.head(10))

    st.divider()

    # --- Filtering Panel ---
    st.subheader("Filter Panel")
    st.markdown("Select columns to filter the dataset before plotting.")
    
    # 1. Let user choose which columns to filter on to avoid creating 100 filters
    filter_cols = st.multiselect("Select columns to add filters for:", options=df.columns)
    
    filtered_df = df.copy()

    if filter_cols:
        # Create columns dynamically for the selected filters
        cols = st.columns(min(len(filter_cols), 4))
        
        for i, col in enumerate(filter_cols):
            with cols[i % 4]:
                with st.container(border=True):
                    st.markdown(f"**{col}**")
                    
                    mode = st.radio("Filter Mode", ["Include", "Exclude"], key=f"mode_{col}", horizontal=True, label_visibility="collapsed")
                    is_include = (mode == "Include")
                    
                    is_numeric_orig = pd.api.types.is_numeric_dtype(df[col])
                    force_cat = False
                    if is_numeric_orig:
                        force_cat = st.toggle("Treat as Categorical", key=f"fcat_{col}")
                        
                    if is_numeric_orig and not force_cat:
                        min_val = float(df[col].min())
                        max_val = float(df[col].max())
                        
                        if pd.isna(min_val) or pd.isna(max_val):
                            st.warning("Contains NaNs.")
                        elif min_val == max_val:
                            st.info(f"Constant: {min_val}")
                        else:
                            step_size = (max_val - min_val) / 100.0 if (max_val - min_val) > 0 else 0.1
                            selected_range = st.slider(
                                "Range", 
                                min_value=min_val, 
                                max_value=max_val, 
                                value=(min_val, max_val),
                                step=step_size,
                                key=f"slider_{col}"
                            )
                            mask = (filtered_df[col] >= selected_range[0]) & (filtered_df[col] <= selected_range[1])
                            if not is_include:
                                mask = ~mask
                            filtered_df = filtered_df[mask]
                    else:
                        if force_cat:
                            unique_vals = df[col].dropna().astype(str).unique().tolist()
                        else:
                            unique_vals = df[col].dropna().unique().tolist()
                            
                        if len(unique_vals) > 500:
                            search_term = st.text_input("Search (substring)", key=f"search_{col}")
                            if search_term:
                                mask = filtered_df[col].astype(str).str.contains(search_term, case=False, na=False)
                                if not is_include:
                                    mask = ~mask
                                filtered_df = filtered_df[mask]
                        else:
                            # Intuitive defaults: if Include, default is everything. If Exclude, default is nothing.
                            def_vals = unique_vals if is_include else []
                            selected_vals = st.multiselect(
                                "Select Values", 
                                options=unique_vals, 
                                default=def_vals,
                                key=f"multi_{col}"
                            )
                            if force_cat:
                                mask = filtered_df[col].astype(str).isin(selected_vals)
                            else:
                                mask = filtered_df[col].isin(selected_vals)
                            
                            if not is_include:
                                mask = ~mask
                            filtered_df = filtered_df[mask]
                        
        st.write(f"Rows after filtering: **{filtered_df.shape[0]:,}** ({(filtered_df.shape[0]/df.shape[0])*100:.2f}%)")
    else:
        st.info("Select columns above to display filters.")

    st.divider()

    # --- Plotting Panel ---
    st.subheader("Dynamic Plotting Variables")
    st.markdown("You can specify mathematical expressions for your axes (e.g., `column_A + column_B * 2`, `1 - log(column_A)`). "
                "The variables must match your column names exactly.")
    
    # Building Blocks UI
    st.markdown("**Expression Building Blocks**")
    bb_col1, bb_col2, bb_col3 = st.columns([2, 1, 1])
    with bb_col1:
        var_to_insert = st.selectbox("Select variable to insert:", options=df.columns)
    with bb_col2:
        if st.button("Append to X-Axis"):
            st.session_state.x_expr = st.session_state.get("x_expr", "") + f" {var_to_insert}"
            st.rerun()
    with bb_col3:
        if st.button("Append to Y-Axis"):
            st.session_state.y_expr = st.session_state.get("y_expr", "") + f" {var_to_insert}"
            st.rerun()

    col_x, col_y, col_color = st.columns(3)
    
    with col_x:
        default_x = df.columns[0] if len(df.columns) > 0 else ""
        x_expr = st.text_input("X-Axis Expression", key="x_expr", value=st.session_state.get("x_expr", default_x))
        if "x_expr" not in st.session_state and default_x: st.session_state.x_expr = default_x
        
    with col_y:
        default_y = df.columns[1] if len(df.columns) > 1 else ""
        y_expr = st.text_input("Y-Axis Expression", key="y_expr", value=st.session_state.get("y_expr", default_y))
        if "y_expr" not in st.session_state and default_y: st.session_state.y_expr = default_y
        
    with col_color:
        color_col = st.selectbox("Color By (optional):", options=["None"] + list(df.columns))
        
        force_categorical = False
        continuous_color_scale = "Viridis"
        if color_col != "None" and pd.api.types.is_numeric_dtype(filtered_df[color_col]):
            force_categorical = st.toggle("Treat as Categorical", help="Forces numerical groups to be plotted as discrete categories (like as.factor())")
            if not force_categorical:
                continuous_color_scale = st.selectbox("Continuous Palette:", 
                    options=["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Turbo", "Rainbow", "Jet", "Blues", "Reds"])

    generate_pressed = st.button("Generate Plot", type="primary")
    update_colors_pressed = False

    # Color Customization Form (Leaves old plot active until "Update Colors" is clicked)
    color_discrete_map = {}
    is_discrete = False
    
    if color_col != "None":
        if force_categorical:
            color_series = filtered_df[color_col].dropna().astype(str)
            is_discrete = True
        else:
            color_series = filtered_df[color_col].dropna()
            is_discrete = not pd.api.types.is_numeric_dtype(color_series)

        unique_vals = list(color_series.unique())
        
        if is_discrete:
            if len(unique_vals) <= 20:
                with st.expander("Customize Group Colors", expanded=True):
                    with st.form("color_form", border=False):
                        st.write("Pick colors for each group. The plot will only redraw when you click 'Update Colors and Plot'.")
                        color_config_cols = st.columns(min(len(unique_vals), 5))
                        default_colors = px.colors.qualitative.Plotly * 10
                        for i, val in enumerate(unique_vals):
                            with color_config_cols[i % 5]:
                                color_discrete_map[str(val)] = st.color_picker(str(val), value=default_colors[i], key=f"cpicker_{val}")
                        update_colors_pressed = st.form_submit_button("Update Colors and Plot")
            else:
                st.info(f"Too many unique values ({len(unique_vals)}) to pick custom colors individually.")
        else:
            st.info(f"Coloring by continuous metric '{color_col}'. Using palette: {continuous_color_scale}")

    if generate_pressed or update_colors_pressed:
        if not x_expr.strip() or not y_expr.strip():
            st.error("Please provide both X and Y axis expressions.")
        elif filtered_df.empty:
            st.warning("Filtered dataset is empty. Adjust your filters.")
        else:
            try:
                plot_data = filtered_df.copy()
                
                # Evaluate expressions safely using pandas eval
                x_series = plot_data.eval(x_expr)
                y_series = plot_data.eval(y_expr)
                
                plot_data["_x_eval"] = x_series
                plot_data["_y_eval"] = y_series
                
                # Handle large datasets
                MAX_POINTS = 1_000_000
                orig_points = len(plot_data)
                if orig_points > MAX_POINTS:
                    st.warning(f"Plotting {orig_points:,} points might crush your browser. "
                               f"Downsampling to {MAX_POINTS:,} random points for visualization.")
                    plot_data = plot_data.sample(n=MAX_POINTS, random_state=42)
                
                fig_args = {
                    "data_frame": plot_data,
                    "x": "_x_eval",
                    "y": "_y_eval",
                    "labels": {"_x_eval": x_expr, "_y_eval": y_expr}
                }
                
                if color_col != "None":
                    if force_categorical:
                        plot_data[color_col] = plot_data[color_col].astype(str)
                    
                    fig_args["color"] = color_col
                    
                    if is_discrete:
                        if len(color_discrete_map) > 0:
                            fig_args["color_discrete_map"] = color_discrete_map
                    else:
                        fig_args["color_continuous_scale"] = continuous_color_scale
                    
                # Use WebGL for better performance on large scatter plots
                if len(plot_data) > 10000:
                    fig_args["render_mode"] = "webgl"
                    
                fig = px.scatter(**fig_args)
                
                fig.update_layout(
                    title=f"Scatter Plot: {y_expr} vs {x_expr}",
                    hovermode="closest",
                    height=700
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error("Error evaluating expression or generating plot.")
                st.exception(e)
                st.info("Make sure you are using valid pandas evaluation syntax and that referenced column names are spelled correctly.")
else:
    st.info("Please load a CSV dataset from the sidebar to get started.")

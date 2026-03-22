import reflex as rx
from reflex.components.plotly.plotly import Point
from pydantic import BaseModel
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import glob
import io


class FilterDef(BaseModel):
    """Pydantic model for a single dynamic filter definition."""
    col_name: str = ""
    is_numeric: bool = False
    is_include: bool = True
    force_cat: bool = False

    min_v: float = 0.0
    max_v: float = 1.0
    range_v: list[float] = [0.0, 1.0]

    unique: list[str] = []
    selected: list[str] = []


PALETTE_OPTIONS: list[str] = [
    "Viridis", "Plasma", "Inferno", "Magma",
    "Cividis", "Turbo", "Rainbow", "Blues", "Reds",
]


class PlotState(rx.State):
    """The advanced app state."""

    # Backend-only DataFrames (never synced to React)
    _df: pd.DataFrame = None
    _filtered_df: pd.DataFrame = None
    _pending_colors: dict = {}
    _plot_index_map: list = []  # maps plotted point index → original DF row index

    # Synced state
    columns: list[str] = []
    preview_rows: list[dict] = []

    folder_path: str = "/Users/jonas/Documents/Master_thesis/streamlit/"
    available_files: list[str] = []
    selected_file: str = ""
    status_msg: str = "Load a CSV to begin."

    x_expr: str = ""
    y_expr: str = ""
    color_col: str = "None"
    force_cat_color: bool = False
    cont_palette: str = "Viridis"
    color_map: dict[str, str] = {}
    color_groups: list[str] = []
    point_size: list[float] = [5.0]
    point_alpha: list[float] = [0.7]

    plot_fig: go.Figure = go.Figure()

    # Selection state
    selected_rows: list[dict] = []
    selected_count: int = 0
    selected_columns: list[str] = []

    filters: list[FilterDef] = []
    filter_col_dropdown: str = ""
    builder_col_dropdown: str = ""

    row_count: int = 0
    col_count: int = 0
    filtered_row_count: int = 0

    show_filters: bool = True
    show_preview: bool = False
    show_color_pickers: bool = False
    sidebar_open: bool = False

    # ── Computed Vars ──────────────────────────────────────────────

    @rx.var
    def filter_cols_available(self) -> list[str]:
        used = {f.col_name for f in self.filters}
        return [c for c in self.columns if c not in used]

    @rx.var
    def file_options(self) -> list[str]:
        return ["Select a file…"] + self.available_files

    @rx.var
    def color_options(self) -> list[str]:
        return ["None"] + self.columns

    @rx.var
    def data_loaded(self) -> bool:
        return len(self.columns) > 0

    @rx.var
    def filter_pct(self) -> str:
        if self.row_count == 0:
            return "—"
        return f"{self.filtered_row_count / self.row_count * 100:.1f}%"

    @rx.var
    def has_filters(self) -> bool:
        return len(self.filters) > 0

    @rx.var
    def has_selection(self) -> bool:
        return self.selected_count > 0

    @rx.var
    def is_discrete_color(self) -> bool:
        return len(self.color_groups) > 0

    @rx.var
    def has_color_col(self) -> bool:
        return self.color_col != "None"

    # ── Explicit Setters ──────────────────────────────────────────

    def set_folder_path(self, v: str): self.folder_path = v
    def set_selected_file(self, v: str): self.selected_file = v
    def set_x_expr(self, v: str): self.x_expr = v
    def set_y_expr(self, v: str): self.y_expr = v
    def set_force_cat_color(self, v: bool):
        self.force_cat_color = v
        self._update_color_groups()
    def set_cont_palette(self, v: str): self.cont_palette = v
    def set_filter_col_dropdown(self, v: str): self.filter_col_dropdown = v
    def set_builder_col_dropdown(self, v: str): self.builder_col_dropdown = v
    def set_point_size(self, v: list[float]): self.point_size = v
    def set_point_alpha(self, v: list[float]): self.point_alpha = v

    def set_color_col(self, v: str):
        self.color_col = v
        self._update_color_groups()

    def _update_color_groups(self):
        """Rebuild the list of unique color groups and default color map."""
        if self._df is None or self.color_col == "None" or self.color_col not in self._df.columns:
            self.color_groups = []
            self.color_map = {}
            self._pending_colors = {}
            return
        is_disc = self.force_cat_color or not pd.api.types.is_numeric_dtype(self._df[self.color_col])
        if is_disc:
            groups = sorted(self._df[self.color_col].dropna().astype(str).unique().tolist()[:50])
            default_colors = px.colors.qualitative.Plotly
            new_map = {}
            for i, g in enumerate(groups):
                new_map[g] = self.color_map.get(g, default_colors[i % len(default_colors)])
            self.color_groups = groups
            self.color_map = new_map
            self._pending_colors = dict(new_map)
        else:
            self.color_groups = []
            self.color_map = {}
            self._pending_colors = {}

    def update_group_color(self, group: str, color: str):
        """Store color change in backend-only dict (no frontend sync = no lag)."""
        self._pending_colors[group] = color

    def apply_colors(self):
        """Copy pending colors into the synced color_map and rebuild the plot."""
        self.color_map = dict(self._pending_colors)
        self._rebuild_plot()

    def toggle_filters(self, v: bool):
        self.show_filters = v

    def toggle_preview(self, v: bool):
        self.show_preview = v

    def toggle_color_pickers(self, v: bool):
        self.show_color_pickers = v

    def open_sidebar(self):
        self.sidebar_open = True

    def close_sidebar(self):
        self.sidebar_open = False

    # ── Expression Builder ────────────────────────────────────────

    def append_col_to_x(self):
        col = self.builder_col_dropdown
        if col:
            self.x_expr = (self.x_expr + " " + col).strip()

    def append_col_to_y(self):
        col = self.builder_col_dropdown
        if col:
            self.y_expr = (self.y_expr + " " + col).strip()

    def clear_x(self):
        self.x_expr = ""

    def clear_y(self):
        self.y_expr = ""

    # ── Filter Mutations ──────────────────────────────────────────

    def add_filter(self):
        col = self.filter_col_dropdown
        if not col or self._df is None:
            return
        is_num = pd.api.types.is_numeric_dtype(self._df[col])

        if is_num:
            mn = float(self._df[col].min())
            mx = float(self._df[col].max())
            uniq: list[str] = []
            sel: list[str] = []
        else:
            mn, mx = 0.0, 1.0
            uniq = sorted(self._df[col].dropna().astype(str).unique().tolist()[:300])
            sel = uniq.copy()

        self.filters.append(
            FilterDef(
                col_name=col, is_numeric=is_num, is_include=True, force_cat=False,
                min_v=mn, max_v=mx, range_v=[mn, mx], unique=uniq, selected=sel,
            )
        )
        self.filter_col_dropdown = ""
        self._apply_filters()

    def remove_filter(self, col: str):
        self.filters = [f for f in self.filters if f.col_name != col]
        self._apply_filters()

    def clear_all_filters(self):
        self.filters = []
        self._apply_filters()

    def update_filter_range(self, val: list[float], col: str):
        for f in self.filters:
            if f.col_name == col:
                f.range_v = val
        self._apply_filters()

    def toggle_filter_val(self, checked: bool, col: str, val: str):
        for f in self.filters:
            if f.col_name == col:
                if checked and val not in f.selected:
                    f.selected.append(val)
                elif not checked and val in f.selected:
                    f.selected.remove(val)
        self._apply_filters()

    def select_all_filter(self, col: str):
        for f in self.filters:
            if f.col_name == col:
                f.selected = f.unique.copy()
        self._apply_filters()

    def deselect_all_filter(self, col: str):
        for f in self.filters:
            if f.col_name == col:
                f.selected = []
        self._apply_filters()

    def toggle_filter_include(self, mode: str, col: str):
        for f in self.filters:
            if f.col_name == col:
                f.is_include = mode == "Include"
                f.selected = f.unique.copy() if f.is_include else []
        self._apply_filters()

    def toggle_filter_force_cat(self, force: bool, col: str):
        for f in self.filters:
            if f.col_name == col:
                f.force_cat = force
                if force:
                    f.unique = sorted(self._df[col].dropna().astype(str).unique().tolist()[:300])
                    f.selected = f.unique.copy() if f.is_include else []
        self._apply_filters()

    # ── Core Logic ────────────────────────────────────────────────

    def _apply_filters(self):
        if self._df is None:
            return
        df = self._df.copy()
        for f in self.filters:
            if f.is_numeric and not f.force_cat:
                mask = (df[f.col_name] >= f.range_v[0]) & (df[f.col_name] <= f.range_v[1])
                if not f.is_include:
                    mask = ~mask
                df = df[mask]
            else:
                mask = df[f.col_name].astype(str).isin(f.selected)
                if not f.is_include:
                    mask = ~mask
                df = df[mask]
        self._filtered_df = df
        self.filtered_row_count = len(df)
        self._rebuild_plot()

    async def handle_upload(self, files: list[rx.UploadFile]):
        if not files:
            return
        self.status_msg = "Uploading and parsing…"
        yield
        f = files[0]
        data = await f.read()
        try:
            self._df = pd.read_csv(io.BytesIO(data), engine="pyarrow")
        except Exception:
            self._df = pd.read_csv(io.BytesIO(data), low_memory=False)
        self._setup_post_load()
        yield

    def _setup_post_load(self):
        self._filtered_df = self._df.copy()
        self.columns = [str(c) for c in self._df.columns]
        self.row_count = len(self._df)
        self.col_count = len(self.columns)
        self.filtered_row_count = self.row_count
        self.filters = []
        if self.columns:
            self.x_expr = self.columns[0]
        if len(self.columns) > 1:
            self.y_expr = self.columns[1]
        # Build preview (first 5 rows as dicts)
        self.preview_rows = self._df.head(5).fillna("").astype(str).to_dict(orient="records")
        self.status_msg = f"Loaded {self.row_count:,} rows × {self.col_count} cols."
        self._rebuild_plot()

    def find_csvs(self):
        if os.path.isdir(self.folder_path):
            files = glob.glob(os.path.join(self.folder_path, "*.csv"))
            self.available_files = sorted(os.path.basename(f) for f in files)
            self.status_msg = (
                f"Found {len(self.available_files)} CSVs."
                if self.available_files
                else "No CSV files found."
            )
        else:
            self.status_msg = "Invalid directory path."
            self.available_files = []

    def load_file(self):
        if not self.selected_file or self.selected_file.startswith("Select"):
            self.status_msg = "Select a file first."
            return
        full_path = os.path.join(self.folder_path, self.selected_file)
        self.status_msg = f"Loading {self.selected_file}…"
        yield
        try:
            self._df = pd.read_csv(full_path, engine="pyarrow")
        except Exception:
            self._df = pd.read_csv(full_path, low_memory=False)
        self._setup_post_load()

    def generate_plot(self):
        """Public event handler triggered by the Generate Plot button."""
        self._rebuild_plot()

    def _rebuild_plot(self):
        """Internal non-generator plot builder."""
        if self._filtered_df is None or self._filtered_df.empty:
            self.status_msg = "Dataset empty after filtering."
            return
        if not self.x_expr.strip() or not self.y_expr.strip():
            return

        try:
            plot_data = self._filtered_df.copy()
            plot_data["_x_eval"] = plot_data.eval(self.x_expr)
            plot_data["_y_eval"] = plot_data.eval(self.y_expr)

            MAX_PTS = 1_000_000
            n = len(plot_data)
            if n > MAX_PTS:
                plot_data = plot_data.sample(n=MAX_PTS, random_state=42)
                self.status_msg = f"Rendered {MAX_PTS:,} / {n:,} pts (downsampled)."
            else:
                self.status_msg = f"Rendered {n:,} points."

            kw: dict = dict(
                data_frame=plot_data, x="_x_eval", y="_y_eval",
                labels={"_x_eval": self.x_expr, "_y_eval": self.y_expr},
            )

            if self.color_col != "None" and self.color_col in plot_data.columns:
                if self.force_cat_color:
                    plot_data[self.color_col] = plot_data[self.color_col].astype(str)
                kw["color"] = self.color_col
                is_disc = self.force_cat_color or not pd.api.types.is_numeric_dtype(plot_data[self.color_col])
                if is_disc and self.color_map:
                    kw["color_discrete_map"] = self.color_map
                elif not is_disc:
                    kw["color_continuous_scale"] = self.cont_palette

            if len(plot_data) > 10_000:
                kw["render_mode"] = "webgl"

            fig = px.scatter(**kw)
            fig.update_traces(marker=dict(size=self.point_size[0], opacity=self.point_alpha[0]))

            # Store original row indices as customdata for selection lookup
            self._plot_index_map = plot_data.index.tolist()
            for trace in fig.data:
                n_pts = len(trace.x) if trace.x is not None else 0
                if n_pts > 0:
                    # Each trace gets its own subset of indices
                    trace_indices = list(range(len(self._plot_index_map)))
                    # For traces split by color, use pointIndex to map back
                    pass

            fig.update_layout(
                title=f"{self.y_expr}  vs  {self.x_expr}",
                hovermode="closest",
                height=750,
                autosize=True,
                margin=dict(l=40, r=20, t=50, b=40),
                template="plotly_dark",
                dragmode="select",
            )
            self.plot_fig = fig
            # Clear previous selection when plot is regenerated
            self.selected_rows = []
            self.selected_count = 0
            self.selected_columns = []

        except Exception as e:
            self.status_msg = f"Expression error: {e}"
            print(f"[PlotState] {e}")

    def handle_plot_selected(self, points: list[Point]):
        """Handle lasso/box selection on the Plotly chart."""
        if not points or self._filtered_df is None:
            return

        # Collect original DF row indices from all selected points
        row_indices = set()
        for pt in points:
            pi = pt.get("pointIndex", pt.get("pointNumber"))
            if pi is not None and pi < len(self._plot_index_map):
                row_indices.add(self._plot_index_map[pi])

        if not row_indices:
            return

        # Look up rows and convert to list of dicts (max 200 for performance)
        valid = [i for i in row_indices if i in self._filtered_df.index]
        sel_df = self._filtered_df.loc[valid[:200]]
        self.selected_columns = [str(c) for c in sel_df.columns]
        self.selected_rows = sel_df.fillna("").astype(str).to_dict(orient="records")
        self.selected_count = len(row_indices)
        self.status_msg = f"Selected {len(row_indices)} points (showing {len(self.selected_rows)})."

    def clear_selection(self):
        self.selected_rows = []
        self.selected_count = 0
        self.selected_columns = []
        self.status_msg = "Selection cleared."


# ═══════════════════════════════════════════════════════════════════
#  UI COMPONENTS
# ═══════════════════════════════════════════════════════════════════

def _filter_card(f: FilterDef) -> rx.Component:
    """Single filter card rendered via rx.foreach."""
    return rx.card(
        rx.hstack(
            rx.badge(f.col_name, color_scheme="blue", size="2", variant="surface"),
            rx.spacer(),
            rx.radio(
                ["Include", "Exclude"],
                value=rx.cond(f.is_include, "Include", "Exclude"),
                on_change=lambda v: PlotState.toggle_filter_include(v, f.col_name),
                direction="row", size="1",
            ),
            rx.icon_button(
                rx.icon("x"), size="1", color_scheme="red", variant="ghost",
                on_click=lambda: PlotState.remove_filter(f.col_name),
            ),
            align="center", width="100%",
        ),
        rx.box(height="6px"),
        # "Treat as Categorical" for numeric columns
        rx.cond(
            f.is_numeric,
            rx.checkbox(
                "Treat as Categorical", checked=f.force_cat,
                on_change=lambda v: PlotState.toggle_filter_force_cat(v, f.col_name),
                size="1",
            ),
            rx.fragment(),
        ),
        rx.box(height="6px"),
        rx.cond(
            f.is_numeric & ~f.force_cat,
            # Numeric range slider
            rx.box(
                rx.slider(
                    min=f.min_v, max=f.max_v, value=f.range_v,
                    on_change=lambda v: PlotState.update_filter_range(v, f.col_name),
                ),
                width="100%",
            ),
            # Category checklist with select/deselect all
            rx.vstack(
                rx.hstack(
                    rx.button("All", size="1", variant="ghost", on_click=lambda: PlotState.select_all_filter(f.col_name)),
                    rx.button("None", size="1", variant="ghost", on_click=lambda: PlotState.deselect_all_filter(f.col_name)),
                    spacing="1",
                ),
                rx.scroll_area(
                    rx.vstack(
                        rx.foreach(
                            f.unique,
                            lambda val: rx.checkbox(
                                val, checked=f.selected.contains(val),
                                on_change=lambda c: PlotState.toggle_filter_val(c, f.col_name, val),
                                size="1",
                            ),
                        ),
                        spacing="1",
                    ),
                    type="always", scrollbars="vertical",
                    style={"height": "120px", "border": "1px solid var(--gray-5)", "padding": "6px", "borderRadius": "6px"},
                ),
                spacing="1", width="100%",
            ),
        ),
        size="1", width="100%",
    )


def _sidebar() -> rx.Component:
    """Animated collapsible sidebar — content stays mounted, only visually hidden."""

    # CSS helpers driven by state
    content_opacity = rx.cond(PlotState.sidebar_open, "1", "0")
    content_pointer = rx.cond(PlotState.sidebar_open, "auto", "none")
    icon_opacity = rx.cond(PlotState.sidebar_open, "0", "1")
    icon_pointer = rx.cond(PlotState.sidebar_open, "none", "auto")

    # Icon rail — visible when collapsed
    icon_rail = rx.vstack(
        rx.icon("database", size=20, color="var(--accent-9)"),
        rx.icon("upload", size=20, color="gray"),
        rx.icon("folder-open", size=20, color="gray"),
        rx.icon("bar-chart-3", size=20, color="gray"),
        spacing="5",
        align="center",
        padding_top="1em",
        position="absolute",
        top="0", left="0",
        width="60px",
        opacity=icon_opacity,
        pointer_events=icon_pointer,
        transition="opacity 0.2s ease",
    )

    # Full content — always mounted, faded out when collapsed
    full_content = rx.vstack(
        rx.heading("Data Hub", size="5"),
        rx.divider(),

        rx.text("Upload CSV", weight="bold", size="2"),
        rx.upload(
            rx.vstack(
                rx.icon("cloud-upload", size=24, color="var(--accent-9)"),
                rx.text("Drop file or click", size="1", color="gray"),
                align="center", spacing="1",
            ),
            id="upload_csv", multiple=False,
            accept={"text/csv": [".csv"]}, max_files=1,
            on_drop=PlotState.handle_upload(rx.upload_files(upload_id="upload_csv")),
            border="2px dashed var(--gray-7)", padding="1em",
            border_radius="10px", width="100%", cursor="pointer",
            _hover={"border_color": "var(--accent-9)"},
        ),

        rx.divider(),

        rx.text("Local Files", weight="bold", size="2"),
        rx.input(
            value=PlotState.folder_path, on_change=PlotState.set_folder_path,
            placeholder="/path/to/csvs…", size="1",
        ),
        rx.button("Scan", on_click=PlotState.find_csvs, width="100%", variant="soft", size="1"),
        rx.select(
            PlotState.file_options, value=PlotState.selected_file,
            on_change=PlotState.set_selected_file, width="100%", size="1",
        ),
        rx.button("Load", on_click=PlotState.load_file, color_scheme="blue", width="100%", size="1"),

        rx.divider(),

        rx.vstack(
            rx.hstack(
                rx.text("Total:", size="1", color="gray"),
                rx.text(PlotState.row_count.to(str), size="1", weight="bold"),
            ),
            rx.hstack(
                rx.text("Filtered:", size="1", color="gray"),
                rx.text(PlotState.filtered_row_count.to(str), size="1", weight="bold", color="var(--green-11)"),
            ),
            rx.hstack(
                rx.text("Kept:", size="1", color="gray"),
                rx.text(PlotState.filter_pct, size="1", weight="bold", color="var(--blue-11)"),
            ),
            rx.text(PlotState.status_msg, color="var(--accent-11)", size="1", weight="bold"),
            spacing="1",
        ),

        align_items="stretch",
        spacing="2",
        padding="0.8em",
        width="280px",
        opacity=content_opacity,
        pointer_events=content_pointer,
        transition="opacity 0.2s ease",
    )

    return rx.box(
        icon_rail,
        full_content,
        width=rx.cond(PlotState.sidebar_open, "280px", "60px"),
        height="100vh",
        overflow_y="auto",
        overflow_x="hidden",
        flex_shrink="0",
        border_right="1px solid var(--gray-5)",
        background="var(--color-panel)",
        transition="width 0.25s ease-in-out",
        on_mouse_enter=PlotState.open_sidebar,
        on_mouse_leave=PlotState.close_sidebar,
        position="relative",
        z_index="10",
    )


def _color_picker_item(group: str) -> rx.Component:
    """Render a single color picker for a categorical group."""
    return rx.hstack(
        rx.el.input(
            type="color",
            value=PlotState.color_map[group],
            on_change=lambda c: PlotState.update_group_color(group, c),
            width="28px", height="28px",
            style={"padding": "0", "border": "none", "cursor": "pointer", "background": "transparent"},
        ),
        rx.text(group, size="1"),
        align="center", spacing="2",
    )


def _main_canvas() -> rx.Component:
    """Main content area — filters, expression builder, and plot."""
    return rx.box(
        rx.vstack(
            # ── Header ──
            rx.hstack(
                rx.heading("Dynamic CSV Visualizer", size="6"),
                rx.spacer(),
                rx.color_mode.button(),
                width="100%",
            ),

            # ── Data preview (collapsible) ──
            rx.cond(
                PlotState.data_loaded,
                rx.vstack(
                    rx.hstack(
                        rx.switch(checked=PlotState.show_preview, on_change=PlotState.toggle_preview, size="1"),
                        rx.text("Show Column List", size="2", color="gray"),
                        align="center",
                    ),
                    rx.cond(
                        PlotState.show_preview,
                        rx.card(
                            rx.flex(
                                rx.foreach(PlotState.columns, lambda c: rx.badge(c, variant="surface", size="1")),
                                wrap="wrap", spacing="2",
                            ),
                            width="100%", size="1",
                        ),
                        rx.fragment(),
                    ),
                    width="100%", spacing="2",
                ),
                rx.fragment(),
            ),

            # ── Filter Panel (collapsible, in main area) ──
            rx.cond(
                PlotState.data_loaded,
                rx.card(
                    rx.vstack(
                        rx.hstack(
                            rx.hstack(
                                rx.icon("filter", size=16),
                                rx.heading("Filters", size="4"),
                                align="center", spacing="2",
                            ),
                            rx.spacer(),
                            rx.hstack(
                                rx.switch(checked=PlotState.show_filters, on_change=PlotState.toggle_filters, size="1"),
                                rx.text("Show", size="1", color="gray"),
                                align="center",
                            ),
                            width="100%", align="center",
                        ),
                        rx.cond(
                            PlotState.show_filters,
                            rx.vstack(
                                rx.hstack(
                                    rx.select(
                                        PlotState.filter_cols_available,
                                        value=PlotState.filter_col_dropdown,
                                        on_change=PlotState.set_filter_col_dropdown,
                                        placeholder="Select column to filter…",
                                        size="2", width="280px",
                                    ),
                                    rx.button("Add Filter", on_click=PlotState.add_filter, size="2", variant="soft", color_scheme="crimson"),
                                    rx.cond(
                                        PlotState.has_filters,
                                        rx.button("Clear All", on_click=PlotState.clear_all_filters, size="1", variant="ghost", color_scheme="gray"),
                                        rx.fragment(),
                                    ),
                                    align="center", spacing="2",
                                ),
                                rx.flex(
                                    rx.foreach(PlotState.filters, lambda f: rx.box(_filter_card(f), min_width="250px", max_width="320px", flex="1")),
                                    wrap="wrap", spacing="3", width="100%",
                                ),
                                spacing="3", width="100%",
                            ),
                            rx.fragment(),
                        ),
                        spacing="2", width="100%",
                    ),
                    width="100%",
                ),
                rx.fragment(),
            ),

            # ── Expression Builder & Plot Config ──
            rx.cond(
                PlotState.data_loaded,
                rx.card(
                    rx.vstack(
                        rx.hstack(
                            rx.icon("calculator", size=16),
                            rx.text("Axis & Color Configuration", weight="bold", size="3"),
                            align="center", spacing="2",
                        ),
                        # Builder row
                        rx.hstack(
                            rx.text("Insert column:", size="1", color="gray"),
                            rx.select(
                                PlotState.columns, placeholder="Column…",
                                size="1", width="180px",
                                on_change=PlotState.set_builder_col_dropdown,
                            ),
                            rx.button("→ X", on_click=PlotState.append_col_to_x, size="1", variant="outline"),
                            rx.button("→ Y", on_click=PlotState.append_col_to_y, size="1", variant="outline"),
                            align="center", spacing="2", wrap="wrap",
                        ),
                        # Axis inputs
                        rx.flex(
                            rx.vstack(
                                rx.text("X-Axis Expression", weight="bold", size="2"),
                                rx.hstack(
                                    rx.input(value=PlotState.x_expr, on_change=PlotState.set_x_expr, placeholder="e.g.  col_A + col_B * 2", size="2", width="100%"),
                                    rx.icon_button(rx.icon("eraser"), size="1", variant="ghost", on_click=PlotState.clear_x),
                                ),
                                width="100%", flex="1", min_width="200px",
                            ),
                            rx.vstack(
                                rx.text("Y-Axis Expression", weight="bold", size="2"),
                                rx.hstack(
                                    rx.input(value=PlotState.y_expr, on_change=PlotState.set_y_expr, placeholder="e.g.  log(col_C)", size="2", width="100%"),
                                    rx.icon_button(rx.icon("eraser"), size="1", variant="ghost", on_click=PlotState.clear_y),
                                ),
                                width="100%", flex="1", min_width="200px",
                            ),
                            rx.vstack(
                                rx.text("Color By", weight="bold", size="2"),
                                rx.select(PlotState.color_options, value=PlotState.color_col, on_change=PlotState.set_color_col, size="2", width="100%"),
                                width="100%", flex="1", min_width="150px",
                            ),
                            rx.vstack(
                                rx.text("Color Style", weight="bold", size="2"),
                                rx.cond(
                                    PlotState.has_color_col,
                                    rx.vstack(
                                        rx.checkbox("Categorical", checked=PlotState.force_cat_color, on_change=PlotState.set_force_cat_color, size="1"),
                                        rx.cond(
                                            ~PlotState.force_cat_color,
                                            rx.select(PALETTE_OPTIONS, value=PlotState.cont_palette, on_change=PlotState.set_cont_palette, size="1", width="100%"),
                                            rx.fragment(),
                                        ),
                                        spacing="1",
                                    ),
                                    rx.text("—", color="gray", size="2"),
                                ),
                                width="100%", flex="1", min_width="130px",
                            ),
                            spacing="4", wrap="wrap", width="100%",
                        ),
                        # ── Per-group color pickers (categorical) ──
                        rx.cond(
                            PlotState.is_discrete_color,
                            rx.vstack(
                                rx.hstack(
                                    rx.switch(checked=PlotState.show_color_pickers, on_change=PlotState.toggle_color_pickers, size="1"),
                                    rx.text("Customize Group Colors", size="2", color="gray"),
                                    align="center",
                                ),
                                rx.cond(
                                    PlotState.show_color_pickers,
                                    rx.vstack(
                                        rx.flex(
                                            rx.foreach(PlotState.color_groups, _color_picker_item),
                                            wrap="wrap", spacing="3",
                                        ),
                                        rx.button("Apply Colors", on_click=PlotState.apply_colors, size="2", color_scheme="violet", variant="soft", width="160px"),
                                        spacing="2",
                                    ),
                                    rx.fragment(),
                                ),
                                spacing="2", width="100%",
                            ),
                            rx.fragment(),
                        ),
                        # ── Point style sliders ──
                        rx.flex(
                            rx.vstack(
                                rx.text("Point Size", weight="bold", size="2"),
                                rx.slider(min=1, max=20, step=0.5, value=PlotState.point_size, on_change=PlotState.set_point_size, width="160px", size="1"),
                                spacing="1",
                            ),
                            rx.vstack(
                                rx.text("Opacity", weight="bold", size="2"),
                                rx.slider(min=0.05, max=1.0, step=0.05, value=PlotState.point_alpha, on_change=PlotState.set_point_alpha, width="160px", size="1"),
                                spacing="1",
                            ),
                            rx.button("Generate Plot", on_click=PlotState.generate_plot, size="3", color_scheme="jade", width="220px"),
                            spacing="4", align="end", wrap="wrap",
                        ),
                        spacing="3",
                    ),
                    width="100%",
                ),
                rx.fragment(),
            ),

            # ── Full-width Plot ──
            rx.box(
                rx.plotly(
                    data=PlotState.plot_fig,
                    height="750px",
                    width="100%",
                    on_selected=PlotState.handle_plot_selected,
                ),
                width="100%",
                border="1px solid var(--gray-4)",
                border_radius="8px",
                overflow="hidden",
            ),

            # ── Selected Points Table ──
            rx.cond(
                PlotState.has_selection,
                rx.card(
                    rx.vstack(
                        rx.hstack(
                            rx.hstack(
                                rx.icon("table-2", size=16),
                                rx.text("Selected Data", weight="bold", size="3"),
                                align="center", spacing="2",
                            ),
                            rx.badge(PlotState.selected_count.to(str) + " points", color_scheme="blue", size="1"),
                            rx.spacer(),
                            rx.button("Clear", on_click=PlotState.clear_selection, size="1", variant="ghost", color_scheme="gray"),
                            align="center", width="100%",
                        ),
                        rx.scroll_area(
                            rx.el.table(
                                rx.el.thead(
                                    rx.el.tr(
                                        rx.foreach(
                                            PlotState.selected_columns,
                                            lambda col: rx.el.th(
                                                col,
                                                style={"padding": "6px 10px", "fontSize": "12px", "fontWeight": "600",
                                                       "borderBottom": "1px solid var(--gray-6)", "whiteSpace": "nowrap",
                                                       "position": "sticky", "top": "0", "background": "var(--color-panel)"},
                                            ),
                                        ),
                                    ),
                                ),
                                rx.el.tbody(
                                    rx.foreach(
                                        PlotState.selected_rows,
                                        lambda row: rx.el.tr(
                                            rx.foreach(
                                                PlotState.selected_columns,
                                                lambda col: rx.el.td(
                                                    row[col],
                                                    style={"padding": "4px 10px", "fontSize": "11px",
                                                           "borderBottom": "1px solid var(--gray-3)", "whiteSpace": "nowrap"},
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                                style={"borderCollapse": "collapse", "width": "100%"},
                            ),
                            type="always", scrollbars="both",
                            style={"maxHeight": "300px"},
                        ),
                        spacing="2", width="100%",
                    ),
                    width="100%",
                ),
                rx.fragment(),
            ),

            spacing="4",
            width="100%",
            padding="1.5em",
        ),
        flex="1",
        min_width="0",
        min_height="100vh",
        overflow_y="auto",
    )


# ═════════════════════════════════════════════════════════════
#  PAGE
# ═════════════════════════════════════════════════════════════

def index() -> rx.Component:
    return rx.flex(
        _sidebar(),
        _main_canvas(),
        spacing="0",
        width="100%",
        min_height="100vh",
    )


app = rx.App(
    theme=rx.theme(
        appearance="dark",
        has_background=True,
        radius="large",
        accent_color="indigo",
    )
)
app.add_page(index)

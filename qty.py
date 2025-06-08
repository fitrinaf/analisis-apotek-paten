import streamlit as st
import pandas as pd
import numpy as np
import base64
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform


# Inisialisasi awal untuk session_state
if 'df_result' not in st.session_state:
    st.session_state.df_result = None
if 'optimal_k' not in st.session_state:
    st.session_state.optimal_k = None
if 'monthly_files' not in st.session_state:
    st.session_state.monthly_files = {}
if 'integrated_df' not in st.session_state:
    st.session_state.integrated_df = None

# Set page configuration
st.set_page_config(
    page_title="Apotek Data Analytics",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS to improve appearance with more colors and better styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #4E54C8;
        --secondary-color: #8F94FB;
        --accent-color: #FF6B6B;
        --light-color: #F0F3FF;
        --dark-color: #2A2D43;
        --success-color: #00C853;
        --warning-color: #FFD740;
        --danger-color: #FF5252;
    }
    
    /* Main header styling */
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: bold;
        padding: 0.5rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Sub header styling */
    .sub-header {
        font-size: 1.6rem;
        color: var(--primary-color);
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-left: 5px solid var(--accent-color);
        padding-left: 10px;
        font-weight: 600;
    }
    
    /* Card styling with better shadow and hover effect */
    .card {
        border-radius: 12px;
        background: linear-gradient(145deg, #ffffff, #f5f7ff);
        padding: 0.3rem;
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
        transition: all 0.3s;
        border-top: 4px solid var(--primary-color);
        margin-bottom: 1.5rem;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 20px rgba(0, 0, 0, 0.15);
    }
    
    /* Highlight text */
    .highlight-text {
        color: var(--accent-color);
        font-weight: bold;
    }
    
    /* Metric container styling */
    .metric-container {
        background: linear-gradient(145deg, #ffffff, #f0f6ff);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.08);
        margin-bottom: 1rem;
        border-left: 4px solid var(--primary-color);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, var(--primary-color), var(--dark-color));
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    
    /* Expander styling */
    .st-expander {
        border-radius: 10px;
        border: 1px solid #e0e0ff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    
    /* DataTable styling */
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.08);
        padding: 0.5rem;
    }
    
    /* Info box styling */
    .stAlert {
        border-radius: 10px;
        border-left: 5px solid var(--primary-color);
    }
    
    /* Section styling */
    .section {
        padding: 1.5rem;
        margin-bottom: 2rem;
        border-radius: 12px;
        background-color: #ffffff;
        box-shadow: 0 6px 12px rgba(0,0,0,0.08);
        border-top: 5px solid var(--secondary-color);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #f5f7ff;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        border-bottom: none;
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color) !important;
        color: white !important;
        font-weight: bold;
    }
    
    /* Status indicators */
    .status-high {
        color: var(--success-color);
        font-weight: bold;
    }
    
    .status-medium {
        color: var(--warning-color);
        font-weight: bold;
    }
    
    .status-low {
        color: var(--danger-color);
        font-weight: bold;
    }
    
    /* Dashboard title banner */
    .dashboard-banner {
        background: linear-gradient(120deg, var(--primary-color), var(--secondary-color));
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.15);
    }
    
    .banner-title {
        font-size: 2.2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .banner-subtitle {
        opacity: 0.9;
        font-size: 1.1rem;
    }
    
    /* Navigation button */
    .nav-button {
        background-color: rgba(255,255,255,0.2);
        border-radius: 8px;
        padding: 12px 20px;
        margin: 8px 0;
        transition: all 0.3s;
        border-left: 4px solid transparent;
    }
    
    .nav-button:hover, .nav-button.active {
        background-color: rgba(255,255,255,0.3);
        border-left: 4px solid var(--accent-color);
    }
    
    /* Stat card */
    .stat-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.08);
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        transition: all 0.3s;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.12);
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        color: var(--primary-color);
        margin: 0.5rem 0;
    }
    
    .stat-label {
        color: #666;
        font-size: 0.9rem;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background-color: var(--primary-color);
    }
    
    /* Upload box */
    .stUploadedFile {
        border-radius: 10px;
        border: 2px dashed var(--primary-color);
        padding: 10px;
    }
    
    /* Month tag styling */
    .month-tag {
        display: inline-block;
        padding: 4px 8px;
        background-color: var(--light-color);
        border-left: 3px solid var(--primary-color);
        border-radius: 4px;
        margin-right: 8px;
        font-size: 0.85rem;
        color: var(--dark-color);
    }
    
    /* Active month */
    .month-tag.active {
        background-color: var(--primary-color);
        color: white;
        border-left: 3px solid var(--accent-color);
    }
    
    /* Month selector container */
    .month-selector {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 1rem;
    }
    
    /* Summary card for integrated data */
    .summary-card {
        border-radius: 12px;
        background: linear-gradient(145deg, #ffffff, #f0f5ff);
        padding: 1rem;
        box-shadow: 0 6px 12px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
        border-left: 5px solid var(--accent-color);
    }
</style>
""", unsafe_allow_html=True)

# Function to create download link for dataframes
def get_table_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv" class="download-button">{text}</a>'
    return href

# Function to create download link for excel files
def get_excel_download_link(df, filename, text):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Check if df has MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            # Keep index when writing DataFrame with MultiIndex columns
            df.to_excel(writer, sheet_name='Sheet1')
        else:
            # For regular DataFrame, you can still use index=False
            df.to_excel(writer, index=False, sheet_name='Sheet1')
    excel_data = output.getvalue()
    b64 = base64.b64encode(excel_data).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}.xlsx" class="download-button">{text}</a>'
    return href

# Dictionary of month names in Indonesian
month_names = {
    1: "Januari", 2: "Februari", 3: "Maret", 4: "April", 5: "Mei", 6: "Juni",
    7: "Juli", 8: "Agustus", 9: "September", 10: "Oktober", 11: "November", 12: "Desember"
}

# Initialize session state
if 'df_raw' not in st.session_state:
    st.session_state.df_raw = None
if 'df_cleaned' not in st.session_state:
    st.session_state.df_cleaned = None
if 'df_aggregated' not in st.session_state:
    st.session_state.df_aggregated = None
if 'df_transformed' not in st.session_state:
    st.session_state.df_transformed = None
if 'optimal_k' not in st.session_state:
    st.session_state.optimal_k = 3
if 'cluster_labels' not in st.session_state:
    st.session_state.cluster_labels = None
if 'silhouette_avg' not in st.session_state:
    st.session_state.silhouette_avg = None
if 'centroids' not in st.session_state:
    st.session_state.centroids = None

# Create stylish dashboard header banner
st.markdown("""
<div class="dashboard-banner">
    <div class="banner-title">Analisis Data Apotek Paten</div>
    <div class="banner-subtitle">K-Means Clustering untuk Analisis Data Produk</div>
</div>
""", unsafe_allow_html=True)

# Create tabs for different steps
tabs = st.tabs(["Upload Data Bulanan", "Hasil Analisis"])

with tabs[0]:
    st.markdown("<h2 class='sub-header'>📊 Upload Data Bulanan</h2>", unsafe_allow_html=True)
    
    with st.expander("Instruksi Upload Data", expanded=True):
        st.markdown("""
        <div class="card">
            <h4 style="color: #4E54C8; margin-bottom: 1rem;">👨‍💻 Panduan Upload Data Bulanan</h4>
            <ol>
                <li>Upload file Excel data apotek untuk setiap bulan.</li>
                <li>Setiap file harus memiliki kolom <span class="highlight-text">'Item Name'</span> dan <span class="highlight-text">'Qty'</span> minimal.</li>
                <li>Pilih bulan yang sesuai untuk masing-masing file.</li>
                <li>Setelah semua file diupload, lanjutkan ke halaman Hasil Analisis.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    # Month selection
    selected_month = st.selectbox(
            "Pilih Bulan:",
            options=list(range(1, 13)),
            format_func=lambda x: month_names[x]
    )
    

        # Create attractive upload area
    uploaded_file = st.file_uploader(
            "", 
            type=["xlsx", "xls"], 
            key=f"file_uploader_{selected_month}",
            label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        try:
            # Create progress bar for loading
            progress_bar = st.progress(0)
            
            # Load data
            progress_bar.progress(25)
            df = pd.read_excel(uploaded_file)
            
            # Clean column names
            df.columns = df.columns.str.strip()
            progress_bar.progress(50)
            
            # Check if required columns exist
            required_cols = ['Item Name', 'Qty', 'SubTotal']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Kolom berikut tidak ditemukan dalam data: {', '.join(missing_cols)}")
                st.warning("⚠️ Pastikan file Excel memiliki kolom 'Item Name', 'Qty', dan 'SubTotal'")
            else:
                # Selection columns
                df_selected = df[['Item Name', 'Qty', 'SubTotal']].copy()
                progress_bar.progress(75)
                
                # Clean data Pre-Processing
                # Check null values
                null_count = df_selected.isnull().sum()
    
                # Option to drop null values
                if df_selected.isnull().sum().sum() > 0:
                    pre_drop_count = len(df_selected)
                    df_selected.dropna(inplace=True)
                    post_drop_count = len(df_selected)
                
                # Ensure numeric data types
                df_selected['Qty'] = pd.to_numeric(df_selected['Qty'], errors='coerce')
                
                # Group by product name
                pre_group_count = len(df_selected)
                df_agg = df_selected.groupby(['Item Name']).agg({
                    'Qty': 'sum'
                }).reset_index()
                post_group_count = len(df_agg)
                progress_bar.progress(100)
                
                # Store in session state with month suffix
                month_key = month_names[selected_month].lower()
                st.session_state.monthly_files[month_key] = {
                    'df_raw': df,
                    'df_selected': df_selected,
                    'df_agg': df_agg,
                    'month_number': selected_month,
                    'month_name': month_names[selected_month]
                }
                
                # Display success message
                st.success(f"✅ Data bulan {month_names[selected_month]} berhasil di-upload dan diproses!")
                
                # Display preview
                #with st.expander(f"Preview Data Bulan {month_names[selected_month]}", expanded=True):
                    # Data stats in cards
                    #col1, col2, col3 = st.columns(3)
                    #with col1:
                     #   st.markdown("""
                      #  <div class="stat-card" style="margin-bottom: 1rem;">
                       #     <div class="stat-label">Jumlah Produk</div>
                        #    <div class="stat-number">{}</div>
                    #    </div>
                     #   """.format(len(df_agg)), unsafe_allow_html=True)
                    
                  #  with col2:
                   #     st.markdown("""
                    #    <div class="stat-card" style="margin-bottom: 1rem;">
                     #       <div class="stat-label">Total Qty</div>
                      #      <div class="stat-number">{:,}</div>
                    #    </div>
                     #   """.format(int(df_agg['Qty'].sum())), unsafe_allow_html=True)
                    
                   # with col3:
                    #    st.markdown("""
                     #   <div class="stat-card" style="margin-bottom: 1rem;">
                      #      <div class="stat-label">Total Pendapatan</div>
                       #     <div class="stat-number">Rp {:,.0f}</div>
                     #   </div>
                     #   """.format(df_agg['SubTotal'].sum()), unsafe_allow_html=True)
                    
                  #  st.dataframe(df_agg, use_container_width=True)
                    
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    
    # Display uploaded files summary
    if st.session_state.monthly_files:
        st.markdown("<h3 class='sub-header' style='margin-bottom: 1rem;'>🗂️ File Data yang Sudah Diupload</h3>", unsafe_allow_html=True)
        
        # Create grid of month cards
        cols = st.columns(4)
        
        for i, (month_key, data) in enumerate(sorted(st.session_state.monthly_files.items(), 
                                                     key=lambda x: x[1]['month_number'])):
            with cols[i % 4]:
                month_name = data['month_name']
                
                st.markdown(f"""
                <div class="stat-card" style="margin-bottom: 1rem;">
                    <div class="stat-label">Bulan</div>
                    <div class="stat-number">{month_name}</div>
                    <div>
                        <span class="month-tag" style="margin-bottom: 5px;">Product: {len(data['df_agg'])}</span>
                        <span class="month-tag">Qty: {int(data['df_agg']['Qty'].sum()):,}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # st.success(f"✅ Data berhasil di-upload dan diproses!")


# with tabs[1]:
    #st.markdown("<h2 class='sub-header'>🔄 Integrasi & Transformasi Data</h2>", unsafe_allow_html=True)
    
    if len(st.session_state.monthly_files) < 1:
        st.warning("⚠️ Harap upload terlebih dahulu")
    else:
        
                # Create progress bar
        progress_bar = st.progress(0)
                
                # Get all unique product names across all months
        all_products = set()
        for month_data in st.session_state.monthly_files.values():
            products = set(month_data['df_agg']['Item Name'])
            all_products.update(products)
                
        progress_bar.progress(20)
                
                # Create a base dataframe with all product names
        df_integrated = pd.DataFrame({'Item Name': list(all_products)})
                
                # Progress counter
        total_months = len(st.session_state.monthly_files)
        current_month = 0
                
                # Add data for each month
        for month_key, month_data in sorted(st.session_state.monthly_files.items(), 
                                               key=lambda x: x[1]['month_number']):
            month_name = month_data['month_name']
            df_month = month_data['df_agg']
                    
                    # Merge with integrated dataframe
            df_integrated = df_integrated.merge(
                df_month[['Item Name', 'Qty']], 
                    on='Item Name', 
                        how='left',
                    suffixes=('', f'_{month_key}')
                )
                    
                    # Rename columns to add month name
            df_integrated.rename(columns={
                'Qty': f'Qty_{month_key}',
            }, inplace=True)
                    
                    # Replace NaN with 0 for this month's columns
            df_integrated[f'Qty_{month_key}'].fillna(0, inplace=True)
                    
                    # Update progress
            current_month += 1
            progress_bar.progress(20 + int(70 * current_month / total_months))
                
            # Calculate total metrics across all months for each product
            qty_cols = [col for col in df_integrated.columns if col.startswith('Qty_')]
                
                # Store in session state
            st.session_state.integrated_df = df_integrated
                
                # Complete progress
            progress_bar.progress(100)
            
        # Display integrated data if available
        df_integrated['Total_Qty'] = df_integrated[qty_cols].sum(axis=1)
      
        df_integrated = st.session_state.integrated_df
            
        st.markdown("<h3 class='sub-header' style='margin-top: 1rem;'>📊 Dataset</h3>", unsafe_allow_html=True)
            
            # Display summary stats
        col1, col2 = st.columns(2)
            
        with col1:
                st.markdown("""
                <div class="stat-card" style="margin: 1rem 0;">
                    <div class="stat-label">Jumlah Produk</div>
                    <div class="stat-number">{}</div>
                </div>
                """.format(len(df_integrated)), unsafe_allow_html=True)
            
        with col2:
                st.markdown("""
                <div class="stat-card"  style="margin: 1rem 0";>
                    <div class="stat-label">Total Qty</div>
                    <div class="stat-number">{:,}</div>
                </div>
                """.format(int(df_integrated['Total_Qty'].sum())), unsafe_allow_html=True)
            
            # Create preview of integrated data
        with st.expander("Preview Dataset", expanded=True):
                preview_df = df_integrated.drop(columns=['Total_Qty'])
                preview_df = preview_df.sort_values(by='Item Name', ignore_index=True)
                st.dataframe(preview_df, use_container_width=True)
            
        # Data Transformation Section

        # Create columns for transformation - consider all monthly columns and totals
        selected_cols = [col for col in df_integrated.columns if col.startswith('Qty_')]

        df_selected = df_integrated[selected_cols].copy()
                    
        # Normalization with progress animation
        my_bar = st.progress(0)
                    
        # Normalize data
        scaler = MinMaxScaler()
        df_scaled = scaler.fit_transform(df_selected)
                    
        # Convert scaled data back to DataFrame
        df_transformed = pd.DataFrame(df_scaled, columns=df_selected.columns)
                    
        # Add product names
        df_transformed['Item Name'] = df_integrated['Item Name'].values

        df_transformed = df_transformed.sort_values(by='Item Name', ignore_index=True)
                    
        # Save to session state
        st.session_state.df_transformed = df_transformed
        st.session_state.df_aggregated = df_integrated
                    
        my_bar.progress(100)
                
        # st.success("✅ Data berhasil ditransformasi! Silakan lanjutkan ke tab 'Clustering & Analisis'")
                
                # Display transformed data preview
                #with st.expander("Preview Data Transformasi", expanded=True):
                 #   st.dataframe(df_transformed, use_container_width=True)

with tabs[1]:
    st.markdown("<h2 class='sub-header' style='margin-bottom: 15px;'>📊 Hasil Analisis</h2>", unsafe_allow_html=True)

    if st.session_state.df_transformed is None:
        st.warning("⚠️ Harap upload data pada halaman Upload Data Bulanan terlebih dahulu")
    else:

        # Clustering Parameters
        #st.markdown("<h3>Parameter Clustering</h3>", unsafe_allow_html=True)
        #col1, col2 = st.columns(2)

        #with col1:
         #   k_min = st.number_input("K Min", min_value=2, max_value=10, value=2, key="k_min")
         #   k_max = st.number_input("K Max", min_value=k_min + 1, max_value=15, value=10, key="k_max")

        #with col2:
         #   random_state = st.number_input("Random State", min_value=0, max_value=100, value=42, key="random_state")
          #  max_iter = st.number_input("Max Iterations", min_value=100, max_value=1000, value=300, step=100, key="max_iter")

        # Tombol untuk menjalankan clustering
                df_for_clustering = st.session_state.df_transformed.drop(['Item Name'], axis=1)

                progress_bar = st.progress(0)
                silhouette_scores = []
                k_values = list(range(2, 11))
                
                for i, k in enumerate(k_values):
                    progress_bar.progress((i / len(k_values)) * 0.6)
                    kmeans = KMeans(n_clusters=k, random_state=42, max_iter=300, n_init=10)
                    cluster_labels = kmeans.fit_predict(df_for_clustering)
                    score = silhouette_score(df_for_clustering, cluster_labels)
                    silhouette_scores.append(score)

                optimal_k = k_values[np.argmax(silhouette_scores)]
                st.session_state.optimal_k = optimal_k

                progress_bar.progress(0.7)
                kmeans = KMeans(n_clusters=optimal_k, random_state=42, max_iter=300, n_init=10)
                cluster_labels = kmeans.fit_predict(df_for_clustering)
                silhouette_avg = silhouette_score(df_for_clustering, cluster_labels)

                st.session_state.cluster_labels = cluster_labels
                st.session_state.silhouette_avg = silhouette_avg
                st.session_state.centroids = kmeans.cluster_centers_

                # Pastikan df_aggregated sudah ada
                if 'df_aggregated' in st.session_state:
                    st.session_state.df_aggregated['Cluster'] = cluster_labels

                progress_bar.progress(1.0)
               # st.success(f"✅ K-Means clustering selesai! Jumlah cluster optimal: {optimal_k}")
                # Get results and continue with analysis
                df_result = st.session_state.df_aggregated.copy()
                df_result['Cluster'] = st.session_state.cluster_labels
                st.session_state.df_result = df_result

            # Continue with visualizations only if clustering has been completed
                # Elbow curve visualization
             #   st.markdown("<h3 class='sub-header'>📈 Analisis Jumlah Cluster Optimal</h3>", unsafe_allow_html=True)
                
                df_for_clustering = st.session_state.df_transformed.drop(['Item Name'], axis=1)
                k_range = range(2, 11)
                
                # Calculate inertia (within-cluster sum of squares)
                inertia = []
                silhouette_scores = []
                
                for k in k_range:
                    kmeans = KMeans(n_clusters=k, random_state=42, max_iter=300, n_init=10)
                    kmeans.fit(df_for_clustering)
                    inertia.append(kmeans.inertia_)
                    
                    labels = kmeans.labels_
                    silhouette_scores.append(silhouette_score(df_for_clustering, labels))
                
                # Display cluster statistics
             #   st.markdown("<h3 class='sub-header'>🧩 Analisis Hasil Clustering</h3>", unsafe_allow_html=True)
                
                # Hitung jarak Euclidean antar produk
                df_clustering_only = df_for_clustering.copy()
                distance_matrix = pdist(df_clustering_only.values, metric='euclidean')  # vektor 1D
                euclidean_df = pd.DataFrame(
                squareform(distance_matrix),  # ubah ke bentuk matriks
                index=st.session_state.df_aggregated['Item Name'],
                columns=st.session_state.df_aggregated['Item Name']
                )

                # Simpan dalam session_state jika ingin ditampilkan nanti
                st.session_state.euclidean_df = euclidean_df

                # Basic cluster statistics
                cluster_stats = df_result.groupby('Cluster').agg({
                    'Item Name': 'count',
                    'Total_Qty': 'sum'
                }).reset_index()
                
                cluster_stats.rename(columns={
                    'Item Name': 'Jumlah Produk',
                    'Total_Qty': 'Total Quantity'
                }, inplace=True)
                
                # Calculate percentage of total
                total_products = cluster_stats['Jumlah Produk'].sum()
                total_qty = cluster_stats['Total Quantity'].sum()
                
                cluster_stats['% Produk'] = (cluster_stats['Jumlah Produk'] / total_products * 100).round(2)
                cluster_stats['% Quantity'] = (cluster_stats['Total Quantity'] / total_qty * 100).round(2)
                
                # Scatter plot hasil clustering (2 fitur pertama untuk visualisasi)
    # Ambil 2 fitur pertama untuk scatter plot
                X = st.session_state.df_transformed.drop(['Item Name'], axis=1)
    
                scatter_df = df_result.copy()
                    # Jalankan PCA ke 2 komponen
                pca = PCA(n_components=2, random_state=42)
                X_pca = pca.fit_transform(X)

                # Ambil nama-nama bulan dari kolom Qty_*
                bulan = [col.replace('Qty_', '') for col in df_integrated.columns if col.startswith('Qty_')]

                # Ambil bulan pertama dan terakhir (diasumsikan urut)
                rentang_bulan = f"{bulan[0]} - {bulan[-1]}"
    
    # Buat dataframe untuk plotting
                scatter_df = pd.DataFrame({
                    'Dimensi 1': X_pca[:,0],
                    'Dimensi 2': X_pca[:,1],
                    'Cluster': st.session_state.cluster_labels,
                    'Item Name': st.session_state.df_transformed['Item Name']
                })
                
                # IMPORTANT: Scatter plot adalah referensi utama
                # Update df_result agar cluster-nya mengikuti scatter plot
                scatter_cluster_mapping = dict(zip(scatter_df['Item Name'], scatter_df['Cluster']))
                df_result['Cluster'] = df_result['Item Name'].map(scatter_cluster_mapping)
                st.session_state.df_result = df_result
    
                fig_scatter = px.scatter(
                    scatter_df,
                    x='Dimensi 1',
                    y='Dimensi 2',
                    color='Cluster',
                    title=f'Visualisasi Pola Penjualan Produk ({rentang_bulan})',
                    hover_data=['Item Name'],
                    color_continuous_scale='viridis'
                    
                )

                centroids = scatter_df.groupby('Cluster')[['Dimensi 1', 'Dimensi 2']].mean().reset_index()
                fig_scatter.add_trace(go.Scatter(
                    x=centroids['Dimensi 1'],
                    y=centroids['Dimensi 2'],
                    mode='markers',
                    name='Centroid',
                    marker=dict(color='red', size=15, symbol='x'),
                    hovertemplate='Cluster %{text}<br>Dimensi 1: %{x:.2f}<br>Dimensi 2: %{y:.2f}',
                    text=centroids['Cluster'].astype(str)
                ))

                fig_scatter.update_layout(
                    plot_bgcolor='white',  # background dalam plot putih
                    paper_bgcolor='white', # background luar plot putih
                    xaxis=dict(
                        showline=True,      # tampilkan garis sumbu x
                        linecolor='black',  # warna garis sumbu x
                        linewidth=0.5,
                        mirror=True,         # garis sumbu juga muncul di atas
                        showgrid=True,     # grid garis putus-putus
                        gridcolor='gray',
                        gridwidth=0.5,
                        griddash='dot'  
                    ),
                    yaxis=dict(
                        showline=True,      # tampilkan garis sumbu y
                        linecolor='black',  # warna garis sumbu y
                        linewidth=0.5,
                        mirror=True,         # garis sumbu juga muncul di kanan
                        showgrid=True,
                        gridcolor='gray',
                        gridwidth=0.5,
                        griddash='dot'
                    )
                )
                
    
                st.plotly_chart(fig_scatter, use_container_width=True)

                # Display cluster statistics
                # Tabel Statistik Clustering yang Baru
                st.markdown("""
                <div class="summary-card">
                    <h4>📊 Statistik Kelompok Penjualan</h4>
                    <p>Rincian statistik dari masing-masing kelompok produk yang terbentuk:</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Hitung statistik lengkap per cluster
                detailed_stats = []
                monthly_cols = [col for col in df_result.columns if col.startswith('Qty_')]
                
                for cluster_id in sorted(df_result['Cluster'].unique()):
                    cluster_data = df_result[df_result['Cluster'] == cluster_id]
                    
                    # Basic stats
                    stats = {
                        'Kelompok': cluster_id,
                        'Jumlah Produk': len(cluster_data),
                        'Total Quantity': cluster_data['Total_Qty'].sum(),
                        'Rata-rata Qty': cluster_data['Total_Qty'].mean(),
                        'Median Qty': cluster_data['Total_Qty'].median(),
                        'Min Qty': cluster_data['Total_Qty'].min(),
                    }
                    
                    detailed_stats.append(stats)
                
                detailed_stats_df = pd.DataFrame(detailed_stats)
                
                # Format numbers
                numeric_cols = ['Total Quantity', 'Rata-rata Qty per Produk', 'Median Qty', 'Min Qty', 'Max Qty', 'Std Deviasi']
                for col in numeric_cols:
                    if col in detailed_stats_df.columns:
                        detailed_stats_df[col] = detailed_stats_df[col].round(1)
                
                # Calculate percentages
                total_products = detailed_stats_df['Jumlah Produk'].sum()
                total_qty = detailed_stats_df['Total Quantity'].sum()
                detailed_stats_df['% Produk'] = (detailed_stats_df['Jumlah Produk'] / total_products * 100).round(1)
                detailed_stats_df['% Quantity'] = (detailed_stats_df['Total Quantity'] / total_qty * 100).round(1)
                
                st.dataframe(detailed_stats_df, use_container_width=True, hide_index=True)
                
                
                # Create a color palette for the clusters
                colors = px.colors.qualitative.Bold
                
                # Extract feature names for centroids
                centroid_features = st.session_state.df_transformed.drop(['Item Name'], axis=1).columns
                
                # Create centroid dataframe
                centroids_df = pd.DataFrame(
                    st.session_state.centroids,
                    columns=centroid_features
                )
                
                # Add cluster column
                centroids_df['Cluster'] = range(len(centroids_df))
                
                # Create feature importance / radar chart for each cluster
                # Select the most important features (e.g., Total Qty, Total Pendapatan, Avg Price)
                
                
                # Filter centroids to important features only
                centroids_viz = centroids_df[['Cluster']]
                
                # Create a radar chart for each cluster
                
                
                # Display products in each cluster
                st.markdown("<h3 class='sub-header' style='margin-bottom: 20px;'>📋 Daftar Produk per Kelompok</h3>", unsafe_allow_html=True)
                
                # Create tabs for each cluster
                cluster_tabs = st.tabs([f"Kelompok {i}" for i in range(st.session_state.optimal_k)])
                
                # Add content for each cluster tab
                for i, tab in enumerate(cluster_tabs):
                    with tab:
                        cluster_data = df_result[df_result['Cluster'] == i].copy()

                        # Create summary statistics
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"""
                            <div style="margin-bottom: 1rem;" class="stat-card">
                                <div class="stat-label">Jumlah Produk</div>
                                <div class="stat-number">{len(cluster_data)}</div>
                                <div class="stat-label">({cluster_stats.iloc[i]['% Produk']}% dari total)</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div style="margin-bottom: 1rem;" class="stat-card">
                                <div class="stat-label">Total Quantity</div>
                                <div class="stat-number">{int(cluster_data['Total_Qty'].sum()):,}</div>
                                <div class="stat-label">({cluster_stats.iloc[i]['% Quantity']}% dari total)</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("""
                            <div class="summary-card" style="margin: 10px 0 ;">
                            <h5>🔥 Produk dengan Penjualan Tertinggi</h5>
                            <p>Grafik ini menampilkan 10 produk dengan total penjualan tertinggi berdasarkan kelompok</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Sort by Total_Qty or another relevant metric
                        if 'Total_Qty' in cluster_data.columns:
                            cluster_data = cluster_data.sort_values('Total_Qty', ascending=False)
                        
                        top_10_products = cluster_data.head(10).copy()
                        top_10_products['Ranking'] = range(1, len(top_10_products) + 1)
                        

                        fig_bar = px.bar(
                top_10_products,
                x='Total_Qty',
                y='Item Name',
                orientation='h',
                title=f'Top 10 Produk Terlaris - Kelompok {i}',
                labels={'Total_Qty': 'Total Quantity', 'Item Name': 'Nama Produk'},
                color='Total_Qty',
                text='Total_Qty'
            )
            
            # Customize the bar chart
                        fig_bar.update_layout(
                height=500,
                yaxis={'categoryorder': 'total ascending'},
                showlegend=False,
                title_font_size=16,
                xaxis_title_font_size=14,
                yaxis_title_font_size=14
            )
            
            # Format text on bars
                        fig_bar.update_traces(
                texttemplate='%{text:,}',
                textposition='outside',
                textfont_size=10
            )
            
                        st.plotly_chart(fig_bar, use_container_width=True)
                        # Display cluster products
                        st.markdown(f"#### 📅 Daftar Produk di Kelompok {i}")
                        st.dataframe(cluster_data, use_container_width=True, hide_index=True)



                # Pastikan konsistensi cluster labels di seluruh analisis
                # Scatter plot sebagai referensi utama, semua tabel mengikuti
                st.session_state.df_result = df_result  # Make sure this is updated
                
                               # Time series analysis (if multiple months)
                if any('Qty_januari' in col for col in df_result.columns):
                    # st.markdown("<h3 class='sub-header'>📈 Analisis Tren Bulanan per Cluster</h3>", unsafe_allow_html=True)
                    
                    # Extract monthly data
                    months = [col.split('_')[1] for col in df_result.columns if col.startswith('Qty_')]
                    months.sort(key=lambda x: list(month_names.keys())[list(month_names.values()).index(x.capitalize())])
                    
                    # Create monthly trend dataframes
                    monthly_qty = pd.DataFrame()
                    monthly_revenue = pd.DataFrame()
                    
                    for month in months:
                        # Group by cluster and calculate sum for the month
                        month_qty = df_result.groupby('Cluster')[f'Qty_{month}'].sum().reset_index()
                        
                        
                        # Rename columns
                        month_qty.rename(columns={f'Qty_{month}': month}, inplace=True)
                        
                        
                        # If first month, create dataframe, otherwise merge
                        if monthly_qty.empty:
                            monthly_qty = month_qty
                            
                        else:
                            monthly_qty = monthly_qty.merge(month_qty, on='Cluster')
                            

                    st.markdown("""
                        <div class="summary-card" style="margin-top: 10px;">
                        <h4 s>📈 Perkembangan Penjualan Bulanan </h4>
                        <p>Grafik ini menunjukkan pola penjualan setiap kelompok dari bulan ke bulan.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Create tabs for Qty and Revenue
                        # Create line chart for Qty trends
                    qty_data = []
                        
                    for i, row in monthly_qty.iterrows():
                            cluster = row['Cluster']
                            for month in months:
                                qty_data.append({
                                    'Cluster': f'Cluster {cluster}',
                                    'Bulan': month.capitalize(),
                                    'Quantity': row[month]
                                })
                        
                    qty_trend_df = pd.DataFrame(qty_data)
                        
                        # Sort months chronologically
                    month_order = [m.lower() for m in list(month_names.values())]
                    qty_trend_df['Bulan'] = pd.Categorical(
                            qty_trend_df['Bulan'], 
                            categories=[m.capitalize() for m in month_order if m.lower() in months],
                            ordered=True
                        )
                    qty_trend_df = qty_trend_df.sort_values('Bulan')
                        
                        # Create line chart
                    fig_qty_trend = px.line(
                            qty_trend_df,
                            x='Bulan',
                            y='Quantity',
                            color='Cluster',
                            markers=True,
                            title='Perkembangan Penjualan Bulanan per Kelompok',
                            color_discrete_sequence=px.colors.qualitative.Bold
                        )
                        
                    fig_qty_trend.update_layout(
                            xaxis_title='Bulan',
                            yaxis_title='Total Penjualan',
                            legend_title='Cluster',
                            height=500
                        )
                        
                    st.plotly_chart(fig_qty_trend, use_container_width=True)
               
                # Final conclusions and recommendations
                st.markdown("<h3 class='sub-header' style='margin-bottom: 20px;'>💡 Rekomendasi Strategi</h3>", unsafe_allow_html=True)
                
                # Generate recommendations based on cluster characteristics
                st.markdown("""
                <div class="summary-card">
                    <p>
                        Berikut adalah beberapa rekomendasi strategi berdasarkan hasil analisis:
                    </p>
                    <ol>
                        <li><b>Manajemen Inventaris:</b> Fokuskan stok pada produk-produk di kelompok dengan tingkat penjualan tinggi dan stabil.</li>
                        <li><b>Strategi Promosi:</b> Lakukan promosi bundle untuk produk-produk dalam cluster yang sama karena memiliki pola pembelian serupa.</li>
                        <li><b>Analisis Lebih Lanjut:</b> Lakukan analisis lebih mendalam pada produk-produk di kelompok dengan kontribusi revenue tinggi namun jumlah produk sedikit.</li>
                        <li><b>Optimasi Harga:</b> Evaluasi strategi harga untuk produk-produk di kelompok dengan tren penjualan menurun.</li>
                        <li><b>Fokus pada Produk Strategis:</b> Prioritaskan resources pada Top 10 produk strategis yang memiliki kombinasi volume tinggi dan posisi cluster optimal.</li>
                    </ol>
                </div>
                """, unsafe_allow_html=True)
                
                # Add download button for full results
                               

# Fungsi untuk convert DataFrame ke Excel bytes
                def convert_df_to_excel(df):
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df.to_excel(writer, index=False, sheet_name='Clustering Results')
                    output.seek(0)
                    return output.getvalue()

                excel_data = convert_df_to_excel(df_result)

                st.download_button(
                    label="📥 Download Hasil Analisis",
                    data=excel_data,
                    file_name='apotek_clustering_results.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    key='download-excel',
                    type="primary"
                )
                


# Main app footer
st.markdown("""
<div style="text-align: center; margin-top: 3rem; padding: 1rem; background: linear-gradient(90deg, rgba(78, 84, 200, 0.1), rgba(143, 148, 251, 0.1)); border-radius: 10px;">
    <p style="color: #4E54C8; margin-bottom: 0.5rem;">💊 Analisis Data Apotek Paten  - K-Means Clustering</p>
    <p style="font-size: 0.8rem; color: #666;">Developed for Apotek Data Analysis</p>
</div>
""", unsafe_allow_html=True)